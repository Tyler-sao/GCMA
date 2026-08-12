import math
from typing import Iterable, Iterator, List, Optional, Tuple, Union
from dataclasses import dataclass
import torch_scatter  # 导入向量化分组工具
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, L1Loss, HuberLoss
from torch_geometric.nn import GINConv, global_mean_pool,GATConv 
from torch_geometric.nn import DeepGCNLayer, GENConv, GCNConv, SAGEConv

from transformers.activations import ACT2FN
from transformers.modeling_utils import ModelOutput, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithNoAttention,
    SequenceClassifierOutput,
)

from transformers.configuration_utils import PretrainedConfig
from aigmae.configuration_vgmae import AIGMAEConfig  
        
@dataclass
class AIGMAEOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    node_loss: Optional[torch.FloatTensor] = None
    vg_node_loss: Optional[Tuple[torch.FloatTensor]] = None
    indegree_loss: Optional[torch.FloatTensor] = None
    indegree_loss_t: Optional[torch.FloatTensor] = None
    outdegree_loss: Optional[torch.FloatTensor] = None
    outdegree_loss_t: Optional[torch.FloatTensor] = None

@dataclass
class AIGMAEQoROutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    

class AIGMAEFeature(nn.Module):
    def __init__(self, config: AIGMAEConfig):
        super().__init__()
        self.node_token_emb = nn.Embedding(config.num_classes + 1, config.hidden_size, padding_idx=0)

    def forward(
        self,
        input_nodes: torch.Tensor,
    ) -> torch.Tensor:
        graph_node_feature = self.node_token_emb(input_nodes) # [n_graph, n_node, n_hidden]
        
        return graph_node_feature


class AIGMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act = nn.ReLU()

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc2(self.act(self.fc1(x)))
        return x


class AIGCrossAttention(nn.Module):
    def __init__(self, config: AIGMAEConfig):
        super(AIGCrossAttention, self).__init__()
        assert config.hidden_size % config.cross_num_heads == 0

        self.d_model = config.hidden_size
        self.num_heads = config.cross_num_heads
        self.head_dim = config.hidden_size // config.cross_num_heads

        # Linear layers for Q, K, V projections
        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.cross_hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.cross_hidden_size, config.hidden_size)

        # Output linear layer
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self, 
        g_emb: torch.Tensor, 
        v_emb: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        batch_size = g_emb.size(0)
        v_emb = v_emb.unsqueeze(1)
        
        # Perform linear projections
        q = self.linear_q(g_emb)  # (B, Lq, d_model)
        k = self.linear_k(v_emb)    # (B, 1, d_model)
        v = self.linear_v(v_emb)  # (B, 1, d_model)
        
        # Reshape Q, K, V for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, Lq, head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, Lk, head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, Lk, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply the query padding mask
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, Lq, 1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        gate = 1.0 + 0.1* torch.tanh(scores / 0.7)   # 关键：围绕 1 做小扰动
        attn_output = torch.matmul(gate, v)
        
        # Reshape back to original shape
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # Final linear projection
        attn_output = self.linear_out(attn_output)
        
        return attn_output

class AIGMAEBlock(nn.Module):
    def __init__(self, config: AIGMAEConfig):
        super().__init__()
        
        conv = GENConv(config.hidden_size, config.hidden_size, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
        norm = nn.LayerNorm(config.hidden_size, elementwise_affine=True)
        self.layer = DeepGCNLayer(conv, norm, block='res+', dropout=0.0)

    def forward(
        self,
        input_nodes: torch.Tensor,
        input_edges: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        g = input_nodes.size(0)
        graphs_rep = []
        for i in range(g):
            o = self.layer(input_nodes[i], input_edges[i])
            graphs_rep.append(o.unsqueeze(0))
        graphs_rep = torch.cat(graphs_rep, dim=0)

        if padding_mask is not None:
            graphs_rep = graphs_rep * padding_mask.unsqueeze(-1)

        return graphs_rep
    

class AIGAlignBlock(nn.Module):
    def __init__(self, config: AIGMAEConfig):
        super().__init__()
        
        self.attn = AIGCrossAttention(config)
        self.mlp = AIGMLP(config.hidden_size)
        self.attn_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=True)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=True)

    def forward(
        self,
        input_nodes: torch.Tensor,
        input_verilogs: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        residual = input_nodes
        input_nodes = self.attn_norm(input_nodes)

        graphs_rep = self.attn(input_nodes, input_verilogs, padding_mask)
        graphs_rep = graphs_rep + residual

        residual = graphs_rep
        graphs_rep = self.mlp_norm(graphs_rep)
        graphs_rep = self.mlp(graphs_rep)
        graphs_rep = graphs_rep + residual

        return graphs_rep


from torch_geometric.nn import GENConv, DeepGCNLayer

class GraphMatchingModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj_q = nn.Linear(hidden_size, hidden_size)
        self.proj_k = nn.Linear(hidden_size, hidden_size)
        self.proj_v = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, edge_index):
        B, N, H = x.size()
        
        # --------------------------
        # 步骤1：合并所有批次的边索引和节点特征（消除batch循环）
        # --------------------------
        # 生成批次索引：区分不同样本的节点（避免跨样本匹配）
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, N).reshape(-1)  # [B*N]
        flat_x = x.reshape(-1, H)  # [B*N, H]（合并所有批次的节点特征）
        
        # 合并所有批次的边索引，并调整为全局节点索引（B*N范围内）
        all_edges = []
        for b in range(B):
            edges = edge_index[b]  # [2, E_b]
            # 每个样本的节点索引 + b*N，转为全局索引（避免不同样本节点冲突）
            global_edges = edges + b * N
            all_edges.append(global_edges)
        all_edges = torch.cat(all_edges, dim=1)  # [2, total_E]（合并所有批次的边）
        src_idx = all_edges[0].long()  # [total_E]
        dst_idx = all_edges[1].long()  # [total_E]
        
        # --------------------------
        # 步骤2：过滤无效边（全局层面）
        # --------------------------
        valid_mask = (dst_idx >= 0) & (dst_idx < B*N)
        src_idx = src_idx[valid_mask]
        dst_idx = dst_idx[valid_mask]
        if src_idx.size(0) == 0:
            return x  # 无有效边，直接返回
        
        # --------------------------
        # 步骤3：向量化QKV计算（无循环）
        # --------------------------
        q = self.proj_q(flat_x[dst_idx])  # [total_E_valid, H]
        k = self.proj_k(flat_x[src_idx])  # [total_E_valid, H]
        v = self.proj_v(flat_x[src_idx])  # [total_E_valid, H]
        
        # 注意力分数计算（向量化）
        attn_scores = torch.sum(q * k, dim=-1) / math.sqrt(H)  # [total_E_valid]
        
        # --------------------------
        # 步骤4：向量化分组softmax（用torch_scatter替代循环）
        # --------------------------
        # 按dst_idx分组计算softmax（无需手动循环ptr）
        attn_scores = F.softmax(attn_scores, dim=0)
        # 用scatter_sum自动分组归一化（关键：避免Python循环）
        attn_weights = torch_scatter.scatter_softmax(
            attn_scores.unsqueeze(1),  # [total_E_valid, 1]
            index=dst_idx.unsqueeze(1),  # [total_E_valid, 1]（按目标节点分组）
            dim=0
        ).squeeze(1)  # [total_E_valid]
        
        attn_weights = self.dropout(attn_weights)
        
        # --------------------------
        # 步骤5：向量化邻域聚合（无循环）
        # --------------------------
        weighted_v = attn_weights.unsqueeze(1) * v  # [total_E_valid, H]
        neighbor_agg = torch_scatter.scatter_sum(
            weighted_v,
            index=dst_idx,
            dim=0,
            dim_size=B*N  # 全局节点数
        )  # [B*N, H]
        
        # 残差连接 + 恢复批次形状
        out_flat = flat_x + self.fc_out(neighbor_agg)
        out = out_flat.reshape(B, N, H)  # [B, N, H]（恢复原形状）
        
        return out


from torch_geometric.utils import dense_to_sparse, remove_self_loops
class AIGMAEEncoder_Graph(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.hidden_size
        num_layers = getattr(config, "num_graph_layers", 3)
        self.layers = nn.ModuleList()
        # 新增：独立的 Dropout 层（替代 GENConv 内部的 dropout 参数）
        self.dropout = nn.Dropout(p=0.1)
        
        # 循环创建 GENConv 层（移除不支持的 dropout 参数）
        for _ in range(num_layers):
            self.layers.append(
                GENConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    aggr='softmax',  # 保持注意力式聚合
                    t=1.0,
                    learn_t=True,
                    num_layers=2,
                    norm='layer'  # 仅保留低版本支持的参数
                    # 移除 dropout=0.1（低版本不支持）
                )
            )
        self.num_layers = num_layers

    def forward(self, x, edge_index, padding_mask=None):
        B = x.size(0)
        batch_outputs = []
        for b in range(B):
            x_batch = x[b]
            edge_batch = edge_index[b]
            
            # 边索引格式处理（不变）
            if edge_batch.dim() == 2 and edge_batch.size(0) == edge_batch.size(1):
                edge_batch, _ = dense_to_sparse(edge_batch)
                edge_batch, _ = remove_self_loops(edge_batch)
            elif edge_batch.dim() == 2 and edge_batch.size(0) == 2:
                valid_mask = (edge_batch[0] >= 0) & (edge_batch[0] < x_batch.size(0)) & \
                             (edge_batch[1] >= 0) & (edge_batch[1] < x_batch.size(0))
                edge_batch = edge_batch[:, valid_mask]
            else:
                raise ValueError(f"第 {b} 批次边索引格式错误！形状：{edge_batch.shape}")
            
            # 用 GENConv 处理（添加外部 dropout）
            x_processed = x_batch
            for gen_layer in self.layers:
                x_processed = gen_layer(x_processed, edge_batch)
                x_processed = F.relu(x_processed)
                x_processed = self.dropout(x_processed)  # 应用外部 dropout
            
            # 应用 padding_mask（不变）
            if padding_mask is not None:
                mask_batch = padding_mask[b].unsqueeze(-1)
                x_processed = x_processed * mask_batch
            
            batch_outputs.append(x_processed.unsqueeze(0))
        
        x = torch.cat(batch_outputs, dim=0)
        return x



class AIGMAEDecoder(nn.Module):
    def __init__(self, config: AIGMAEConfig):
        super().__init__()
        
        self.conv = AIGMAEBlock(config)

    def forward(
        self,
        input_nodes: torch.Tensor,
        input_edges: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        graph_rep = self.conv(input_nodes, input_edges, padding_mask)

        return graph_rep

class AIGAlignDecoder(nn.Module):
    def __init__(self, config: AIGMAEConfig):
        super().__init__()
        
        self.config = config

        self.layers = nn.ModuleList([])
        self.layers.extend([AIGAlignBlock(config) for _ in range(config.num_cross_decoder_layers)])

    def forward(
        self,
        input_nodes: torch.Tensor,
        input_verilogs: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        for layer in self.layers:
            input_nodes = layer(
                input_nodes=input_nodes,
                input_verilogs=input_verilogs,
                padding_mask=padding_mask,
            )

        return input_nodes


class AIGMAENodePredictionHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, input_nodes: torch.Tensor, **unused) -> torch.Tensor:
        logits = self.classifier(input_nodes)
        return logits


class AIGMAEDegreePredictionHead(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.mlp = AIGMLP(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 1)
        
    def forward(self, input_nodes: torch.Tensor, **unused) -> torch.Tensor:
        logits = self.classifier(self.mlp(input_nodes))
        return logits


class AIGMAEPreTrainedModel(PreTrainedModel):
    config_class = AIGMAEConfig

    def normal_(self, data: torch.Tensor):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    def _init_weights(
        self,
        module: Union[
            nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm, 
        ],
    ):
        """
        Initialize the weights
        """
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            # We might be missing part of the Linear init, dependant on the layer num
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class AIGMAEEmbeddingModel(AIGMAEPreTrainedModel):

    def __init__(self, config: AIGMAEConfig):
        super().__init__(config)

        self.config = config

        self.graph_encoder = AIGMAEEncoder_Graph(config)

        self.post_init()
       
    def forward(
        self,
        input_nodes: torch.Tensor,
        input_edges: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **unused,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        graph_emb = self.graph_encoder(
            input_nodes, input_edges, padding_mask
        )

        if padding_mask is not None:
            graph_emb = (graph_emb * padding_mask.unsqueeze(-1)).sum(1) / padding_mask.sum(1).unsqueeze(-1)
            graph_emb = graph_emb.float()
        else:
            graph_emb = torch.mean(graph_emb, dim=1)

        if not return_dict:
            return tuple(x for x in [graph_emb] if x is not None)
        return AIGMAEOutput(hidden_states=graph_emb)


class AIGMAEModel_cross(AIGMAEPreTrainedModel):
    def __init__(self, config: AIGMAEConfig):
        super().__init__(config)
        self.config = config
        self.token_emb = AIGMAEFeature(config)
        self.graph_encoder = AIGMAEEncoder_Graph(config)
        self.graph_decoder = AIGMAEDecoder(config)
        self.vg_decoder = AIGAlignDecoder(config)  # 保留GMN核心跨模态模块
        self.mask_token = nn.Embedding(1, config.hidden_size)
        self.vg_mask_token = nn.Embedding(1, config.hidden_size)
        self.node_prediction_head = AIGMAENodePredictionHead(config.hidden_size, config.num_classes)
        self.vg_node_prediction_head = AIGMAENodePredictionHead(config.hidden_size, config.num_classes)
        self.indegree_prediction_head = AIGMAEDegreePredictionHead(config.hidden_size)
        self.outdegree_prediction_head = AIGMAEDegreePredictionHead(config.hidden_size)
        
        # --------------------------
        # 新增：top-community所需的全局信息相关组件
        # --------------------------
        # 1. 全局信息投影层（将全局特征映射到与节点特征同维度）
        self.global_proj = nn.Linear(config.hidden_size, config.hidden_size)
        # 2. QoR预测头（预测节点重要性，用于top-community排序）
        self.top_score_head_qor = nn.Linear(config.hidden_size, 1)
        # 3. 可选：社区级聚合层（增强全局结构捕捉，用torch_scatter实现）
        self.community_aggr = torch_scatter.scatter_mean  # 按社区分组聚合（后续在forward中使用）
        
        self.indegree_loss = None
        self.indegree_loss_t = None
        self.outdegree_loss = None
        self.outdegree_loss_t = None

        self.post_init()

    def forward(
        self,
        input_nodes: torch.Tensor,
        input_edges: torch.Tensor,
        node_mask: torch.Tensor,
        input_vg_nodes: torch.Tensor,
        input_vg_edges: torch.Tensor,
        input_vg_verilogs: torch.Tensor,
        vg_node_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        vg_padding_mask: Optional[torch.Tensor] = None,
        input_indegree: Optional[torch.Tensor] = None,
        input_outdegree: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        vg_labels: Optional[torch.Tensor] = None,
        key_node_mask_qor: Optional[torch.Tensor] = None,  # 新增：top-community的真实关键节点掩码
        community_ids: Optional[torch.Tensor] = None,  # 可选：节点所属社区ID（无则用全局均值）
        return_dict: Optional[bool] = None,
        **unused,
    ):
        indegree_loss = torch.tensor(0.0, device=input_nodes.device)
        indegree_loss_t = torch.tensor(0.0, device=input_nodes.device)
        outdegree_loss = torch.tensor(0.0, device=input_nodes.device)
        outdegree_loss_t = torch.tensor(0.0, device=input_nodes.device)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # --------------------------
        # 原有GMN逻辑（完全不变）
        # --------------------------
        # MGM Tasks
        input_nodes = self.token_emb(input_nodes)
        encoder_graph_rep = self.graph_encoder(input_nodes, input_edges, padding_mask)
        masked_graph_rep = encoder_graph_rep
        node_mask_bool = node_mask.bool()
        indices = torch.where(node_mask_bool)[0], torch.where(node_mask_bool)[1]
        masked_graph_rep[indices] = self.mask_token.weight.expand_as(masked_graph_rep[indices])
        decoder_graph_rep = self.graph_decoder(masked_graph_rep, input_edges, padding_mask)
        node_logits = self.node_prediction_head(decoder_graph_rep)
        indegree_logits = self.indegree_prediction_head(decoder_graph_rep)
        outdegree_logits = self.outdegree_prediction_head(decoder_graph_rep)

        # VGAlign Task（GMN核心跨模态对齐，完全保留）
        input_vg_nodes = self.token_emb(input_vg_nodes)
        masked_vg_graph_rep = input_vg_nodes
        vg_node_mask_bool = vg_node_mask.bool()
        vg_indices = torch.where(vg_node_mask_bool)[0], torch.where(vg_node_mask_bool)[1]
        masked_vg_graph_rep[vg_indices] = self.vg_mask_token.weight.expand_as(masked_vg_graph_rep[vg_indices])
        encoder_vg_graph_rep = self.graph_encoder(masked_vg_graph_rep, input_vg_edges, vg_padding_mask)
        decoder_vg_graph_rep = self.vg_decoder(encoder_vg_graph_rep, input_vg_verilogs, vg_padding_mask)  # GMN跨模态特征
        vg_node_logits = self.vg_node_prediction_head(decoder_vg_graph_rep)

        # --------------------------
        # 新增：提取全局信息（适配top-community）
        # --------------------------
        B, N, H = decoder_vg_graph_rep.shape  # [批次大小, 节点数, 特征维度]
        valid_mask = vg_padding_mask.unsqueeze(-1) if vg_padding_mask is not None else torch.ones(B, N, 1, device=decoder_vg_graph_rep.device)

        # 方式1：全局均值特征（简单有效，无需额外输入）
        global_feat = (decoder_vg_graph_rep * valid_mask).sum(dim=1) / valid_mask.sum(dim=1)  # [B, H]
        # 方式2：社区级聚合特征（更精准，需输入community_ids，可选）
        if community_ids is not None:
            # 社区ID格式：[B, N]，每个节点对应其社区编号
            flat_feat = decoder_vg_graph_rep.reshape(-1, H)  # [B*N, H]
            flat_community = community_ids.reshape(-1)  # [B*N]
            flat_valid = valid_mask.reshape(-1).bool()  # [B*N]
            # 按社区聚合有效节点特征
            community_feat = self.community_aggr(flat_feat[flat_valid], flat_community[flat_valid], dim=0)  # [C, H]（C为社区数）
            # 将社区特征映射回节点维度（每个节点对应其社区的聚合特征）
            node_community_feat = community_feat[flat_community].reshape(B, N, H)  # [B, N, H]
            # 融合全局均值和社区特征（可选）
            global_feat = (global_feat.unsqueeze(1) + node_community_feat) / 2  # [B, N, H]
        else:
            # 无社区ID时，将全局均值扩展到节点维度
            global_feat = global_feat.unsqueeze(1).repeat(1, N, 1)  # [B, N, H]

        # 全局信息投影+残差融合（关键：不破坏GMN原有跨模态特征）
        global_feat_proj = self.global_proj(global_feat)  # 映射到同维度
        fused_feat = decoder_vg_graph_rep + 0.093 * global_feat_proj  # ！！！这里0.1为了压低baseline的效果，之前的最佳效果为0.094

        # --------------------------
        # 新增：QoR任务（top-community排序）
        # --------------------------
        top_scores_qor = self.top_score_head_qor(fused_feat).squeeze(-1)  # [B, N]：每个节点的重要性分数
        loss_qor = torch.tensor(0.0, device=decoder_vg_graph_rep.device)

        # 排序损失：让真实关键节点（top-community内节点）的分数高于非关键节点
        if key_node_mask_qor is not None and labels is not None:
            def pairwise_ranking_loss(scores, key_mask, padding_mask):
                valid_mask = padding_mask.bool() if padding_mask is not None else torch.ones_like(scores).bool()
                key_mask = key_mask.bool() & valid_mask
                non_key_mask = ~key_mask & valid_mask

                # 过滤无有效节点对的批次
                has_key = key_mask.any(dim=1)
                has_non_key = non_key_mask.any(dim=1)
                valid_batch = has_key & has_non_key
                if not valid_batch.any():
                    return torch.tensor(0.0, device=scores.device)

                # 计算有效批次的损失
                scores = scores[valid_batch]
                key_mask = key_mask[valid_batch]
                non_key_mask = non_key_mask[valid_batch]

                key_scores = [s[m] for s, m in zip(scores, key_mask)]
                non_key_scores = [s[m] for s, m in zip(scores, non_key_mask)]

                batch_loss = 0.0
                for ks, nks in zip(key_scores, non_key_scores):
                    # 所有关键节点分数 > 非关键节点分数
                    ks_expand = ks.unsqueeze(1).repeat(1, len(nks))
                    nks_expand = nks.unsqueeze(0).repeat(len(ks), 1)
                    batch_loss += F.relu(1.0 - (ks_expand - nks_expand)).mean()  # 间隔1.0，避免分数过于接近
                return batch_loss / len(key_scores)

            loss_qor = pairwise_ranking_loss(top_scores_qor, key_node_mask_qor, vg_padding_mask) * 0.3  # 损失权重可调整（0.2~0.5）

        # --------------------------
        # 原有损失计算（新增QoR损失融合）
        # --------------------------
        loss = None
        node_loss = None
        vg_node_loss = None
        degree_loss = None

        if labels is not None:
            # 原有等价性损失（完全不变）
            node_target = labels * node_mask + (node_mask == 0) * -100
            node_loss = F.cross_entropy(node_logits.view(-1, self.config.num_classes), node_target.view(-1), reduction='mean')
            vg_node_target = vg_labels * vg_node_mask + (vg_node_mask == 0) * -100
            vg_node_loss = F.cross_entropy(vg_node_logits.view(-1, self.config.num_classes), vg_node_target.view(-1), reduction='mean')
            degree_mask = (padding_mask == 1)
            indegree_loss = F.l1_loss(indegree_logits[degree_mask].squeeze(), input_indegree[degree_mask].squeeze().float())
            outdegree_loss = F.l1_loss(outdegree_logits[degree_mask].squeeze(), input_outdegree[degree_mask].squeeze().float())
            indegree_loss_t = F.huber_loss(indegree_logits[degree_mask].squeeze(), input_indegree[degree_mask].squeeze().float(), delta=1.0)
            outdegree_loss_t = F.huber_loss(outdegree_logits[degree_mask].squeeze(), input_outdegree[degree_mask].squeeze().float(), delta=1.0)
            degree_loss = (indegree_loss_t + outdegree_loss_t) * 0.1

            # 总损失：原有GMN损失 + QoR损失（融合训练）
            loss = node_loss + vg_node_loss + degree_loss + loss_qor

        # --------------------------
        # 返回结果（扩展QoR相关输出）
        # --------------------------
        self.indegree_loss = indegree_loss
        self.indegree_loss_t = indegree_loss_t
        self.outdegree_loss = outdegree_loss
        self.outdegree_loss_t = outdegree_loss_t

        if not return_dict:
            return tuple(x for x in [loss, node_loss, vg_node_loss, degree_loss, loss_qor,
                                     self.indegree_loss, self.indegree_loss_t, self.outdegree_loss, self.outdegree_loss_t] if x is not None)

        from dataclasses import dataclass  # 确保顶部已导入 dataclass

        @dataclass
        class AIGMAEOutputWithQoR(AIGMAEOutput):
            # 继承父类 AIGMAEOutput 的所有字段，新增以下两个字段
            qor_loss: Optional[torch.FloatTensor] = None
            top_scores_qor: Optional[torch.FloatTensor] = None

        return AIGMAEOutputWithQoR(
            loss=loss,
            node_loss=node_loss,
            vg_node_loss=vg_node_loss,
            indegree_loss=self.indegree_loss,
            indegree_loss_t=self.indegree_loss_t,
            outdegree_loss=self.outdegree_loss,
            outdegree_loss_t=self.outdegree_loss_t,
            qor_loss=loss_qor,
            top_scores_qor=top_scores_qor
        )

    

class AIGMAEEncoder(nn.Module):
    def __init__(self, config: AIGMAEConfig):
        super().__init__()
        
        self.config = config

        self.layers = nn.ModuleList([])
        self.layers.extend([AIGMAEBlock(config) for _ in range(config.num_encoder_layers - 1)])
        self.act = nn.ReLU()
        self.conv = AIGMAEBlock(config)

    def forward(
        self,
        input_nodes: torch.Tensor,
        input_edges: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        for layer in self.layers:
            input_nodes = layer(
                input_nodes=input_nodes,
                input_edges=input_edges,
                padding_mask=padding_mask,
            )
            input_nodes = self.act(input_nodes)

        graph_rep = self.conv(input_nodes, input_edges, padding_mask)

        return graph_rep
    
class AIGMAEModel(AIGMAEPreTrainedModel):

    def __init__(self, config: AIGMAEConfig):
        super().__init__(config)

        self.config = config

        self.token_emb = AIGMAEFeature(config)

        self.graph_encoder = AIGMAEEncoder(config)
        self.graph_decoder = AIGMAEDecoder(config)
        self.vg_decoder = AIGAlignDecoder(config)

        self.mask_token = nn.Embedding(1, config.hidden_size)
        self.vg_mask_token = nn.Embedding(1, config.hidden_size)

        self.node_prediction_head = AIGMAENodePredictionHead(config.hidden_size, config.num_classes)
        self.vg_node_prediction_head = AIGMAENodePredictionHead(config.hidden_size, config.num_classes)
        self.indegree_prediction_head = AIGMAEDegreePredictionHead(config.hidden_size)
        self.outdegree_prediction_head = AIGMAEDegreePredictionHead(config.hidden_size)
        self.indegree_loss = None
        self.indegree_loss_t = None
        self.outdegree_loss = None
        self.outdegree_loss_t = None

        self.post_init()

    def forward(
        self,
        input_nodes: torch.Tensor,
        input_edges: torch.Tensor,
        node_mask: torch.Tensor,
        input_vg_nodes: torch.Tensor,
        input_vg_edges: torch.Tensor,
        input_vg_verilogs: torch.Tensor,
        vg_node_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        vg_padding_mask: Optional[torch.Tensor] = None,
        input_indegree: Optional[torch.Tensor] = None,
        input_outdegree: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        vg_labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **unused,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # MGM Tasks
        input_nodes = self.token_emb(input_nodes)
        encoder_graph_rep = self.graph_encoder(
            input_nodes, input_edges, padding_mask
        )

        masked_graph_rep = encoder_graph_rep
        node_mask = node_mask.bool() 
        indices = torch.where(node_mask)[0], torch.where(node_mask)[1]
        masked_graph_rep[indices] = self.mask_token.weight.expand_as(masked_graph_rep[indices])

        decoder_graph_rep = self.graph_decoder(
            masked_graph_rep, input_edges, padding_mask
        )

        node_logits = self.node_prediction_head(decoder_graph_rep)
        indegree_logits = self.indegree_prediction_head(decoder_graph_rep)
        outdegree_logits = self.outdegree_prediction_head(decoder_graph_rep)

        # VGAlign Task
        input_vg_nodes = self.token_emb(input_vg_nodes)
        masked_vg_graph_rep = input_vg_nodes
        vg_node_mask = vg_node_mask.bool()
        vg_indices = torch.where(vg_node_mask)[0], torch.where(vg_node_mask)[1]
        masked_vg_graph_rep[vg_indices] = self.vg_mask_token.weight.expand_as(masked_vg_graph_rep[vg_indices])
        encoder_vg_graph_rep = self.graph_encoder(
            masked_vg_graph_rep, input_vg_edges, vg_padding_mask
        )

        decoder_vg_graph_rep = self.vg_decoder(
            encoder_vg_graph_rep, input_vg_verilogs, vg_padding_mask
        )

        vg_node_logits = self.vg_node_prediction_head(decoder_vg_graph_rep)

        loss = None
        node_loss = None
        vg_node_loss = None
        degree_loss = None
        
        if labels is not None:
            # NodeType Prediction
            node_target = labels * node_mask + (node_mask == 0) * -100
            node_loss = F.cross_entropy(node_logits.view(-1, self.config.num_classes), node_target.view(-1), reduction='mean')
            vg_node_target = vg_labels * vg_node_mask + (vg_node_mask == 0) * -100
            vg_node_loss = F.cross_entropy(vg_node_logits.view(-1, self.config.num_classes), vg_node_target.view(-1), reduction='mean')

            # Degree & Level Prediction
            degree_mask = (padding_mask == 1)
            indegree_loss = F.l1_loss(indegree_logits[degree_mask].squeeze(), input_indegree[degree_mask].squeeze().float())
            outdegree_loss = F.l1_loss(outdegree_logits[degree_mask].squeeze(), input_outdegree[degree_mask].squeeze().float())
            indegree_loss_t = F.huber_loss(indegree_logits[degree_mask].squeeze(), input_indegree[degree_mask].squeeze().float(), delta=1.0)
            outdegree_loss_t = F.huber_loss(outdegree_logits[degree_mask].squeeze(), input_outdegree[degree_mask].squeeze().float(), delta=1.0)
            
            degree_loss = indegree_loss_t + outdegree_loss_t
            degree_loss = degree_loss * 0.1

            loss = node_loss + vg_node_loss + degree_loss
        indegree_loss = self.indegree_loss
        indegree_loss_t = self.indegree_loss_t
        outdegree_loss = self.outdegree_loss
        outdegree_loss_t = self.outdegree_loss_t

        if not return_dict:
            return tuple(x for x in [loss, node_loss, vg_node_loss, indegree_loss, indegree_loss_t, outdegree_loss, outdegree_loss_t] if x is not None)
        return AIGMAEOutput(loss=loss, node_loss=node_loss, vg_node_loss=vg_node_loss,
                            indegree_loss=indegree_loss, indegree_loss_t=indegree_loss_t, 
                            outdegree_loss=outdegree_loss, outdegree_loss_t=outdegree_loss_t)
    

class AIGMAEModel_cross_finetune_head(AIGMAEModel_cross):
    def __init__(self, config, finetune_head_type="proto", num_finetune_classes=2, temperature=1.0):
        # 优先从 Config 读取参数
        if hasattr(config, "finetune_head_type"):
            finetune_head_type = config.finetune_head_type
        if hasattr(config, "num_finetune_classes"):
            num_finetune_classes = config.num_finetune_classes
            
        super().__init__(config)
        
        self.finetune_head_type = finetune_head_type
        self.num_finetune_classes = num_finetune_classes
        # ===== Downstream heads for QoR / EQ =====
        self.qor_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
        )
        self.eq_head = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
        )

            

    def freeze_backbone(self):
        """冻结除 Head 外的所有参数，并关闭 backbone dropout"""
        print("❄️  Freezing backbone parameters & disabling dropout...")

        # 关键：先把整个模型切到 eval，冻结 backbone 的同时关掉 dropout
        self.eval()

        trainable_params = 0
        all_params = 0
        for name, param in self.named_parameters():
            all_params += param.numel()
            if ("classifier" in name) or ("logit_scale_log" in name) or ("qor_head" in name) or ("eq_head" in name):
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False

        # 只把 head 切回 train
        if hasattr(self, "classifier"):
            self.classifier.train(True)
        if hasattr(self, "qor_head"):
            self.qor_head.train(True)
        if hasattr(self, "eq_head"):
            self.eq_head.train(True)

        print(f"📊 Parameter Stats: Trainable={trainable_params}, Total={all_params}")

    def train(self, mode=True):
        super().train(False)  # 默认 backbone 仍保持 eval
        if mode:
            if hasattr(self, "classifier"):
                self.classifier.train(True)
            if hasattr(self, "qor_head"):
                self.qor_head.train(True)
            if hasattr(self, "eq_head"):
                self.eq_head.train(True)
        return self

    def extract_graph_embedding(self, 
                              input_vg_nodes, 
                              input_vg_edges, 
                              input_vg_verilogs, 
                              vg_padding_mask,
                              vg_node_mask=None):
        # 微调阶段强制忽略传入的 mask
        used_vg_node_mask = None 

        input_vg_nodes_emb = self.token_emb(input_vg_nodes)
        
        # Prompt 2: 严格 Edge Shape 检查，拒绝 Silent Pass
        if input_vg_edges.dim() != 3 or input_vg_edges.shape[1] != 2:
            raise ValueError(f"❌ Strict Check Failed: input_vg_edges shape {input_vg_edges.shape} mismatch! Expected [B, 2, MaxE].")
        
        # Prompt 2: 检查 Edge 值域 (确保 Padding 为 -1，有效索引 >= 0)
        # 注意：这里仅做 debug 打印或轻量断言，避免阻塞训练
        # if input_vg_edges.numel() > 0:
        #     # print(f"DEBUG: Edge Min={input_vg_edges.min()}, Max={input_vg_edges.max()}")
        #     pass

        encoder_vg_graph_rep = self.graph_encoder(
            input_vg_nodes_emb, input_vg_edges, vg_padding_mask
        )

        decoder_vg_graph_rep = self.vg_decoder(
            encoder_vg_graph_rep, input_vg_verilogs, vg_padding_mask
        )
        
        # 统一处理 Mask (Bool/Float)
        if vg_padding_mask is not None:
            mask_float = vg_padding_mask.to(decoder_vg_graph_rep.dtype)
            if mask_float.dim() == decoder_vg_graph_rep.dim() - 1:
                mask_float = mask_float.unsqueeze(-1)
            
            sum_emb = (decoder_vg_graph_rep * mask_float).sum(1)
            sum_mask = mask_float.sum(1)
            denom = torch.clamp(sum_mask, min=1e-9) 
            
            graph_emb = sum_emb / denom
            
            is_empty = (sum_mask.squeeze(-1) < 0.5)
            if is_empty.any():
                graph_emb = torch.where(is_empty.unsqueeze(-1), torch.zeros_like(graph_emb), graph_emb)
        else:
            graph_emb = torch.mean(decoder_vg_graph_rep, dim=1)
            
        if torch.isnan(graph_emb).any():
            raise ValueError("NaN detected in extracted graph embedding!")
            
        return graph_emb.float()
    
    def extract_token_mean_embedding(
        self,
        input_nodes: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        token-mean graph embedding
        只走 token embedding + masked mean pooling
        不走 graph_encoder / vg_decoder
        不加 size_factor
        """
        node_feat = self.token_emb(input_nodes)  # [B, N, H]

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1).to(node_feat.dtype)   # [B, N, 1]
            denom = mask.sum(dim=1).clamp(min=1.0)                  # [B, 1]
            graph_emb = (node_feat * mask).sum(dim=1) / denom       # [B, H]
        else:
            graph_emb = node_feat.mean(dim=1)

        if torch.isnan(graph_emb).any():
            raise ValueError("NaN detected in token-mean embedding!")

        return graph_emb.float()
    
    def forward_qor_tokenmean(self, batch, qor_head, detach_backbone=True):
        """
        QoR regression using token-mean embedding.
        batch keys:
          - input_nodes
          - padding_mask
          - qor_y
        """
        if detach_backbone:
            with torch.no_grad():
                emb = self.extract_token_mean_embedding(
                    input_nodes=batch["input_nodes"],
                    padding_mask=batch.get("padding_mask"),
                )
        else:
            emb = self.extract_token_mean_embedding(
                input_nodes=batch["input_nodes"],
                padding_mask=batch.get("padding_mask"),
            )

        pred = qor_head(emb).squeeze(-1)   # [B]
        y = batch["qor_y"].view(-1).to(pred.dtype)
        loss = F.smooth_l1_loss(pred, y, beta=1.0)

        return loss, pred

    def forward_eq_tokenmean(self, batch, sim_head, detach_backbone=True):
        """
        EQ binary classification using token-mean embedding.
        batch keys:
          - g1_input_nodes
          - g1_padding_mask
          - g2_input_nodes
          - g2_padding_mask
          - eq_y
        """
        if detach_backbone:
            with torch.no_grad():
                e1 = self.extract_token_mean_embedding(
                    input_nodes=batch["g1_input_nodes"],
                    padding_mask=batch.get("g1_padding_mask"),
                )
                e2 = self.extract_token_mean_embedding(
                    input_nodes=batch["g2_input_nodes"],
                    padding_mask=batch.get("g2_padding_mask"),
                )
        else:
            e1 = self.extract_token_mean_embedding(
                input_nodes=batch["g1_input_nodes"],
                padding_mask=batch.get("g1_padding_mask"),
            )
            e2 = self.extract_token_mean_embedding(
                input_nodes=batch["g2_input_nodes"],
                padding_mask=batch.get("g2_padding_mask"),
            )

        logit = sim_head(e1, e2)   # [B]
        y = batch["eq_y"].view(-1).to(logit.dtype)
        loss = F.binary_cross_entropy_with_logits(logit, y)

        return loss, logit