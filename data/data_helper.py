import os
import os.path as osp

import math
import pandas as pd
from zipfile import ZipFile

import torch
from torch.utils.data import Dataset
from torch import Tensor

import torch_geometric
from torch_geometric.utils import degree

import copy
import random
import numpy as np

from typing import Any, Dict, List, Mapping

import pickle

def getMeanAndVariance(targetList):
    return np.mean(np.array(targetList)),np.std(np.array(targetList))


class CustomDataCollator:
    def __init__(self, mgm_mask_ratio=0.5, align_mask_ratio=0.5, cross_hidden_size=3584):
        self.mgm_mask_ratio = mgm_mask_ratio
        self.align_mask_ratio = align_mask_ratio
        self.cross_hidden_size = cross_hidden_size

    def __call__(self, inputs) -> Dict[str, Any]:
        # Unpack input data
        graphs = [i[0] for i in inputs]
        vgraphs = [i[1] for i in inputs]
        batch_size = len(graphs)
        batch = {}

        # ----------------- MGM Data Processing -----------------
        max_node_num = max((g['input_nodes'].shape[0] for g in graphs if 'input_nodes' in g), default=0)
        max_edge_num = max((g['input_edges'].shape[-1] for g in graphs if 'input_edges' in g), default=0)

        batch["input_nodes"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["input_edges"] = torch.zeros(batch_size, 2, max_edge_num, dtype=torch.long)
        batch["node_mask"] = torch.zeros(batch_size, max_node_num, dtype=torch.bool)
        batch["padding_mask"] = torch.zeros(batch_size, max_node_num, dtype=torch.bool)
        batch["input_indegree"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["input_outdegree"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["labels"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)

        for idx, g in enumerate(graphs):
            input_nodes = g.get("input_nodes", torch.tensor([], dtype=torch.long))
            node_num = input_nodes.shape[0]
            
            edge_index = g.get("input_edges", torch.empty((2, 0), dtype=torch.long))
            edge_index = edge_index if edge_index.dim() == 2 and edge_index.shape[0] == 2 else torch.empty((2, 0), dtype=torch.long)
            edge_num = edge_index.shape[1] if edge_index.numel() > 0 else 0

            # Calculate degrees
            indegree = degree(edge_index[0], num_nodes=node_num) if edge_num > 0 and node_num > 0 else torch.zeros(node_num, dtype=torch.long)
            outdegree = degree(edge_index[1], num_nodes=node_num) if edge_num > 0 and node_num > 0 else torch.zeros(node_num, dtype=torch.long)

            # Process Edges
            if edge_num > 0 and max_edge_num > 0:
                valid_edge_num = min(edge_num, max_edge_num)
                # Ensure edge indices point to valid nodes
                safe_edges = edge_index[:, :valid_edge_num].clamp(0, node_num - 1)
                batch["input_edges"][idx, :, :valid_edge_num] = safe_edges
                if valid_edge_num < max_edge_num:
                    batch["input_edges"][idx, :, valid_edge_num:] = 0

            # Process Nodes
            if node_num > 0 and max_node_num > 0:
                # FIX: Do NOT clamp input_nodes (types) to node_num!
                # input_nodes contains categorical types (0,1,2,3...), not indices.
                safe_nodes = input_nodes.clone() 
                
                # Shift by +1 for embedding padding (0 is reserved for padding)
                batch["input_nodes"][idx, :node_num] = safe_nodes.view(-1) + 1
                batch["padding_mask"][idx, :node_num] = True
                batch["input_indegree"][idx, :node_num] = indegree
                batch["input_outdegree"][idx, :node_num] = outdegree
                # Labels use original type values (0,1,2,3...)
                batch["labels"][idx, :node_num] = safe_nodes.squeeze(-1)

                # Generate Mask
                masked_node_num = max(1, int(node_num * self.mgm_mask_ratio))
                masked_node_num = min(masked_node_num, node_num)
                perm = torch.randperm(node_num)[:masked_node_num]
                batch["node_mask"][idx, perm] = True

            if node_num > 0 and batch["node_mask"][idx].sum() == 0:
                batch["node_mask"][idx, 0] = True

        # ----------------- VGAlign Data Processing -----------------
        max_vg_node_num = max((g['input_nodes'].shape[0] for g in vgraphs if 'input_nodes' in g), default=0)
        max_vg_edge_num = max((g['input_edges'].shape[-1] for g in vgraphs if 'input_edges' in g), default=0)

        batch["input_vg_nodes"] = torch.zeros(batch_size, max_vg_node_num, dtype=torch.long)
        batch["input_vg_edges"] = torch.zeros(batch_size, 2, max_vg_edge_num, dtype=torch.long)
        batch["input_vg_verilogs"] = torch.zeros(batch_size, self.cross_hidden_size, dtype=torch.float)
        batch["vg_node_mask"] = torch.zeros(batch_size, max_vg_node_num, dtype=torch.bool)
        batch["vg_padding_mask"] = torch.zeros(batch_size, max_vg_node_num, dtype=torch.bool)
        batch["vg_labels"] = torch.zeros(batch_size, max_vg_node_num, dtype=torch.long)

        for idx, g in enumerate(vgraphs):
            input_nodes = g.get("input_nodes", torch.tensor([], dtype=torch.long))
            if input_nodes.dim() > 1:
                input_nodes = input_nodes.squeeze()
            if input_nodes.dim() == 0:
                input_nodes = torch.tensor([], dtype=torch.long)
            node_num = input_nodes.shape[0]

            edge_index = g.get("input_edges", torch.empty((2, 0), dtype=torch.long))
            edge_index = edge_index if edge_index.dim() == 2 and edge_index.shape[0] == 2 else torch.empty((2, 0), dtype=torch.long)
            edge_num = edge_index.shape[1] if edge_index.numel() > 0 else 0

            # Process Edges
            if edge_num > 0 and max_vg_edge_num > 0:
                valid_edge_num = min(edge_num, max_vg_edge_num)
                # Clamp edge indices to be within valid node range
                batch["input_vg_edges"][idx, 0, :valid_edge_num] = edge_index[0, :valid_edge_num].clamp(0, node_num - 1)
                batch["input_vg_edges"][idx, 1, :valid_edge_num] = edge_index[1, :valid_edge_num].clamp(0, node_num - 1)
                if valid_edge_num < max_vg_edge_num:
                    batch["input_vg_edges"][idx, :, valid_edge_num:] = 0

            # Process Nodes
            if node_num > 0 and max_vg_node_num > 0:
                # FIX: Do NOT clamp input_nodes (types) to node_num!
                safe_nodes = input_nodes.clone()

                # Shift by +1 for embedding padding
                batch["input_vg_nodes"][idx, :node_num] = safe_nodes.view(-1) + 1
                batch["vg_padding_mask"][idx, :node_num] = True
                # Labels use original values
                batch["vg_labels"][idx, :node_num] = safe_nodes

                # Generate Mask
                masked_node_num = max(1, int(node_num * self.align_mask_ratio))
                masked_node_num = min(masked_node_num, node_num)
                perm = torch.randperm(node_num)[:masked_node_num]
                batch["vg_node_mask"][idx, perm] = True

            if node_num > 0 and batch["vg_node_mask"][idx].sum() == 0:
                batch["vg_node_mask"][idx, 0] = True

            # Process Verilog
            verilog = g.get('input_verilogs', torch.zeros(self.cross_hidden_size, dtype=torch.float))
            if verilog.numel() != self.cross_hidden_size:
                verilog = torch.zeros(self.cross_hidden_size, dtype=torch.float)
            batch["input_vg_verilogs"][idx] = verilog

        return batch
class CustomDataset(Dataset):
    def __init__(self, dataset: List[Dataset]):
        super(CustomDataset, self).__init__()
        self.ds_mgm = dataset[0]
        self.ds_vgalign = dataset[1]
        self.len_mgm = len(self.ds_mgm)
        self.len_vgalign = len(self.ds_vgalign)
        self.total_len = max(self.len_mgm, self.len_vgalign)
        
    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        """
        Problems:
        If training for >1 epoch in unified mode, the same generative & embedding samples will 
        always be in the same batch as the same index is used for both datasets.
        Solution:
        Don't train for >1 epoch by duplicating the dataset you want to repeat in the folder.
        Upon loading, each dataset is shuffled so indices will be different.
        """
        mgm, vgalign = None, None

        if item >= self.len_mgm:
            random_item = random.randint(0, self.len_mgm - 1)
            mgm = self.ds_mgm[random_item]
        else:
            mgm = self.ds_mgm[item]

        if item >= self.len_vgalign:
            random_item = random.randint(0, self.len_vgalign - 1)
            vgalign = self.ds_vgalign[random_item]
        else:
            vgalign = self.ds_vgalign[item]

        return mgm, vgalign
    

class GraphDataCollator:
    def __init__(self, mask_ratio=0.5):
        self.mask_ratio = mask_ratio

    def __call__(self, graphs: List[dict]) -> Dict[str, Any]:

        max_node_num = max(g['input_nodes'].shape[0] for g in graphs)
        max_edge_num = max((g['input_edges'].shape[-1]) for g in graphs)
        batch_size = len(graphs)
        batch = {}

        batch["input_nodes"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["input_edges"] = torch.zeros(batch_size, 2, max_edge_num, dtype=torch.long)
        batch["node_mask"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["padding_mask"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)

        # SSL Labels
        batch["input_indegree"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["input_outdegree"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["labels"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)

        # print(self.mask_ratio)
        mask_ratio = self.mask_ratio

        for idx, g in enumerate(graphs):
            input_nodes = g["input_nodes"]
            node_num = input_nodes.shape[0]
            masked_node_num = int(node_num * mask_ratio)

            edge_index = copy.deepcopy(g['input_edges'])

            indegree = degree(edge_index[0], num_nodes=node_num)
            outdegree = degree(edge_index[1], num_nodes=node_num)

            edge_num = edge_index.shape[1]
            batch["input_edges"][idx, 0, :edge_num] = edge_index[0]
            batch["input_edges"][idx, 0, edge_num:] = edge_index[0][0]
            batch["input_edges"][idx, 1, :edge_num] = edge_index[1]
            batch["input_edges"][idx, 1, edge_num:] = edge_index[1][0]

            batch["input_nodes"][idx, :node_num] = input_nodes + 1
            batch["node_mask"][idx][torch.randperm(node_num)[:masked_node_num]] = 1
            batch["padding_mask"][idx, :node_num] = 1

            batch["input_indegree"][idx, :node_num] = indegree
            batch["input_outdegree"][idx, :node_num] = outdegree
            batch["labels"][idx, :node_num] = input_nodes
        
        return batch


class GraphDataset(Dataset):
    """安全版 GraphDataset，文件缺失或加载失败会跳过"""
    def __init__(self, root_path: str, data_path: str):
        super().__init__()
        import pandas as pd
        fileDF = pd.read_csv(data_path)
        self.root_path = root_path
        self.all_files = fileDF['fileName'].tolist()
        self.files = []

        # 预先检查文件是否存在，存在才加入列表
        for f in self.all_files:
            filePathArchive = osp.join(self.root_path, f)
            filePathName = osp.basename(osp.splitext(filePathArchive)[0])
            if osp.exists(filePathArchive):
                try:
                    with ZipFile(filePathArchive) as myzip:
                        if filePathName in myzip.namelist():
                            self.files.append(f)
                        else:
                            print(f"[Warning] {filePathName} 不在压缩包中，跳过")
                except:
                    print(f"[Warning] 打开 {filePathArchive} 失败，跳过")
            else:
                print(f"[Warning] 文件 {filePathArchive} 不存在，跳过")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filePathArchive = osp.join(self.root_path, self.files[i])
        filePathName = osp.basename(osp.splitext(filePathArchive)[0])
        with ZipFile(filePathArchive) as myzip:
            with myzip.open(filePathName) as myfile:
                g = torch.load(myfile, map_location="cpu")

        # 处理空边：确保 edge_index 始终为 (2, E) 形状（即使 E=0）
        if g.edge_index.dim() != 2 or g.edge_index.shape[0] != 2:
            g.edge_index = torch.empty((2, 0), dtype=torch.long)

        # ---------------------- 1. 修复：统一原始 node_type 维度为 (N, 1) 二维 ----------------------
        # 处理 g.node_type：无论原始是1维/2维，均转为 (N, 1) 格式
        if hasattr(g, 'node_type'):
            # 情况1：原始 node_type 是1维（如 [0,1,2]）→ 转为 (N,1)
            if g.node_type.dim() == 1:
                raw_node_type = g.node_type.view(-1, 1)
            # 情况2：原始 node_type 是2维但列数≠1（如 (N,2)）→ 取第一列并转为 (N,1)
            elif g.node_type.dim() == 2:
                if g.node_type.shape[1] != 1:
                    raw_node_type = g.node_type[:, 0].view(-1, 1)
                else:
                    raw_node_type = g.node_type  # 已符合 (N,1)，直接使用
            # 其他异常维度（如3维）→ 报错提示
            else:
                raise ValueError(f"原始 node_type 维度异常：{g.node_type.dim()} 维，需1或2维")
        else:
            # 极端情况：无 node_type 字段→用全0填充（需根据实际场景调整）
            num_nodes = g.x.shape[0] if hasattr(g, 'x') else 0
            raw_node_type = torch.zeros((num_nodes, 1), dtype=torch.long)

        num_nodes = raw_node_type.shape[0]  # 原始节点数（基于统一后的二维张量）

        # ---------------------- 2. 筛选边类型：分离 "edge_type=1" 和 "edge_type≠1" 的边 ----------------------
        # 确保 edge_type 是一维张量（避免维度不匹配）
        E = g.edge_index.shape[1]

        if hasattr(g, 'edge_type'):
            edge_type = g.edge_type.squeeze() if g.edge_type.dim() > 1 else g.edge_type

            # 处理空 edge_type：mask 必须是长度 E 的 bool，否则后面索引会炸
            if edge_type.numel() == 0:
                mask = torch.zeros(E, dtype=torch.bool)
            else:
                mask = (edge_type == 1).view(-1)
        else:
            # 无 edge_type 字段→同样给长度 E 的全 False mask（表示没有 edge_type=1 的边）
            mask = torch.zeros(E, dtype=torch.bool)

        # ---- 兜底：保证 mask 长度严格等于 E（非常关键，避免 boolean indexing 维度错误）----
        if mask.numel() != E:
            if mask.numel() == 0:
                mask = torch.zeros(E, dtype=torch.bool)
            elif mask.numel() < E:
                pad = torch.zeros(E - mask.numel(), dtype=torch.bool)
                mask = torch.cat([mask.bool(), pad], dim=0)
            else:
                mask = mask[:E].bool()

        not_edge_index = g.edge_index[:, mask]    # 用于生成新边的基础边
        buff_edge_index = g.edge_index[:, ~mask]  # 直接保留的边
        new_nodes_count = mask.sum().item()       # 新增节点数（与 edge_type=1 的边数一致）

        # ---------------------- 3. 安全生成新边（u_new: 原始节点→新增节点；v_new: 新增节点→原始节点） ----------------------
        # 初始化空的新边张量（确保形状为 (2, 0)，便于后续拼接）
        u_new = torch.empty((2, 0), dtype=torch.long)
        v_new = torch.empty((2, 0), dtype=torch.long)

        if new_nodes_count > 0 and not_edge_index.numel() > 0:
            # 提取 not_edge_index 的起点（u）和终点（v）（形状均为 (E1,)，E1为 mask.sum()）
            u = not_edge_index[0, :]  # 起点：原始节点索引
            v = not_edge_index[1, :]  # 终点：原始节点索引

            # 生成新增节点的索引（从原始节点数开始编号，避免与原始节点重复）
            new_nodes = num_nodes + torch.arange(new_nodes_count, dtype=torch.long)

            # 生成 u_new: 原始起点 → 新增节点（形状 (2, E1)）
            u_new = torch.stack([u, new_nodes], dim=0)
            # 生成 v_new: 新增节点 → 原始终点（形状 (2, E1)）
            v_new = torch.stack([new_nodes, v], dim=0)

        # ---------------------- 4. 生成新增节点的类型（确保与原始 node_type 维度一致） ----------------------
        if new_nodes_count > 0:
            new_node_type = torch.tensor([3] * new_nodes_count, dtype=torch.long).view(-1, 1)  # (E1, 1) 二维
        else:
            new_node_type = torch.empty((0, 1), dtype=torch.long)  # 空节点类型，保持 (0,1) 二维

        # ---------------------- 5. 拼接新边（buff_edge_index + 新生成的 u_new + v_new） ----------------------
        new_edge_index = torch.cat([buff_edge_index, u_new, v_new], dim=1)

        # ---------------------- 6. 拼接节点类型（此时二者均为 (N,1) 二维，可正常拼接） ----------------------
        # 提前校验维度（避免拼接报错）
        assert raw_node_type.dim() == 2 and raw_node_type.shape[1] == 1, f"原始 node_type 需为 (N,1)，当前 {raw_node_type.shape}"
        assert new_node_type.dim() == 2 and new_node_type.shape[1] == 1, f"new_node_type 需为 (M,1)，当前 {new_node_type.shape}"
        node_type = torch.cat((raw_node_type, new_node_type), dim=0)

        # ---------------------- 7. 最终校验：确保输出张量维度符合模型要求 ----------------------
        assert node_type.dim() == 2 and node_type.shape[1] == 1, "input_nodes 需为 (N, 1) 形状"
        assert new_edge_index.dim() == 2 and new_edge_index.shape[0] == 2, "input_edges 需为 (2, E) 形状"

        return dict(input_nodes=node_type, input_edges=new_edge_index)






class VerilogGraphDataset(Dataset):
    """安全版 VerilogGraphDataset，文件缺失或加载失败会跳过"""
    def __init__(self, pyg_path: str, verilog_path: str):
        super().__init__()
        self.pyg_path = pyg_path
        self.verilog_list = {}
        try:
            self.verilog_list = np.load(verilog_path, allow_pickle=True).item()
        except:
            print(f"[Warning] 加载 {verilog_path} 失败，数据为空")

        # 只保留存在的 pyg 文件
        self.pyg_list = []
        for f in os.listdir(pyg_path):
            if f.endswith(".pt"):
                design = f.split(".")[0]
                if design in self.verilog_list:
                    self.pyg_list.append(f)
                else:
                    print(f"[Warning] {design} 不在 verilog_list 中，跳过")

        self.designs = [d.split(".")[0] for d in self.pyg_list]

    def __len__(self):
        return len(self.designs)

    def __getitem__(self, i):
            design = self.designs[i]
            pyg_file = osp.join(self.pyg_path, f"{design}.pt")
            g = torch.load(pyg_file, map_location="cpu")

            # ---------------------- 1. Handle edge_index ----------------------
            if g.edge_index.dim() != 2 or g.edge_index.shape[0] != 2:
                g.edge_index = torch.empty((2, 0), dtype=torch.long)
            num_edges = g.edge_index.shape[1]

            # ---------------------- 2. FIX: Check 'node_type' AND 'x' ----------------------
            # Determine source of node types
            if hasattr(g, 'node_type'):
                raw_source = g.node_type
            elif hasattr(g, 'x'):
                # FIX IS HERE: Use 'x' if 'node_type' is missing
                raw_source = g.x.long() # Ensure it is LongTensor
            else:
                raw_source = None

            # Process the source into (N, 1) format
            if raw_source is not None:
                if raw_source.dim() == 1:
                    # 1D -> 2D (N, 1)
                    node_type = raw_source.view(-1, 1)
                elif raw_source.dim() == 2:
                    # 2D -> Take first column if multiple, or keep if (N, 1)
                    if raw_source.shape[1] != 1:
                        node_type = raw_source[:, 0].view(-1, 1)
                    else:
                        node_type = raw_source
                else:
                    raise ValueError(f"Node type/x dimension error: {raw_source.dim()}")
            else:
                # Fallback if neither exists (fills with 0)
                num_nodes = g.x.shape[0] if hasattr(g, 'x') else 0
                node_type = torch.zeros((num_nodes, 1), dtype=torch.long)

            num_nodes = node_type.shape[0]  # Raw node count

            # ---------------------- 3. Handle edge_type and mask ----------------------
            if hasattr(g, 'edge_type'):
                edge_type = g.edge_type
                if edge_type.dim() == 2:
                    edge_type = edge_type.squeeze(dim=1)
                if num_edges == 0 or edge_type.numel() == 0:
                    mask = torch.tensor([], dtype=torch.bool)
                else:
                    mask = (edge_type == 1)[:num_edges]
            else:
                mask = torch.tensor([], dtype=torch.bool)

            not_edge_index = g.edge_index[:, mask] if mask.numel() > 0 else torch.empty((2, 0), dtype=torch.long)
            buff_edge_index = g.edge_index[:, ~mask] if mask.numel() > 0 else g.edge_index
            new_nodes_count = mask.sum().item()

            # ---------------------- 4. Generate new edges ----------------------
            u_new = torch.empty((2, 0), dtype=torch.long)
            v_new = torch.empty((2, 0), dtype=torch.long)

            if new_nodes_count > 0 and not_edge_index.numel() > 0:
                u, v = not_edge_index
                new_nodes = num_nodes + torch.arange(new_nodes_count, dtype=torch.long)
                u_new = torch.stack([u, new_nodes], dim=0)
                v_new = torch.stack([new_nodes, v], dim=0)

            # ---------------------- 5. Generate new_node_type ----------------------
            if new_nodes_count > 0:
                new_node_type = torch.tensor([3] * new_nodes_count, dtype=torch.long).view(-1, 1)
            else:
                new_node_type = torch.empty((0, 1), dtype=torch.long)

            # ---------------------- 6. Concatenate Node Types ----------------------
            node_type = torch.cat((node_type, new_node_type), dim=0)

            # ---------------------- 7. Concatenate Edges ----------------------
            new_edge_index = torch.cat([buff_edge_index, u_new, v_new], dim=1)

            # ---------------------- 8. Process Verilog ----------------------
            max_len = 3584
            if design in self.verilog_list and self.verilog_list[design].size > 0:
                input_verilog = torch.tensor(self.verilog_list[design], dtype=torch.float).squeeze()
                if input_verilog.dim() == 0:
                    input_verilog = input_verilog.unsqueeze(0)
                if input_verilog.shape[0] < max_len:
                    input_verilog = torch.cat([input_verilog, torch.zeros(max_len - input_verilog.shape[0], dtype=torch.float)], dim=0)
                else:
                    input_verilog = input_verilog[:max_len]
            else:
                input_verilog = torch.zeros(max_len, dtype=torch.float)
            input_verilog = input_verilog.view(-1)

            return dict(
                input_nodes=node_type,
                input_edges=new_edge_index,
                input_verilogs=input_verilog
            )





class GraphQoRDataCollator:
    def __call__(self, graphs: List[dict]) -> Dict[str, Any]:

        max_node_num = max(g['input_nodes'].shape[0] for g in graphs)
        max_edge_num = max((g['input_edges'].shape[-1]) for g in graphs)
        batch_size = len(graphs)
        batch = {}

        batch["input_nodes"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["input_synvecs"] = torch.zeros(batch_size, 20, dtype=torch.long)
        batch["input_edges"] = torch.zeros(batch_size, 2, max_edge_num, dtype=torch.long)
        batch["padding_mask"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["labels"] = torch.zeros(batch_size, dtype=torch.float)

        for idx, g in enumerate(graphs):
            input_nodes = g["input_nodes"]
            node_num = input_nodes.shape[0]
            edge_index = copy.deepcopy(g['input_edges'])

            edge_num = edge_index.shape[1]
            batch["input_edges"][idx, 0, :edge_num] = edge_index[0]
            batch["input_edges"][idx, 0, edge_num:] = edge_index[0][0]
            batch["input_edges"][idx, 1, :edge_num] = edge_index[1]
            batch["input_edges"][idx, 1, edge_num:] = edge_index[1][0]

            batch["input_nodes"][idx, :node_num] = input_nodes + 1
            batch["input_synvecs"][idx] = g["input_synvecs"]
            batch["padding_mask"][idx, :node_num] = 1
            batch["labels"][idx] = g["labels"]
            # batch["labels"][idx] = g["labels"] / node_num
        
        return batch


class GraphQoRDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, root_path: str, data_path: str, pkl_path: str, target: str = None):
        super(GraphQoRDataset, self).__init__()
        fileDF = pd.read_csv(data_path)
        self.files = fileDF['fileName'].tolist()

        with open(pkl_path,'rb') as f:
            self.targetStatsDict = pickle.load(f)
        if target is not None:
            self.meanVarDataDict = self.computeMeanAndVarianceOfTargets(target)
        else:
            self.meanVarDataDict = self.computeMeanAndVarianceOfTargets()
        self.target = target

        self.root_path = root_path

    def computeMeanAndVarianceOfTargets(self, targetVar='nodes'):
        meanAndVarTargetDict = {}
        for des in self.targetStatsDict.keys():
            numNodes, _, _, areaVar, delayVar = self.targetStatsDict[des]
            if targetVar == 'delay':
                meanTarget, varTarget = getMeanAndVariance(delayVar)
            elif targetVar == 'area':
                meanTarget, varTarget = getMeanAndVariance(areaVar)
            else:
                meanTarget, varTarget = getMeanAndVariance(numNodes)
            meanAndVarTargetDict[des] = [meanTarget,varTarget]
        return meanAndVarTargetDict
    
    def addNormalizedTargets(self, data):
        sid = data.synID[0]
        desName = data.desName[0]
        if self.target == 'delay':    
            targetIdentifier = 4 # Column number of target 'Delay' in synthesisStatistics.pickle entries
            normTarget = (self.targetStatsDict[desName][targetIdentifier][sid] - self.meanVarDataDict[desName][0]) / self.meanVarDataDict[desName][1]
            # normTarget = self.targetStatsDict[desName][targetIdentifier][sid]
            label = torch.tensor([normTarget], dtype=torch.float32)
        elif self.target == 'area':
            targetIdentifier = 3 # Column number of target 'Area' in synthesisStatistics.pickle entries
            normTarget = (self.targetStatsDict[desName][targetIdentifier][sid] - self.meanVarDataDict[desName][0]) / self.meanVarDataDict[desName][1]
            # normTarget = self.targetStatsDict[desName][targetIdentifier][sid]
            label = torch.tensor([normTarget],dtype=torch.float32)
        else:
            targetIdentifier = 0 # Column number of target 'Nodes' in synthesisStatistics.pickle entries
            normTarget = (self.targetStatsDict[desName][targetIdentifier][sid] - self.meanVarDataDict[desName][0]) / self.meanVarDataDict[desName][1]
            # normTarget = self.targetStatsDict[desName][targetIdentifier][sid]
            label = torch.tensor([normTarget], dtype=torch.float32)
        return label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filePathArchive = osp.join(self.root_path, self.files[i])
        filePathName = osp.basename(osp.splitext(filePathArchive)[0])
        with ZipFile(filePathArchive) as myzip:
            with myzip.open(filePathName) as myfile:
                g = torch.load(myfile, map_location="cpu")

        mask = (g.edge_type == 1).squeeze()
        not_edge_index = g.edge_index[:, mask]
        buff_edge_index = g.edge_index[:, ~mask]

        num_nodes = g.node_type.shape[0]
        new_nodes_count = mask.sum().item()

        u, v = not_edge_index
        new_nodes = num_nodes + torch.arange(new_nodes_count)
        u_new = torch.stack([u, new_nodes])
        v_new = torch.stack([new_nodes, v])

        new_node_type = torch.tensor([3]*new_nodes_count, dtype=torch.long)
        new_edge_index = torch.cat([u_new, v_new], dim=1)

        node_type = torch.cat((g.node_type, new_node_type), dim=0)
        edge_index = torch.cat((buff_edge_index, new_edge_index), dim=1)

        synvec = g['synVec']
        label = self.addNormalizedTargets(g)

        return_dict = dict(input_nodes=node_type, input_edges=edge_index, input_synvecs=synvec, labels=label)
        return return_dict
    

class GraphQoRDeepGateDataCollator:
    def __call__(self, graphs: List[dict]) -> Dict[str, Any]:

        batch_size = len(graphs)
        batch = {}

        batch["input_nodes"] = torch.zeros(batch_size, 128, dtype=torch.float)
        batch["input_synvecs"] = torch.zeros(batch_size, 20, dtype=torch.long)
        batch["input_node_num"] = torch.zeros(batch_size, dtype=torch.long)
        batch["labels"] = torch.zeros(batch_size, dtype=torch.float)

        for idx, g in enumerate(graphs):
            node_num = g["input_node_num"]
            batch["input_nodes"][idx] = g["input_nodes"]
            batch["input_synvecs"][idx] = g["input_synvecs"]
            batch["input_node_num"] = node_num
            # batch["labels"][idx] = g["labels"]
            batch["labels"][idx] = g["labels"] / node_num * 100
        
        return batch


class GraphQoRDeepGateDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, root_path: str, data_path: str, deepgate_path: str, pkl_path: str, target: str = None):
        super(GraphQoRDeepGateDataset, self).__init__()
        fileDF = pd.read_csv(data_path)
        self.files = []

        design_delete = ["or1200", "or1200_fpu", "aes_ncsu", 's35932', 's38417', 's38584']
        for f in fileDF['fileName'].tolist():
            if f.split("_syn")[0] not in design_delete:
                self.files.append(f)

        with open(pkl_path,'rb') as f:
            self.targetStatsDict = pickle.load(f)
        if target is not None:
            self.meanVarDataDict = self.computeMeanAndVarianceOfTargets(target)
        else:
            self.meanVarDataDict = self.computeMeanAndVarianceOfTargets()
        self.target = target

        self.nodes = np.load(deepgate_path, allow_pickle=True)[0]

        self.root_path = root_path

    def computeMeanAndVarianceOfTargets(self, targetVar='nodes'):
        meanAndVarTargetDict = {}
        for des in self.targetStatsDict.keys():
            numNodes, _, _, areaVar, delayVar = self.targetStatsDict[des]
            if targetVar == 'delay':
                meanTarget, varTarget = getMeanAndVariance(delayVar)
            elif targetVar == 'area':
                meanTarget, varTarget = getMeanAndVariance(areaVar)
            else:
                meanTarget, varTarget = getMeanAndVariance(numNodes)
            meanAndVarTargetDict[des] = [meanTarget,varTarget]
        return meanAndVarTargetDict
    
    def addNormalizedTargets(self, data):
        sid = data.synID[0]
        desName = data.desName[0]
        if self.target == 'delay':    
            targetIdentifier = 4 # Column number of target 'Delay' in synthesisStatistics.pickle entries
            # normTarget = (self.targetStatsDict[desName][targetIdentifier][sid] - self.meanVarDataDict[desName][0]) / self.meanVarDataDict[desName][1]
            normTarget = self.targetStatsDict[desName][targetIdentifier][sid]
            label = torch.tensor([normTarget], dtype=torch.float32)
        elif self.target == 'area':
            targetIdentifier = 3 # Column number of target 'Area' in synthesisStatistics.pickle entries
            # normTarget = (self.targetStatsDict[desName][targetIdentifier][sid] - self.meanVarDataDict[desName][0]) / self.meanVarDataDict[desName][1]
            normTarget = self.targetStatsDict[desName][targetIdentifier][sid]
            label = torch.tensor([normTarget],dtype=torch.float32)
        else:
            targetIdentifier = 0 # Column number of target 'Nodes' in synthesisStatistics.pickle entries
            # normTarget = (self.targetStatsDict[desName][targetIdentifier][sid] - self.meanVarDataDict[desName][0]) / self.meanVarDataDict[desName][1]
            normTarget = self.targetStatsDict[desName][targetIdentifier][sid]
            label = torch.tensor([normTarget], dtype=torch.float32)
        return label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filePathArchive = osp.join(self.root_path, self.files[i])
        filePathName = osp.basename(osp.splitext(filePathArchive)[0])
        with ZipFile(filePathArchive) as myzip:
            with myzip.open(filePathName) as myfile:
                g = torch.load(myfile, map_location="cpu")
        design = self.files[i].split("_syn")[0]

        node_num = g['node_type'].shape[0]

        nodes = torch.mean(torch.mean(torch.tensor(self.nodes[design], dtype=torch.float), dim=0), dim=0)
        synvec = g['synVec']
        label = self.addNormalizedTargets(g)

        return_dict = dict(input_nodes=nodes, input_synvecs=synvec, input_node_num=node_num, labels=label)
        return return_dict