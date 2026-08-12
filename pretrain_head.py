import os
from os.path import exists, join, isdir
import gc
import json
import math
import random
import copy
from copy import deepcopy
from tqdm import tqdm
import zipfile

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Callable, List, Tuple, Union, Any

import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from tqdm import tqdm
import time

import re
import transformers
from transformers import TrainingArguments, Trainer, set_seed

from data_helper import CustomDataCollator, CustomDataset, GraphDataset, VerilogGraphDataset
from aigmae.configuration_vgmae import AIGMAEConfig
from aigmae.modeling_vgmae_attention import AIGMAEModel_cross_finetune_head,AIGMAEModel_cross

from custom_trainer import CustomTrainer

import warnings

warnings.filterwarnings("ignore")

# =========================================================
# Global: keep node token clamp consistent with eval_eq.py
# Raw node types are clamped to [0, NODE_MAX_CLASS], then +1 (0 reserved for padding)
# =========================================================
NODE_MAX_CLASS = None

import math
import random
from collections import defaultdict
from torch.utils.data import Sampler


class SimilarityHead(nn.Module):
    """与旧 eval_eq.py 一致：MLP([e1,e2,|e1-e2|]) -> logit"""
    def __init__(self, emb_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([e1, e2, (e1 - e2).abs()], dim=-1)
        return self.fc(feat).squeeze(-1)  # [B]
    
class QoRRegressionHead(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, graph_emb):
        return self.fc(graph_emb).squeeze(-1)
    

@dataclass
class DataArguments:
    root_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    pyg_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    verilog_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    report_to: str = field(default="none")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    lr_scheduler_type: str = field(default="cosine")
    min_lr_ratio: float = field(default=0.1)
    lr_scheduler_kwargs: Dict[str, str] = field(
        default_factory=lambda:{"num_cycles": 0.5}
    )
    # --------------------------
    # 新增：DataLoader 并行参数（关键）
    # --------------------------
    dataloader_num_workers: int = field(
        default=16, metadata={"help": "DataLoader 进程数，128核CPU建议设为16"}
    )
    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "锁定内存，加速CPU→GPU传输"}
    )
    dataloader_prefetch_factor: int = field(
        default=4, metadata={"help": "预加载批次数量，建议设为4"}
    )
    dataloader_persistent_workers: bool = field(
        default=True, metadata={"help": "保持进程不销毁，减少开销"}
    )
    #---------------------------
    #新增参数
    #--------------------------
    mode: str = field(
        default="pretrain",
        metadata={"help": "pretrain 或 fewshot"}
    )
    pretrained_ckpt: Optional[str] = field(
        default=None,
        metadata={"help": "fewshot 模式下加载的预训练权重（.pt 或 save_pretrained 目录）"}
    )
    fewshot_steps_eq: int = field(
        default=200,
        metadata={"help": "只训分类头的微调步数"}
    )
    fewshot_steps_qor: int = field(
        default=200,
        metadata={"help": "只训分类头的微调步数"}
    )
    fewshot_lr: float = field(
        default=1e-3,
        metadata={"help": "分类头学习率"}
    )
    fewshot_batches: int = field(
        default=1,
        metadata={"help": "用多少个 batch 的 support 来做微调（最简=1）"}
    )
    finetune_head_type: str = field(
        default="linear",
        metadata={"help": "fewshot 微调头类型：linear 或 proto。建议从 linear 开始。"}
    )
    num_finetune_classes: int = field(
        default=2,
        metadata={"help": "fewshot 分类类别数（linear/proto 都会用到）"}
    )
    fewshot_input_mode: str = field(
        default="graph",
        metadata={"help": "fewshot 输入模式：graph 表示从 .pt(PYG Data) 微调；vg 表示旧的 VG 输入。默认 graph。"}
    )
    fewshot_max_skip_ratio: float = field(
        default=3.0,
        metadata={"help": "proto 模式下允许跳过 batch 的比例上限，超过则报错（防止死循环）。"}
    )
    qor_graphs_dir: str = field(default=None, metadata={"help": "QoR 图所在目录（包含各子目录或 pt 文件）"})

    eq_pairs_csv: str = field(default=None, metadata={"help": "EQ pairs csv（graph1_id,graph2_id,label）"})
    eq_graphs_dir: str = field(default=None, metadata={"help": "EQ 图目录（{id}.pt）"})

    loss_w_qor: float = field(default=1.0, metadata={"help": "QoR loss 权重"})
    loss_w_eq: float = field(default=1.0, metadata={"help": "EQ loss 权重"})
    
def find_pt_path_in_dir(subdir_path: str, graph_id: str):
    import os

    gid = str(graph_id).strip()
    if not gid:
        return None

    # 1) 直接同名匹配：graph_id.pt
    p = os.path.join(subdir_path, f"{gid}.pt")
    if os.path.exists(p):
        return p

    # 2) 有些人 graph_id 里已经带 .pt
    if gid.endswith(".pt"):
        p2 = os.path.join(subdir_path, gid)
        if os.path.exists(p2):
            return p2

    # 3) fallback：遍历目录里所有 pt，按“去掉后缀后的文件名”匹配
    for fname in os.listdir(subdir_path):
        if fname.endswith(".pt") and os.path.splitext(fname)[0] == gid:
            return os.path.join(subdir_path, fname)

    return None



def extract_gate_count_from_row(row):
    import pandas as pd
    for c in ["optimized_gate_count", "optimized_gate", "gate_count", "and_gate_count"]:
        if c in row and pd.notna(row[c]):
            try:
                return float(row[c])
            except Exception:
                pass
    return 0.0

class QoRFewShotDataset(torch.utils.data.Dataset):
    def __init__(self, graphs_root: str, skip_subdirs=None):
        import os
        import pandas as pd
        import numpy as np

        self.items = []
        self.graphs_root = graphs_root
        skip_subdirs = set(skip_subdirs or [])

        for subdir in sorted(os.listdir(graphs_root)):
            if subdir in skip_subdirs: continue
            subdir_path = os.path.join(graphs_root, subdir)
            if not os.path.isdir(subdir_path): continue

            qor_csv = os.path.join(subdir_path, "qor_summary.csv")
            if not os.path.exists(qor_csv): continue

            df = pd.read_csv(qor_csv)
            if "graph_id" not in df.columns: continue

            for _, row in df.iterrows():
                csv_graph_id = str(row.get("graph_id", "")).strip()
                if not csv_graph_id: continue

                pt_path = find_pt_path_in_dir(subdir_path, csv_graph_id)
                if pt_path is None: continue

                y = extract_gate_count_from_row(row)
                if pd.isna(y) or y <= 0:
                    continue
                # 【核心修复】：放弃按电路归一化，改为全局 Log 变换
                # log 能平滑处理几十到几千的巨大差距，且排序关系严格不变
                safe_y = np.log(y + 1.0)
#                 safe_y = float(y)
                self.items.append((pt_path, safe_y))

        if len(self.items) == 0:
            raise RuntimeError("QoRFewShotDataset: 没找到任何可用样本。")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import torch
        pt_path, y = self.items[idx]
        data = torch.load(pt_path, map_location="cpu")
        return {"graph": data, "qor_y": torch.tensor(y, dtype=torch.float32)}


class EQFewShotPairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs_csv: str, graphs_dir: str):
        import os, re
        self.graphs_dir = graphs_dir
        self.pairs = []

        with open(pairs_csv, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip().strip('"').strip("'") for p in line.split(",")]
                if len(parts) >= 2 and parts[0].lower() == "graph1_id":
                    continue
                if len(parts) != 3:
                    continue
                a, b, l = parts
                if not re.fullmatch(r"-?\d+", l):
                    continue
                a_pt = os.path.join(graphs_dir, f"{a}.pt")
                b_pt = os.path.join(graphs_dir, f"{b}.pt")
                if os.path.exists(a_pt) and os.path.exists(b_pt):
                    self.pairs.append((a_pt, b_pt, float(int(l))))

        if len(self.pairs) == 0:
            raise RuntimeError("EQFewShotPairsDataset: 没读取到有效 pairs（检查 pairs_csv / graphs_dir）")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        import torch
        a_pt, b_pt, l = self.pairs[idx]
        g1 = torch.load(a_pt, map_location="cpu")
        g2 = torch.load(b_pt, map_location="cpu")
        return {"g1": g1, "g2": g2, "eq_y": torch.tensor(l, dtype=torch.float32)}

def _pyg_to_dense_inputs(data):
    """
    兼容 .pt 里的 PyG Data：
    - data.x: [N] 或 [N,1] token ids (原始 node_type: 0,1,2,3...)
    - data.input_nodes: [N,1] 或 [N]
    - data.edge_index: [2,E] 或 data.input_edges
    关键：对 token 做 +1，把 0 留给 padding
    """
    import torch

    # ---- nodes ----
    if hasattr(data, "x"):
        n = data.x
    elif hasattr(data, "input_nodes"):
        n = data.input_nodes
    else:
        raise ValueError("PyG Data has neither x nor input_nodes")

    if n.dim() == 2 and n.size(-1) == 1:
        n = n.squeeze(-1)
    n = n.long()
    # ===== 关键：与 eval_eq.py 对齐的 token 规范化 =====
    # embedding 约定：0 是 padding，真实类别映射到 1..(max_class+1)
    global NODE_MAX_CLASS
    max_cls = NODE_MAX_CLASS if NODE_MAX_CLASS is not None else 3
    n = n.clamp(min=0, max=max_cls)   # 只 +1 一次

    # ---- edges ----
    if hasattr(data, "edge_index"):
        e = data.edge_index
    elif hasattr(data, "input_edges"):
        e = data.input_edges
        if e.dim() == 3:
            e = e[0]
    else:
        raise ValueError("PyG Data has neither edge_index nor input_edges")

    e = e.long()
        # --- 关键：按“该图真实节点数”过滤掉越界边 ---
    num_nodes = n.numel()
    if e.numel() > 0:
        valid = (e[0] >= 0) & (e[0] < num_nodes) & (e[1] >= 0) & (e[1] < num_nodes)
        e = e[:, valid].contiguous()

    if e.dim() != 2 or e.size(0) != 2:
        raise ValueError(f"edge shape {tuple(e.shape)} invalid, expect [2,E]")

    return n, e



class QoRCollator:
    def __call__(self, batch):
        import torch

        nodes, edges, ys = [], [], []
        for item in batch:
            n, e = _pyg_to_dense_inputs(item["graph"])
            nodes.append(n); edges.append(e)
            ys.append(item["qor_y"])

        B = len(nodes)
        maxN = max(n.numel() for n in nodes)
        maxE = max(e.size(1) for e in edges) if len(edges) else 0

        input_nodes = torch.zeros((B, maxN), dtype=torch.long)
        padding_mask = torch.zeros((B, maxN), dtype=torch.bool)
        input_edges = torch.full((B, 2, maxE), -1, dtype=torch.long)

        for i, (n, e) in enumerate(zip(nodes, edges)):
            N = n.numel(); E = e.size(1)
            input_nodes[i, :N] = n
            padding_mask[i, :N] = True
            if E > 0:
                input_edges[i, :, :E] = e

        return {
            "input_nodes": input_nodes,
            "input_edges": input_edges,
            "padding_mask": padding_mask,
            "qor_y": torch.stack(ys).view(-1).float(),
        }



class EQCollator:
    def __call__(self, batch):
        import torch

        g1_nodes, g1_edges = [], []
        g2_nodes, g2_edges = [], []
        ys = []

        for item in batch:
            n1, e1 = _pyg_to_dense_inputs(item["g1"])
            n2, e2 = _pyg_to_dense_inputs(item["g2"])
            g1_nodes.append(n1); g1_edges.append(e1)
            g2_nodes.append(n2); g2_edges.append(e2)
            ys.append(item["eq_y"])

        B = len(batch)

        def pad_side(nodes, edges):
            maxN = max(n.numel() for n in nodes)
            maxE = max(e.size(1) for e in edges) if len(edges) else 0
            input_nodes = torch.zeros((B, maxN), dtype=torch.long)
            padding_mask = torch.zeros((B, maxN), dtype=torch.bool)
            input_edges = torch.full((B, 2, maxE), -1, dtype=torch.long)

            for i, (n, e) in enumerate(zip(nodes, edges)):
                N = n.numel(); E = e.size(1)
                input_nodes[i, :N] = n
                padding_mask[i, :N] = True
                if E > 0:
                    input_edges[i, :, :E] = e
            return input_nodes, input_edges, padding_mask

        g1_in, g1_e, g1_pm = pad_side(g1_nodes, g1_edges)
        g2_in, g2_e, g2_pm = pad_side(g2_nodes, g2_edges)

        return {
            "g1_input_nodes": g1_in,
            "g1_input_edges": g1_e,
            "g1_padding_mask": g1_pm,
            "g2_input_nodes": g2_in,
            "g2_input_edges": g2_e,
            "g2_padding_mask": g2_pm,
            "eq_y": torch.stack(ys).view(-1).float(),
        }




def train():
    parser = transformers.HfArgumentParser((DataArguments, TrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()

    # ====== 你自己需要在 TrainingArguments 里加这些参数 ======
    # training_args.mode: "pretrain" 或 "fewshot"
    # training_args.pretrained_ckpt: 预训练权重路径（fewshot 模式用）
    # training_args.fewshot_ratio: 默认 0.05
    # training_args.fewshot_steps: 默认 50
    # training_args.fewshot_lr: 默认 1e-2
    # ================================================

    num_cycles = math.acos(training_args.min_lr_ratio * 2 - 1) / (math.pi * 2)
    training_args.lr_scheduler_kwargs["num_cycles"] = num_cycles

#     mgm_dataset = GraphDataset(root_path=data_args.root_path, data_path=data_args.data_path)
#     vg_dataset = VerilogGraphDataset(pyg_path=data_args.pyg_path, verilog_path=data_args.verilog_path)
#     train_dataset = CustomDataset([mgm_dataset, vg_dataset])
#     full_dataset = CustomDataset([mgm_dataset, vg_dataset])
    
    model_config = AIGMAEConfig(
        num_classes=5,
        num_encoder_layers=7,
        num_cross_decoder_layers=2,
        hidden_size=64,
        cross_hidden_size=3584,
        cross_num_heads=8,
    )

    collator = CustomDataCollator(
        mgm_mask_ratio=0.3,
        align_mask_ratio=0.7,
        cross_hidden_size=3584,
    )
    # =========================
    # 1) 预训练模式：完全保留你原来的代码
    # =========================
    if getattr(training_args, "mode", "pretrain") == "pretrain":
        model = AIGMAEModel_cross(model_config)

        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": training_args.weight_decay,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
        scheduler = None

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collator,
            optimizers=(optimizer, scheduler),
            extra_losses=["node_loss", "vg_node_loss", "indegree_loss", "outdegree_loss", "indegree_loss_t", "outdegree_loss_t"],
        )

        trainer.train()
        model.save_pretrained(training_args.output_dir)
        return
    # =========================================================
    # 模式 B: 端到端全流程 Few-shot 微调 (Tuple-Aware Version)
    # =========================================================
    elif training_args.mode == "fewshot":
        print(f"\n🚀 进入 Few-shot 微调模式 (Tuple Support Added)")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not training_args.pretrained_ckpt:
            raise ValueError("❌ Few-shot mode requires --pretrained_ckpt to load base config and weights.")

        # 1. Config 加载
        print(f"📥 Loading Config from: {training_args.pretrained_ckpt}")
        try:
            ckpt_config = AIGMAEConfig.from_pretrained(training_args.pretrained_ckpt)
        except Exception:
            print("⚠️ Could not load config.json, falling back to default config (RISKY!).")
            ckpt_config = AIGMAEConfig(
                num_classes=5, num_encoder_layers=7, num_cross_decoder_layers=2,
                hidden_size=64, cross_hidden_size=3584, cross_num_heads=8,
            )
            
        # 2. 初始化模型
        # proto 模式下，batch_size 必须 >= 2，否则会疯狂 skip
        if training_args.finetune_head_type == "proto" and training_args.per_device_train_batch_size < 2:
            raise ValueError("❌ proto fewshot 需要 per_device_train_batch_size >= 2，否则 batch 内类别不足会被跳过。建议先用 --finetune_head_type linear。")

        model = AIGMAEModel_cross_finetune_head(
            ckpt_config,
            finetune_head_type=training_args.finetune_head_type,
            num_finetune_classes=training_args.num_finetune_classes
        )

        # 3. 加载权重 & 物理 Shape 读取
        print(f"📥 Loading Weights from: {training_args.pretrained_ckpt}")
        if os.path.isdir(training_args.pretrained_ckpt):
            temp_model = AIGMAEModel_cross.from_pretrained(training_args.pretrained_ckpt)
            ckpt_state_dict = temp_model.state_dict()
            del temp_model
        else:
            ckpt_state_dict = torch.load(training_args.pretrained_ckpt, map_location='cpu')

        if 'token_emb.node_token_emb.weight' not in ckpt_state_dict:
            raise ValueError("❌ Fatal: 'token_emb.node_token_emb.weight' not found in checkpoint!")
        real_ckpt_vocab_size = ckpt_state_dict['token_emb.node_token_emb.weight'].shape[0]
        
        if 'node_prediction_head.classifier.weight' in ckpt_state_dict:
            real_ckpt_node_pred_classes = ckpt_state_dict['node_prediction_head.classifier.weight'].shape[0]
        else:
            real_ckpt_node_pred_classes = ckpt_config.num_classes 
            
        if 'vg_node_prediction_head.classifier.weight' in ckpt_state_dict:
            real_ckpt_vg_node_pred_classes = ckpt_state_dict['vg_node_prediction_head.classifier.weight'].shape[0]
        else:
            real_ckpt_vg_node_pred_classes = ckpt_config.num_classes

        print(f"\n🔍 [Checkpoint Physical Stats]")
        print(f"   - Real Vocab Size:       {real_ckpt_vocab_size}")
        print(f"   - Real Node Pred Classes: {real_ckpt_node_pred_classes}")
        print(f"   - Real VG Node Pred Classes: {real_ckpt_vg_node_pred_classes}")
        
        # 严格 Shape 验证
        print("\n🔍 [Strict Shape Validation]")
        checks = [
            ("token_emb.node_token_emb.weight", 0, real_ckpt_vocab_size),
            ("node_prediction_head.classifier.weight", 0, real_ckpt_node_pred_classes),
            ("vg_node_prediction_head.classifier.weight", 0, real_ckpt_vg_node_pred_classes)
        ]
        
        model_sd = model.state_dict()
        mismatch_found = False
        print(f"{'Param Key':<50} | {'Model Shape':<15} | {'Ckpt Shape':<15} | {'Status'}")
        print("-" * 100)
        for key, dim, expected_val in checks:
            if key in ckpt_state_dict and key in model_sd:
                m_shape = model_sd[key].shape
                c_shape = ckpt_state_dict[key].shape
                status = "✅ PASS"
                if m_shape != c_shape:
                    status = "❌ MISMATCH"; mismatch_found = True
                print(f"{key:<50} | {str(tuple(m_shape)):<15} | {str(tuple(c_shape)):<15} | {status}")
            else:
                c_exists = "FOUND" if key in ckpt_state_dict else "MISSING"
                m_exists = "FOUND" if key in model_sd else "MISSING"
                print(f"{key:<50} | {m_exists:<15} | {c_exists:<15} | ⚠️ SKIP")

        if mismatch_found:
            raise ValueError("❌ Fatal Shape Mismatch! Checkpoint weights do not match Model config/initialization.")

       # 4. 执行加载
        missing_keys, unexpected_keys = model.load_state_dict(ckpt_state_dict, strict=False)
        print(f"✅ Weights Loaded Successfully. Head Params Initialized from Scratch: {len(missing_keys)}")
        allowed_head_keywords = ['logit_scale_log', 'logit_scale', 'scale', 'temperature', 'classifier', 'qor_head', 'eq_head']
        
        backbone_missing = []
        for k in missing_keys:
            if not any(keyword in k for keyword in allowed_head_keywords):
                backbone_missing.append(k)
        
        if len(backbone_missing) > 0:
             raise ValueError(
                 f"❌ Fatal Error: Backbone parameters are missing in Checkpoint!\n"
                 f"   Missing Keys (Sample): {backbone_missing[:5]} ...\n"
                 "   Cannot proceed with few-shot finetuning on a broken backbone."
             )
        
        print(f"✅ Weights Loaded Successfully. Head Params Initialized from Scratch: {len(missing_keys)}")
        model.to(device)
        model.freeze_backbone()
        
        
        # === 新增：SimilarityHead（路线A）===
        emb_dim = ckpt_config.hidden_size  # 64
        sim_head = SimilarityHead(emb_dim=emb_dim, hidden_dim=256, dropout=0.1).to(device)
        sim_head.train()
        
        emb_dim = ckpt_config.hidden_size
        qor_head = QoRRegressionHead(emb_dim=emb_dim).to(device)
        qor_head.train()
        
        # ===== 关键：设置 NODE_MAX_CLASS，保证训练侧 clamp 与 eval_eq.py 一致 =====
        global NODE_MAX_CLASS
        NODE_MAX_CLASS = int(model.token_emb.node_token_emb.num_embeddings) - 2
        print(f"[Fewshot] NODE_MAX_CLASS set to {NODE_MAX_CLASS} (from vocab={model.token_emb.node_token_emb.num_embeddings})")
        # 记录初始参数
        init_param_val = None
        if hasattr(model, 'logit_scale_log'):
            init_param_val = model.logit_scale_log.exp().item()
            print(f"🔍 Init logit_scale: {init_param_val}")
        elif hasattr(model, 'classifier'):
            init_param_val = model.classifier.weight.data.sum().item()

        # =========================================================
        # 5. 数据准备 (QoR/EQ Few-shot Supervised)
        # =========================================================
        if training_args.qor_graphs_dir is None:
            raise ValueError("❌ QoR fewshot requires --qor_graphs_dir")
        if training_args.eq_pairs_csv is None or training_args.eq_graphs_dir is None:
            raise ValueError("❌ EQ fewshot requires --eq_pairs_csv and --eq_graphs_dir")


        qor_dataset = QoRFewShotDataset(training_args.qor_graphs_dir)
        eq_dataset = EQFewShotPairsDataset(
            pairs_csv=training_args.eq_pairs_csv,
            graphs_dir=training_args.eq_graphs_dir,
        )



        qor_loader = torch.utils.data.DataLoader(
            qor_dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=QoRCollator(),
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
            drop_last=True,
        )


        eq_loader = torch.utils.data.DataLoader(
            eq_dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=EQCollator(),
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
            drop_last=True,
        )

        
        # 6. 训练循环
        optimizer = torch.optim.AdamW(sim_head.parameters(), lr=training_args.fewshot_lr) ##eq训练
        
        print(f"\n🔥 开始微调 (Steps: {training_args.fewshot_steps_eq})...")
        global_step = 0  # micro-batch steps（保持语义不变）
        update_step = 0  # optimizer updates（新增：用于观测）
        grad_accum = max(1, int(training_args.gradient_accumulation_steps))
        model.zero_grad(set_to_none=True)

        eq_iter = iter(eq_loader)

        progress_bar = tqdm(total=training_args.fewshot_steps_eq, desc="Few-shot Finetuning (QoR+EQ)")

        while global_step < training_args.fewshot_steps_eq:

            # -------- EQ step --------
            try:
                batch_e = next(eq_iter)
            except StopIteration:
                eq_iter = iter(eq_loader)
                batch_e = next(eq_iter)

            for k, v in batch_e.items():
                if isinstance(v, torch.Tensor):
                    batch_e[k] = v.to(device)

            # 2) 训练 SimilarityHead
            loss_eq, logit = model.forward_eq_tokenmean(
                batch_e,
                sim_head=sim_head,
                detach_backbone=True,
            )
            (loss_eq / grad_accum).backward()


            global_step += 1
            progress_bar.update(1)
            # 每 grad_accum 个 micro-step 更新一次
            if (global_step % grad_accum) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

            if global_step % 10 == 0:
                progress_bar.set_postfix({
                    "eq": f"{loss_eq.item():.4e}",
                    "upd": f"{update_step}",
                })

        # 处理尾巴：如果最后不足 grad_accum，也把残余梯度更新掉
        if (global_step % grad_accum) != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1
            print(f"[Fewshot] Final partial update applied. update_step={update_step}")
        # 7. 保存
        print(f"\n💾 保存模型到: {training_args.output_dir}")
        model.config.downstream_fewshot = "qor_eq"
        model.config.loss_w_eq  = float(training_args.loss_w_eq)

        
        if init_param_val is not None:
            model.config.finetune_logit_scale_init = float(init_param_val)
        if hasattr(model, 'logit_scale_log'):
            final_val = model.logit_scale_log.exp().item()
            model.config.finetune_logit_scale_final = float(final_val)
            
        model.save_pretrained(training_args.output_dir)
        model.config.save_pretrained(training_args.output_dir)
        head_path = os.path.join(training_args.output_dir, "similarity_head.pt")
        torch.save(sim_head.state_dict(), head_path)
        print(f"💾 Saved SimilarityHead to: {head_path}")



        optimizer = torch.optim.AdamW(qor_head.parameters(), lr=training_args.fewshot_lr) ##qor训练 


        print(f"\n🔥 开始微调 (Steps: {training_args.fewshot_steps_qor})...")
        global_step = 0  # micro-batch steps（保持语义不变）
        update_step = 0  # optimizer updates（新增：用于观测）
        grad_accum = max(1, int(training_args.gradient_accumulation_steps))
        model.zero_grad(set_to_none=True)

        qor_iter = iter(qor_loader) 

        progress_bar = tqdm(total=training_args.fewshot_steps_qor, desc="Few-shot Finetuning (QoR+EQ)")

        while global_step < training_args.fewshot_steps_qor:
            # -------- QoR step --------
            try:
                batch_q = next(qor_iter)
            except StopIteration:
                qor_iter = iter(qor_loader)
                batch_q = next(qor_iter)

            for k, v in batch_q.items():
                if isinstance(v, torch.Tensor):
                    batch_q[k] = v.to(device)

            loss_qor, pred = model.forward_qor_tokenmean(
                batch_q,
                qor_head=qor_head,
                detach_backbone=True,
            )
            (loss_qor / grad_accum).backward()

            global_step += 1
            progress_bar.update(1)
            # 每 grad_accum 个 micro-step 更新一次
            if (global_step % grad_accum) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

            if global_step % 10 == 0:
                progress_bar.set_postfix({
                    "qor": f"{loss_qor.item():.4e}", 
                    "upd": f"{update_step}",
                })

        # 处理尾巴：如果最后不足 grad_accum，也把残余梯度更新掉
        if (global_step % grad_accum) != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1
            print(f"[Fewshot] Final partial update applied. update_step={update_step}")
        # 7. 保存
        print(f"\n💾 保存模型到: {training_args.output_dir}")
        model.config.downstream_fewshot = "qor_eq"
        model.config.loss_w_qor = float(training_args.loss_w_qor) 

        
        if init_param_val is not None:
            model.config.finetune_logit_scale_init = float(init_param_val)
        if hasattr(model, 'logit_scale_log'):
            final_val = model.logit_scale_log.exp().item()
            model.config.finetune_logit_scale_final = float(final_val)
            

        qor_head_path = os.path.join(training_args.output_dir, "qor_head.pt")
        torch.save(qor_head.state_dict(), qor_head_path)
        print(f"💾 Saved QoR head to: {qor_head_path}")

        print("🔍 执行加载自检...")
        try:
            reloaded_model = AIGMAEModel_cross_finetune_head.from_pretrained(
                training_args.output_dir
            )
            print("✅ Reload Check Passed.")
        except Exception as e:
            print(f"❌ Reload Check Failed: {e}")

if __name__ == "__main__":
    set_seed(42) 
    train()