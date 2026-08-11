#!/usr/bin/env python3
# =============================================================
# MGVGA 等价性模型评估脚本（对齐训练脚本逻辑）
# 功能：加载预训练基座模型，直接做分类评估（无需另训练分类头）
# 输出：脚本同级目录下生成指标汇总表格（AUC、Precision、Recall、F1）
# =============================================================
import os
import sys
import argparse
import csv
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve, confusion_matrix
import numpy as np
import traceback
import random
import re
from datetime import datetime
NODE_MAX_CLASS = None  # 会在加载模型后赋值

# -------------------------
# 固定随机种子（与训练脚本一致）
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ 已固定随机种子：{seed}（确保评估可复现）")

# -------------------------
# 添加 aigmae 父目录（与训练脚本一致）
# -------------------------
sys.path.append("/root/autodl-tmp/MGVGA-master/")
# -------------------------
# 导入自定义模型和配置（与训练脚本一致）
# -------------------------
try:
    from aigmae.configuration_vgmae import AIGMAEConfig
    from aigmae.modeling_vgmae_attention import AIGMAEModel_cross_finetune_head,AIGMAEModel_cross
except ImportError as e:
    print(f"ERROR: 无法导入自定义模型，请检查 aigmae 模块路径！")
    print(f"错误信息: {e}")
    sys.exit(1)

# -------------------------
# 加载图结构（仅加载PT文件，与训练脚本一致）
# -------------------------
def load_graph_from_pt(graphs_dir, graph_id):
    pt_file = os.path.join(graphs_dir, f"{graph_id}.pt")
    if not os.path.exists(pt_file):
        raise FileNotFoundError(f"PT文件未找到：{pt_file}")
    data = torch.load(pt_file, map_location="cpu")
    if data.edge_index.dim() != 2 or data.edge_index.shape[0] != 2:
        raise RuntimeError(f"{graph_id}.pt 的 edge_index 维度错误：{data.edge_index.shape}")
    return data

def check_edges(data, graph_id=None):
    """含越界边直接返回False，触发丢弃逻辑，不限节点大小"""
    num_nodes = data.x.size(0)
    edge_index = data.edge_index
    mask = (edge_index >= num_nodes).any(dim=0)
    if mask.any():
        print(f"❌ 图 {graph_id if graph_id else '未知'} 含 {mask.sum().item()} 条越界边 → 丢弃该图")
        return False
    else:
        return True

# -------------------------
# 数据集定义（适配训练脚本逻辑，添加无效图跳过）
# -------------------------
class EquivPairsDataset(Dataset):
    def __init__(self, pairs_csv, graphs_dir):
        self.graphs_dir = graphs_dir
        self.pairs = self._filter_pairs(pairs_csv)
        if len(self.pairs) == 0:
            print("❌ 未读取到有效图对数据，请检查CSV文件")
            sys.exit(1)
    def _filter_pairs(self, pairs_csv):
        valid_pairs = []
        missing_files = []
        invalid_format = 0
        invalid_label = 0
        total_lines = 0

        # 直接逐行读，不再 f.readline() 跳过第一行，
        # 而是根据内容判断 header（更稳健）
        with open(pairs_csv, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                total_lines += 1
                line = line.strip()
                if not line:
                    continue

                # 分割并去除多余的引号/空白
                parts = [p.strip().strip('"').strip("'") for p in line.split(",")]

                # 跳过显式 header 行（无论出现在哪）
                if len(parts) >= 2 and parts[0].lower() == "graph1_id" and parts[1].lower() == "graph2_id":
                    continue

                if len(parts) != 3:
                    invalid_format += 1
                    # 可选地打印错误样例（注释掉避免太多输出）
                    # print(f"⚠️ CSV第{line_num}行格式错误（需3列），跳过：{line}")
                    continue

                a, b, l = parts

                # 检查标签是否为整数（支持负号但你一般不会有）
                if not re.fullmatch(r"-?\d+", l):
                    invalid_label += 1
                    # print(f"⚠️ CSV第{line_num}行标签非整数，跳过：{line}")
                    continue

                l_int = int(l)

                # 检查对应的 .pt 文件是否存在
                a_pt = os.path.join(self.graphs_dir, f"{a}.pt")
                b_pt = os.path.join(self.graphs_dir, f"{b}.pt")
                if os.path.exists(a_pt) and os.path.exists(b_pt):
                    valid_pairs.append((a, b, l_int))
                else:
                    missing_files.append((a, b))

        # 打印统计信息，便于排查
        print(f"📊 CSV 总行数：{total_lines}，有效 pairs：{len(valid_pairs)}")
        if invalid_format:
            print(f"⚠️ 格式错误行数（列数不为3）：{invalid_format}")
        if invalid_label:
            print(f"⚠️ 标签非整数行数：{invalid_label}")
        if missing_files:
            # 仅展示前 10 个示例，避免刷屏
            print(f"⚠️ 找不到对应 .pt 的对（示例最多10条，共 {len(missing_files)} 条）：")
            for a, b in missing_files[:10]:
                print(f"    missing: {a}.pt , {b}.pt")
            print("    （请确认 --graphs_dir 路径是否正确，或文件名是否含前缀/后缀差异）")
        return valid_pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        a, b, l = self.pairs[idx]
        g1 = load_graph_from_pt(self.graphs_dir, a)
        g2 = load_graph_from_pt(self.graphs_dir, b)
        if not (check_edges(g1, a) and check_edges(g2, b)):
            return self.__getitem__((idx + 1) % len(self))
        return g1, g2, torch.tensor(l, dtype=torch.float32)
    
    

# -------------------------
# 分类头（直接用基座模型 + 二分类逻辑，不再另 train）
# -------------------------
class SimilarityHead(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
    def forward(self, e1, e2):
        x = torch.cat([e1, e2, torch.abs(e1 - e2)], dim=-1)
        return self.fc(x).squeeze(-1)

# -------------------------
# 图嵌入器（复用训练脚本模型，直接输出分类结果）
# -------------------------
class GraphEmbedder:
    def __init__(self, model_dir, device="cpu"):
        self.device = device
        self.config = AIGMAEConfig.from_pretrained(model_dir)
        self.model = AIGMAEModel_cross_finetune_head.from_pretrained(model_dir, config=self.config)
        self.model.to(device)
        self.model.eval()
        if not hasattr(self.model.graph_encoder, 'graph_rep'):
            setattr(self.model.graph_encoder, 'graph_rep', torch.tensor([], device=device))
    def get_graph_embedding(self, data: Data):
        data = data.to(self.device)

        # token-only 协议：只使用 token_emb
        input_nodes = data.x.long().squeeze(-1).unsqueeze(0)   # [1, N]

        with torch.no_grad():
            node_emb = self.model.token_emb(input_nodes.squeeze(0).long())  # [N, H] 或 [1, N, H]

        if node_emb.dim() == 1:
            node_emb = node_emb.unsqueeze(0)
        elif node_emb.dim() > 2:
            node_emb = node_emb.view(-1, node_emb.size(-1))

        node_emb = node_emb.float()
        g_emb = node_emb.mean(dim=0)   # [H]

        return g_emb

# -------------------------
# 命令行参数定义
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="MGVGA 等价性模型评估脚本（对齐训练逻辑）")
    parser.add_argument("--pretrained", default="/lcm/mgvga/model_attention_0.2", help="预训练模型目录（与训练脚本一致）")
    parser.add_argument("--graphs_dir", required=True, help="PT文件目录（与训练脚本一致）")
    parser.add_argument("--pairs_csv", required=True, help="评估用的图对CSV路径（测试集）")
    parser.add_argument("--batch_size", type=int, default=64, help="评估批次大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（确保复现性）")
    parser.add_argument("--head_ckpt", type=str, default="/lcm/mgvga/head1/similarity_head.pt", help="Path to trained similarity_head.pt")

    return parser.parse_args()

# -------------------------
# 保存指标汇总表格（脚本同级目录，仅4个核心指标）
# -------------------------
def save_summary_table(dataset_name, auc, precision, recall, f1):
    # 表格保存路径：脚本同级目录下的 evaluation_summary.csv
    summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_summary.csv")
    
    # 检查文件是否存在，不存在则创建并写入表头
    file_exists = os.path.exists(summary_path)
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            # 表头：数据集名称 + 4个核心指标
            writer.writerow(["数据集名称", "AUC", "Precision", "Recall", "F1-Score"])
        # 写入当前数据集的指标（保留4位小数）
        writer.writerow([
            dataset_name,
            round(auc, 4),
            round(precision, 4),
            round(recall, 4),
            round(f1, 4)
        ])
    print(f"\n✅ 指标汇总已保存到：{summary_path}")

def evaluate(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")

    dataset_name = os.path.splitext(os.path.basename(args.pairs_csv))[0]
    print(f"📥 加载数据集：{dataset_name}（路径：{args.pairs_csv}）")

    dataset = EquivPairsDataset(args.pairs_csv, args.graphs_dir)
    
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x)
#     data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=eq_collate_fn)
    
    print(f"✅ 数据集加载完成：共 {len(dataset)} 个有效图对")

    embedder = GraphEmbedder(args.pretrained, device=device)
    first_sample = dataset[0]
    emb_dim = embedder.get_graph_embedding(first_sample[0]).shape[0]
    head = SimilarityHead(emb_dim).to(device)
    if args.head_ckpt is not None:
        sd = torch.load(args.head_ckpt, map_location="cpu")
        head.load_state_dict(sd, strict=True)
        print(f"✅ Loaded head weights from {args.head_ckpt}")

    config = AIGMAEConfig.from_pretrained(args.pretrained)
    model = AIGMAEModel_cross_finetune_head.from_pretrained(args.pretrained, config=config)
    model.to(device)
    model.eval()
    global NODE_MAX_CLASS
    # vocab= num_classes+1, 其中 0 是 padding，所以原始类别最大值 = num_embeddings - 2
    NODE_MAX_CLASS = int(model.token_emb.node_token_emb.num_embeddings) - 2
    print(f"[Eval] NODE_MAX_CLASS set to {NODE_MAX_CLASS} (from vocab={model.token_emb.node_token_emb.num_embeddings})")

    all_labels, all_preds = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估进度"):
            g1_list, g2_list, label_list = zip(*batch)
            e1 = torch.stack([embedder.get_graph_embedding(g) for g in g1_list]).to(device)
            e2 = torch.stack([embedder.get_graph_embedding(g) for g in g2_list]).to(device)
            preds_logits = head(e1, e2)
            preds = torch.sigmoid(preds_logits).cpu().numpy()
            all_labels.extend([int(x.item()) for x in label_list])
            all_preds.extend(preds.tolist())


    # --------------------------
    # 整体指标计算和打印（保留）
    # --------------------------
    ys = np.array(all_labels)
    #ys = 1-ys
    yps = np.array(all_preds)

    if len(set(ys)) < 2:
        auc, optimal_threshold = 0.0, 0.5
        preds_bin = (yps >= optimal_threshold).astype(int)
    else:
        fpr, tpr, thresholds = roc_curve(ys, yps)
        auc = roc_auc_score(ys, yps)
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]
        preds_bin = (yps >= optimal_threshold).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(ys, preds_bin, average="binary", zero_division=0)

    print("\n" + "="*50)
    print(f"📋 {dataset_name} 整体评估结果")
    print("="*50)
    print(f"AUC：{auc:.4f}, Precision：{prec:.4f}, Recall：{rec:.4f}, F1-Score：{f1:.4f}")
    print("="*50)

    save_summary_table(dataset_name, auc, prec, rec, f1)

    # --------------------------
    # ⭐ 按原始文件前缀统计子数据集指标
    # --------------------------
    grouped = {}
    for (a, b, l), pred_score in zip(dataset.pairs, all_preds):
        prefix = a.split("_")[0]
        if prefix not in grouped:
            grouped[prefix] = {"labels": [], "scores": []}
        grouped[prefix]["labels"].append(int(l))
        grouped[prefix]["scores"].append(float(pred_score))

    print("\n" + "="*80)
    print("📌 按原始文件前缀分别统计各子数据集指标")
    print("="*80)

    for prefix, data in grouped.items():
        ys_sub, yps_sub = np.array(data["labels"]), np.array(data["scores"])
        #ys_sub = 1 - ys_sub
        if len(set(ys_sub)) < 2:
            print(f"\n⚠️ 子数据集 {prefix} 标签只有一类，跳过指标计算")
            continue
        fpr, tpr, thresholds = roc_curve(ys_sub, yps_sub)
        auc_sub = roc_auc_score(ys_sub, yps_sub)
        best_th = thresholds[np.argmax(tpr - fpr)]
        preds_bin = (yps_sub >= best_th).astype(int)
        prec_sub, rec_sub, f1_sub, _ = precision_recall_fscore_support(
            ys_sub, preds_bin, average="binary", zero_division=0
        )

        print("\n" + "-"*60)
        print(f"📌 子数据集：{prefix}")
        print("-"*60)
        print(f"样本数：{len(ys_sub)}")
        print(f"AUC：{auc_sub:.4f}, Precision：{prec_sub:.4f}, Recall：{rec_sub:.4f}, F1-Score：{f1_sub:.4f}")
        print(f"最优阈值：{best_th:.4f}")
        print("-"*60)


def main():
    args = parse_args()
    evaluate(args)

if __name__ == "__main__":
    main()    