#!/usr/bin/env python3
# =============================================================
# MGVGA QoR 模型评估脚本（无训练分类头，直接用普通分类头）
# 核心：复用基座模型 + 普通回归头，无需加载额外训练的分类头权重
# 对齐：与 eval_eq.py 数据格式、参数设置、评估逻辑完全一致
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
from sklearn.metrics import ndcg_score
import numpy as np
import traceback
import random
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# -------------------------
# 固定随机种子（与 eval_eq.py 一致）
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
# 添加模型路径（与 eval_eq.py 一致）
# -------------------------
sys.path.append("/root/autodl-tmp/MGVGA-master/")

# -------------------------
# 导入模型（与 eval_eq.py 一致）
# -------------------------
try:
    from aigmae.configuration_vgmae import AIGMAEConfig
    from aigmae.modeling_vgmae_attention import AIGMAEModel_cross_finetune_head,AIGMAEModel_cross
except ImportError as e:
    print(f"ERROR: 无法导入自定义模型，请检查 aigmae 模块路径！")
    print(f"错误信息: {e}")
    sys.exit(1)

# -------------------------
# 辅助：查找PT文件（适配你的数据格式）
# -------------------------
def find_pt_path_in_dir(subdir_path: str, seq_id: int):
    candidates = [
        f"seq_{seq_id}.pt",
        f"seq_{seq_id:03d}.pt",
        f"seq_{seq_id:04d}.pt",
        f"seq_{seq_id:05d}.pt",
    ]
    for name in candidates:
        p = os.path.join(subdir_path, name)
        if os.path.exists(p):
            return p
    
    target_pattern = re.compile(rf"seq_{seq_id}")
    try:
        for fname in os.listdir(subdir_path):
            if fname.endswith(".pt") and target_pattern.search(fname):
                return os.path.join(subdir_path, fname)
    except Exception as e:
        print(f"⚠️ 遍历文件夹 {subdir_path} 失败：{e}")
    return None

# -------------------------
# 加载图结构（与 eval_eq.py 逻辑一致，补充日志）
# -------------------------
def load_graph_from_subdir(subdir_path, seq_id):
    pt_path = find_pt_path_in_dir(subdir_path, seq_id)
    if pt_path is None:
        raise FileNotFoundError(f"PT文件未找到：seq_id={seq_id} in {subdir_path}")
    
    try:
        data = torch.load(pt_path, map_location="cpu")
        if not isinstance(data, Data):
            raise RuntimeError(f"{pt_path} 不是合法的PyG Data对象")
        if data.edge_index.dim() != 2 or data.edge_index.shape[0] != 2:
            raise RuntimeError(f"{pt_path} 的 edge_index 维度错误：{data.edge_index.shape}")
        return data
    except Exception as e:
        print(f"❌ 加载PT文件 {pt_path} 失败：{e}")
        raise

# -------------------------
# 门数提取（适配 qor_summary.csv 格式）
# -------------------------
def extract_gate_count_from_row(row):
    candidates = [
        "optimized_gate_count",
        "optimized_gate",
        "gate_count",
        "and_gate_count",
    ]
    for c in candidates:
        if c in row and pd.notna(row[c]):
            try:
                return float(row[c])
            except Exception:
                continue
    return 0.0

# -------------------------
# 数据集定义（与 eval_eq.py  Dataset 逻辑对齐）
# -------------------------
class QoRDataset(Dataset):
    def __init__(self, root_dir, skip_subdirs=None):
        self.root_dir = Path(root_dir)
        self.skip_subdirs = set(skip_subdirs or [])
        self.graph_info = self._load_graph_info()
        if len(self.graph_info) == 0:
            print("❌ 未读取到有效图数据，请检查数据集目录")
            sys.exit(1)

    def _load_graph_info(self):
        graph_info = []
        # 新增：用于统计每个子文件夹的样本数
        folder_stats = defaultdict(lambda: {"csv_total": 0, "valid_samples": 0})  # 每个文件夹的统计信息
        missing_info = defaultdict(list)
        total_csv_rows = 0
        total_missing = 0
        print(f"\n🔍 【日志】开始加载 graph_info（遍历子目录）")

        for subdir in sorted(os.listdir(self.root_dir)):
            subdir_path = os.path.join(self.root_dir, subdir)
            print(f"\n📂 【日志】处理子目录：{subdir_path}")
            if not os.path.isdir(subdir_path):
                print(f"⏭ 【日志】跳过非目录：{subdir_path}")
                continue
            if subdir in self.skip_subdirs:
                print(f"⏭ 【日志】跳过指定文件夹：{subdir}")
                folder_stats[subdir]["skip_reason"] = "指定跳过"
                continue

            # 检查 qor_summary.csv
            qor_csv = os.path.join(subdir_path, "qor_summary.csv")
            if not os.path.exists(qor_csv):
                print(f"⏭ 【日志】无 qor_summary.csv，跳过子目录：{subdir}")
                folder_stats[subdir]["skip_reason"] = "缺少 qor_summary.csv"
                continue

            # 读取CSV
            try:
                df = pd.read_csv(qor_csv)
                csv_total = len(df) - 1  # CSV总行数 -1（排除表头）
                folder_stats[subdir]["csv_total"] = csv_total  # 记录CSV总行数
                print(f"✅ 【日志】读取 qor_summary.csv：{qor_csv}，有效行数（除表头）={csv_total}")
                if "graph_id" not in df.columns:
                    print(f"❌ 【日志】CSV无 'graph_id' 列，跳过：{qor_csv}")
                    folder_stats[subdir]["skip_reason"] = "CSV无 graph_id 列"
                    continue
            except Exception as e:
                print(f"❌ 【日志】读取 CSV {qor_csv} 失败：{e}")
                folder_stats[subdir]["skip_reason"] = f"CSV读取失败：{str(e)[:20]}..."
                continue

            # 处理CSV每一行
            valid_count = 0  # 统计当前文件夹的有效样本数
            total_csv_rows += csv_total
            for _, row in df.iterrows():
                csv_graph_id = str(row.get("graph_id", "")).strip()
                if not csv_graph_id:
                    total_missing += 1
                    continue
                m = re.search(r"seq_(\d+)", csv_graph_id)
                if not m:
                    total_missing += 1
                    continue
                seq_id = int(m.group(1))
                if find_pt_path_in_dir(subdir_path, seq_id) is None:
                    missing_info[subdir].append(seq_id)
                    total_missing += 1
                    continue

                # 提取门数并添加到 graph_info
                gate_count = extract_gate_count_from_row(row)
                graph_info.append((seq_id, subdir_path, gate_count, csv_graph_id))
                valid_count += 1  # 有效样本数+1

            # 记录当前文件夹的有效样本数
            folder_stats[subdir]["valid_samples"] = valid_count
            print(f"📊 【日志】子目录 {subdir}：CSV有效行数={csv_total}，有效样本数（CSV-PT匹配）={valid_count}")

        # -------------------------
        # 新增：打印所有子文件夹的统计汇总
        # -------------------------
        print(f"\n" + "="*80)
        print("📋 各子文件夹有效样本数统计（按论文4.1节 1500样本/文件夹要求）")
        print("="*80)
        print(f"{'子文件夹名':<20} {'CSV有效行数':<15} {'有效样本数':<15} {'状态/原因':<20}")
        print("-"*80)
        for subdir in sorted(folder_stats.keys()):
            stats = folder_stats[subdir]
            # 处理状态显示
            if "skip_reason" in stats:
                status = stats["skip_reason"]
            else:
                # 对比论文要求的1500样本，标记是否达标
                if stats["valid_samples"] >= 1450:  # 允许少量缺失（如越界边过滤）
                    status = "达标（接近1500）"
                elif stats["valid_samples"] > 0:
                    status = f"样本不足（需1500）"
                else:
                    status = "无有效样本"
            # 打印该行统计
            print(f"{subdir:<20} {stats.get('csv_total', 0):<15} {stats.get('valid_samples', 0):<15} {status:<20}")
        print("="*80)
        print(f"📊 全局统计：")
        print(f"   - 总子文件夹数：{len(folder_stats)}")
        print(f"   - 总CSV有效行数：{total_csv_rows}")
        print(f"   - 总有效样本数：{len(graph_info)}")
        print(f"   - 总缺失样本数（CSV-PT不匹配等）：{total_missing}")
        print("="*80)

        # 保存缺失信息（原逻辑保留）
        output_file = os.path.join(str(self.root_dir), "missing_pt_list.txt")
        with open(output_file, "w") as f:
            for sd, seqs in missing_info.items():
                f.write(f"{sd}: {seqs}\n")
        print(f"✅ 【日志】缺失PT文件信息已保存到：{output_file}")
        return graph_info

    def __len__(self):
        return len(self.graph_info)

    def __getitem__(self, idx):
        seq_id, subdir_path, gate_count, graph_id = self.graph_info[idx]
        subdir_name = os.path.basename(subdir_path)  # 提取电路名称（如 div、sin）
        graph_data = load_graph_from_subdir(subdir_path, seq_id)
        if not check_edges(graph_data, graph_id):
            return self.__getitem__((idx + 1) % len(self))
        # 返回值增加 subdir_name（电路名称）
        return graph_data, torch.tensor(gate_count, dtype=torch.float32), subdir_name



# -------------------------
# 普通QoR回归头（无预训练权重，直接用基座模型输出）
# 与 eval_eq.py 的 SimilarityHead 结构一致，适配回归任务
# -------------------------
class QoRRegressionHead(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # 结构与 eval_eq.py SimilarityHead 对齐，仅输出改为回归值
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, graph_emb):
        x = self.fc(graph_emb)
        return x.squeeze(-1)  # 回归任务输出标量
    

# -------------------------
# 越界边检查（与 eval_eq.py 完全一致）
# -------------------------
def check_edges(data, graph_id=None):
    num_nodes = data.x.size(0)
    edge_index = data.edge_index
    mask = (edge_index >= num_nodes).any(dim=0)
    if mask.any():
        print(f"❌ 图 {graph_id} 含 {mask.sum().item()} 条越界边 → 丢弃")
        return False
    return True
# -------------------------
# 图嵌入器（与 eval_eq.py 的 GraphEmbedder 完全一致）
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
        print("[Embedder] has_final_node_emb:", hasattr(self.model.graph_encoder, "final_node_emb"))

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
# QoR评估指标计算（按论文4.3节要求：NDCG@k + Top-k% Commonality）
# -------------------------
def calculate_qor_metrics(y_true, y_pred, subdirs, k_list=[3, 5, 10]):
    """
    按论文要求：分电路计算指标，再取所有电路的平均值
    y_true: 真实门数列表
    y_pred: 预测门数列表
    subdirs: 电路名称列表（与y_true/y_pred一一对应）
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    subdirs = np.array(subdirs)
    metrics = {
        "ndcg": {k: [] for k in k_list},  # 每个k对应的各电路NDCG
        "top_k_commonality": {k: [] for k in k_list}  # 每个k对应的各电路Commonality
    }

    # 按电路分组计算指标
    unique_circuits = np.unique(subdirs)
    for circuit in unique_circuits:
        # 提取当前电路的所有样本
        circuit_mask = (subdirs == circuit)
        circ_true = y_true[circuit_mask]
        circ_pred = y_pred[circuit_mask]
        if len(circ_true) < 10:  # 样本数过少，跳过（避免统计偏差）
            continue

        # 1. 计算当前电路的 NDCG（论文核心指标）
        relevance = -circ_true  # 门数越少，相关性越高（与论文一致）
        relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-8)  # 归一化
        for k in k_list:
            if len(circ_true) < k:
                ndcg = 0.0
            else:
                ndcg = ndcg_score([relevance], [-circ_pred], k=k)
            metrics["ndcg"][k].append(ndcg)

        # 2. 计算当前电路的 Top-k% Commonality（论文辅助指标）
        total_circ_samples = len(circ_true)
        for k in k_list:
            k_count = max(1, int(total_circ_samples * k / 100))  # 每个电路的Top-k%样本数（至少1个）
            # 真实Top-k%：当前电路内门数最少的k_count个样本
            true_top_idx = np.argsort(circ_true)[:k_count]
            true_top_set = set(true_top_idx)
            # 预测Top-k%：当前电路内预测门数最少的k_count个样本
            pred_top_idx = np.argsort(circ_pred)[:k_count]
            pred_top_set = set(pred_top_idx)
            # 计算重合度
            common = len(true_top_set & pred_top_set)
            commonality = common / k_count if k_count > 0 else 0.0
            metrics["top_k_commonality"][k].append(commonality)

    # 计算所有电路的平均指标（与论文Table 1输出格式一致）
    final_metrics = {
        "ndcg": {},
        "top_k_commonality": {}
    }
    for k in k_list:
        # NDCG取平均
        avg_ndcg = np.mean(metrics["ndcg"][k]) if metrics["ndcg"][k] else 0.0
        final_metrics["ndcg"][f"ndcg@{k}"] = round(avg_ndcg, 4)
        # Top-k% Commonality取平均
        avg_commonality = np.mean(metrics["top_k_commonality"][k]) if metrics["top_k_commonality"][k] else 0.0
        final_metrics["top_k_commonality"][f"top_{k}%_commonality"] = round(avg_commonality, 4)

    return final_metrics, unique_circuits  # 返回平均指标和电路列表


# -------------------------
# 命令行参数（与 eval_eq.py 格式一致，移除 head_path 参数）
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="MGVGA QoR 模型评估脚本（无训练分类头）")
    parser.add_argument("--pretrained", default="/lcm/mgvga/model_attention_0.2", help="预训练模型目录（与 eval_eq.py 一致）")
    parser.add_argument("--data_root", required=True, help="QoR数据集根目录")
    parser.add_argument("--batch_size", type=int, default=64, help="评估批次大小（与 eval_eq.py 一致）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（与 eval_eq.py 一致）")
    parser.add_argument("--skip_file", type=str, default="", help="以逗号分隔的跳过文件夹列表")
    parser.add_argument("--save_result", action="store_true", help="是否保存评估结果到CSV")
#     parser.add_argument("--qor_head_ckpt", type=str, default="/lcm/mgvga/test_qor14/qor_head.pt", help="few-shot 训练得到的 qor_head.pt 路径")
    parser.add_argument("--qor_head_ckpt", type=str, default="/lcm/mgvga/head1/qor_head.pt", help="few-shot 训练得到的 qor_head.pt 路径")
    return parser.parse_args()

# -------------------------
# 核心评估函数（与 eval_eq.py 逻辑对齐）
# -------------------------
def evaluate(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")

    # 加载数据集
    print(f"📥 加载数据集：{args.data_root}")
    skip_list = [s.strip() for s in args.skip_file.split(",") if s.strip()]
    dataset = QoRDataset(args.data_root, skip_subdirs=skip_list)
    collate_fn = lambda x: (
        [item[0] for item in x],  # 图数据列表
        torch.stack([item[1] for item in x]),  # 门数标签
        [item[2] for item in x]  # 电路名称列表（新增）
    )
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    print(f"✅ 数据集加载完成：共 {len(dataset)} 个有效样本")
    # 加载预训练模型和嵌入器
    print(f"📥 加载预训练 MGVGA 模型：{args.pretrained}")
    embedder = GraphEmbedder(args.pretrained, device=device)
    first_graph = dataset[0][0]
    emb_dim = embedder.get_graph_embedding(first_graph).shape[0]
    print(f"✅ 嵌入维度确定：{emb_dim}")

    # 初始化普通回归头（无预训练权重，直接使用）
    print("\n📊 开始评估...")

    qor_head = QoRRegressionHead(emb_dim).to(device)
    sd = torch.load(args.qor_head_ckpt, map_location="cpu")
    qor_head.load_state_dict(sd, strict=True)
    qor_head.eval()


    criterion = nn.MSELoss()  # 新增：定义回归任务的损失函数（与论文一致）
    total_loss = 0.0  # 新增：初始化总损失，用于后续平均损失计算

    # 评估过程中，收集电路名称
    all_true = []
    all_pred = []
    all_subdirs = []  # 新增：存储每个样本的电路名称
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="评估进度")
        for batch_graphs, batch_labels, batch_subdirs in pbar:  # 接收批次电路名称
            # 1. 提取当前批次所有图的嵌入（原逻辑保留）
            batch_embs = torch.stack([embedder.get_graph_embedding(g) for g in batch_graphs]).to(device)
            # 2. 模型预测（新增：计算preds）
            batch_labels = batch_labels.to(device)
            preds = qor_head(batch_embs)  # 回归头输出预测门数
            # 3. 计算损失（补充：用于后续平均损失计算，原逻辑遗漏）
            loss = criterion(preds, batch_labels)
            total_loss += loss.item() * len(batch_graphs)
            # 4. 收集结果
            all_true.extend(batch_labels.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())  # 此时preds已定义
            all_subdirs.extend(batch_subdirs)  # 收集电路名称
            # 5. 更新进度条（可选，增强可读性）
            avg_loss = total_loss / len(all_true) if all_true else 0.0
            pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}", "样本数": len(all_true)})


    # 调用分电路指标计算函数
    print("\n🧮 计算评估指标（按论文要求分电路统计）...")
    metrics, circuits = calculate_qor_metrics(all_true, all_pred, all_subdirs)
    avg_loss = total_loss / len(all_true)

    # 输出结果（与论文Table 1格式一致，先输出各电路详细结果，再输出平均）
    print("\n" + "="*100)
    print("📋 各电路 QoR 评估结果（论文4.3节格式）")
    print("="*100)
    print(f"{'电路名称':<15} {'ndcg@3':<10} {'ndcg@5':<10} {'ndcg@10':<10} {'top_3%_commonality':<20} {'top_5%_commonality':<20} {'top_10%_commonality':<20}")
    print("-"*100)
    # 补充分电路详细结果打印（便于验证）
    y_true_np = np.array(all_true)
    y_pred_np = np.array(all_pred)
    subdirs_np = np.array(all_subdirs)
    for circuit in circuits:
        mask = (subdirs_np == circuit)
        circ_true = y_true_np[mask]
        circ_pred = y_pred_np[mask]
        # 计算当前电路的详细指标
        relevance = -circ_true
        relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-8)
        ndcg3 = ndcg_score([relevance], [-circ_pred], k=3) if len(circ_true)>=3 else 0.0
        ndcg5 = ndcg_score([relevance], [-circ_pred], k=5) if len(circ_true)>=5 else 0.0
        ndcg10 = ndcg_score([relevance], [-circ_pred], k=10) if len(circ_true)>=10 else 0.0
        # 计算当前电路的Commonality
        total = len(circ_true)
        k3_count = max(1, int(total*3/100))
        true3_idx = set(np.argsort(circ_true)[:k3_count])
        pred3_idx = set(np.argsort(circ_pred)[:k3_count])
        common3 = len(true3_idx & pred3_idx) / k3_count
        # 同理计算k=5和k=10
        k5_count = max(1, int(total*5/100))
        true5_idx = set(np.argsort(circ_true)[:k5_count])
        pred5_idx = set(np.argsort(circ_pred)[:k5_count])
        common5 = len(true5_idx & pred5_idx) / k5_count
        k10_count = max(1, int(total*10/100))
        true10_idx = set(np.argsort(circ_true)[:k10_count])
        pred10_idx = set(np.argsort(circ_pred)[:k10_count])
        common10 = len(true10_idx & pred10_idx) / k10_count
        # 打印当前电路结果
        print(f"{circuit:<15} {ndcg3:.4f} {'':<6} {ndcg5:.4f} {'':<6} {ndcg10:.4f} {'':<6} {common3:.4f} {'':<12} {common5:.4f} {'':<12} {common10:.4f}")

    # 输出平均结果（与论文最终报告格式一致）
    print("\n" + "="*100)
    print("📋 QoR 最终评估结果（所有电路平均值，论文对齐版）")
    print("="*100)
    print(f"总评估样本数：{len(all_true)}（{len(circuits)}个电路，每个电路1500个样本）")
    print(f"平均测试损失（MSE）：{avg_loss:.4f}")
    print(f"NDCG 指标：{metrics['ndcg']}")
    print(f"Top-k% Commonality 指标：{metrics['top_k_commonality']}")
    print("="*100)

    
    # 输出平均结果后，复用y_true_np
    print(f"\n🔍 门数分布统计（论文要求：门数需有显著差异）：")
    # y_true = np.array(all_true)  # 删除冗余定义
    print(f"   - 门数范围：{y_true_np.min():.0f} ~ {y_true_np.max():.0f}")
    print(f"   - 门数标准差：{y_true_np.std():.0f}")
    print(f"   - 门数中位数：{np.median(y_true_np):.0f}")
    print(f"   - Top-3%门数阈值：{np.sort(y_true_np)[:int(len(y_true_np)*0.03)][-1]:.0f}")


    # 保存结果
    if args.save_result:
        output_dir = args.data_root
        os.makedirs(output_dir, exist_ok=True)
        result_csv = os.path.join(output_dir, "qor_evaluation_result.csv")
        with open(result_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "评估时间", "样本数", "平均损失(MSE)",
                "ndcg@3", "ndcg@5", "ndcg@10",
                "top_3%_commonality", "top_5%_commonality", "top_10%_commonality"
            ])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                len(all_true), round(avg_loss, 4),
                metrics["ndcg"]["ndcg@3"],
                metrics["ndcg"]["ndcg@5"],
                metrics["ndcg"]["ndcg@10"],
                metrics["top_k_commonality"]["top_3%_commonality"],
                metrics["top_k_commonality"]["top_5%_commonality"],
                metrics["top_k_commonality"]["top_10%_commonality"]
            ])
        print(f"✅ 评估结果已保存到：{result_csv}")

# -------------------------
# 主函数（与 eval_eq.py 一致）
# -------------------------
def main():
    args = parse_args()
    evaluate(args)

if __name__ == "__main__":
    main()
