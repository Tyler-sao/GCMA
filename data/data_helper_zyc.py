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
        # 解包输入数据
        graphs = [i[0] for i in inputs]
        vgraphs = [i[1] for i in inputs]
        batch_size = len(graphs)
        batch = {}

        # ----------------- MGM 数据处理 -----------------
        # 计算最大节点数和边数（处理空图情况）
        max_node_num = max((g['input_nodes'].shape[0] for g in graphs if 'input_nodes' in g), default=0)
        max_edge_num = max((g['input_edges'].shape[-1] for g in graphs if 'input_edges' in g), default=0)

        # 初始化张量（使用0填充，确保维度正确）
        batch["input_nodes"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["input_edges"] = torch.zeros(batch_size, 2, max_edge_num, dtype=torch.long)
        batch["node_mask"] = torch.zeros(batch_size, max_node_num, dtype=torch.bool)
        batch["padding_mask"] = torch.zeros(batch_size, max_node_num, dtype=torch.bool)
        batch["input_indegree"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["input_outdegree"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["labels"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)

        for idx, g in enumerate(graphs):
            # 安全获取节点和边数据（处理缺失键的情况）
            input_nodes = g.get("input_nodes", torch.tensor([], dtype=torch.long))
            node_num = input_nodes.shape[0]
            
            edge_index = g.get("input_edges", torch.empty((2, 0), dtype=torch.long))
            edge_index = edge_index if edge_index.dim() == 2 and edge_index.shape[0] == 2 else torch.empty((2, 0), dtype=torch.long)

            edge_num = edge_index.shape[1] if edge_index.numel() > 0 else 0
            # ----------------- 修复1：提前初始化total_degree（分支外定义） -----------------
            total_degree = torch.tensor([], dtype=torch.long)  # 空张量初始化
            # 仅在节点数>0时计算真实度数
            if node_num > 0:
                indegree = degree(edge_index[0], num_nodes=node_num) if edge_num > 0 else torch.zeros(node_num, dtype=torch.long)
                outdegree = degree(edge_index[1], num_nodes=node_num) if edge_num > 0 else torch.zeros(node_num, dtype=torch.long)
                total_degree = indegree + outdegree  # 覆盖为空张量，赋值真实总度数
            # 处理边数据（确保索引不越界）
            if edge_num > 0 and max_edge_num > 0:
                # 限制边数不超过max_edge_num
                valid_edge_num = min(edge_num, max_edge_num)
                batch["input_edges"][idx, 0, :valid_edge_num] = edge_index[0, :valid_edge_num]
                batch["input_edges"][idx, 1, :valid_edge_num] = edge_index[1, :valid_edge_num]
                # 填充剩余部分（使用0而非第一个元素，避免无效索引）
                if valid_edge_num < max_edge_num:
                    batch["input_edges"][idx, :, valid_edge_num:] = 0

            # 处理节点数据
            if node_num > 0 and max_node_num > 0:
                # 确保节点ID有效（非负且不超过节点数）
                safe_nodes = input_nodes.clone().clamp(0, node_num - 1)
                batch["input_nodes"][idx, :node_num] = safe_nodes.view(-1) + 1
                batch["padding_mask"][idx, :node_num] = True  # 标记有效节点
                batch["input_indegree"][idx, :node_num] = indegree
                batch["input_outdegree"][idx, :node_num] = outdegree
                batch["labels"][idx, :node_num] = safe_nodes.squeeze(-1)

                # 生成掩码（确保掩码范围在有效节点内）
                masked_node_num = max(1, int(node_num * self.mgm_mask_ratio))
                masked_node_num = min(masked_node_num, node_num)  # 避免超出节点数
                 # ----------------- 修正1：加载QoR关键节点掩码（避免变量未定义） -----------------
                # 从样本g中安全获取key_node_mask_qor，无则默认全False（所有节点都是非关键节点）
                key_node_mask = g.get("key_node_mask_qor", torch.zeros(node_num, dtype=torch.bool))
                # 处理维度：若key_node_mask是二维（如[N,1]），转为一维（[N,]），避免索引错误
                if key_node_mask.dim() > 1:
                    key_node_mask = key_node_mask.squeeze(1)
                # 兜底：确保关键节点掩码长度与节点数一致（防止数据异常）
                if key_node_mask.shape[0] != node_num:
                    key_node_mask = torch.zeros(node_num, dtype=torch.bool)

                # ----------------- 修正2：用总度数筛选（原代码"degree"变量未定义，替换为total_degree） -----------------
                # 1. 分离关键节点和非关键节点（关键节点不参与掩码）
                non_key_indices = torch.where(~key_node_mask)[0]  # 非关键节点的索引
                # 2. 非关键节点中，优先掩码低总度数节点（总度数<=1，即孤立节点或仅1条边的节点）
                if len(non_key_indices) == 0:
                    if node_num > 0:
                        mask_indices = torch.tensor([random.randint(0, node_num-1)], dtype=torch.long)
                else:
                    if total_degree.numel() == 0:
                        total_degree = torch.zeros(node_num, dtype=torch.long)  # 空度数视为0
                    low_degree_non_key = non_key_indices[total_degree[non_key_indices] <= 4]
                    if len(low_degree_non_key) < masked_node_num:
                        remaining = masked_node_num - len(low_degree_non_key)
                        other_non_key = non_key_indices[total_degree[non_key_indices] > 1]
                        random_other = other_non_key[torch.randperm(len(other_non_key))[:remaining]] if len(other_non_key) > 0 else torch.tensor([], dtype=torch.long)
                        mask_indices = torch.cat([low_degree_non_key, random_other])
                    else:
                        mask_indices = low_degree_non_key[:masked_node_num]
                if len(mask_indices) == 0:
                    mask_indices = torch.tensor([non_key_indices[0]], dtype=torch.long) if len(non_key_indices) > 0 else torch.tensor([0], dtype=torch.long)
                # 4. 生成最终掩码（仅掩码筛选出的非关键节点）
                node_mask = torch.zeros(node_num, dtype=torch.bool)
                if len(mask_indices) > 0:  # 避免空索引报错
                    node_mask[mask_indices] = True
                # 赋值到批次掩码中
                batch["node_mask"][idx, :node_num] = node_mask

            # 确保至少有一个掩码节点（如果有节点）
            if node_num > 0 and batch["node_mask"][idx].sum() == 0:
                batch["node_mask"][idx, 0] = True
        # ----------------- VGAlign 数据处理（重点修复部分）-----------------
        # 计算最大节点数和边数（处理空图）
        max_vg_node_num = max((g['input_nodes'].shape[0] for g in vgraphs if 'input_nodes' in g), default=0)
        max_vg_edge_num = max((g['input_edges'].shape[-1] for g in vgraphs if 'input_edges' in g), default=0)

        # 初始化张量（保持原样）
        batch["input_vg_nodes"] = torch.zeros(batch_size, max_vg_node_num, dtype=torch.long)
        batch["input_vg_edges"] = torch.zeros(batch_size, 2, max_vg_edge_num, dtype=torch.long)
        batch["input_vg_verilogs"] = torch.zeros(batch_size, self.cross_hidden_size, dtype=torch.float)
        batch["vg_node_mask"] = torch.zeros(batch_size, max_vg_node_num, dtype=torch.bool)
        batch["vg_padding_mask"] = torch.zeros(batch_size, max_vg_node_num, dtype=torch.bool)
        batch["vg_labels"] = torch.zeros(batch_size, max_vg_node_num, dtype=torch.long)

        for idx, g in enumerate(vgraphs):
            # 安全获取节点数据（处理维度异常，原有逻辑保留）
            input_nodes = g.get("input_nodes", torch.tensor([], dtype=torch.long))
            if input_nodes.dim() > 1:
                input_nodes = input_nodes.squeeze()
            if input_nodes.dim() == 0:
                input_nodes = torch.tensor([], dtype=torch.long)
            node_num = input_nodes.shape[0]

            # 安全获取边数据（原有逻辑保留）
            edge_index = g.get("input_edges", torch.empty((2, 0), dtype=torch.long))
            edge_index = edge_index if edge_index.dim() == 2 and edge_index.shape[0] == 2 else torch.empty((2, 0), dtype=torch.long)
            edge_num = edge_index.shape[1] if edge_index.numel() > 0 else 0
            # 计算入度和出度（与MGM逻辑一致，避免空边错误）
            total_degree = torch.tensor([], dtype=torch.long)  # 空张量初始化
            if node_num > 0:
                # 仅在节点数>0时计算度数，避免空张量赋值
                indegree = degree(edge_index[0], num_nodes=node_num) if edge_num > 0 else torch.zeros(node_num, dtype=torch.long)
                outdegree = degree(edge_index[1], num_nodes=node_num) if edge_num > 0 else torch.zeros(node_num, dtype=torch.long)
                total_degree = indegree + outdegree
            # 处理边数据（避免索引越界，原有逻辑保留）
            if edge_num > 0 and max_vg_edge_num > 0:
                valid_edge_num = min(edge_num, max_vg_edge_num)
                batch["input_vg_edges"][idx, 0, :valid_edge_num] = edge_index[0, :valid_edge_num].clamp(0, node_num - 1)
                batch["input_vg_edges"][idx, 1, :valid_edge_num] = edge_index[1, :valid_edge_num].clamp(0, node_num - 1)
                if valid_edge_num < max_vg_edge_num:
                    batch["input_vg_edges"][idx, :, valid_edge_num:] = 0  # 用0填充

            # 处理节点数据（核心修复：替换随机掩码为结构化掩码）
            if node_num > 0 and max_vg_node_num > 0:
                # 确保节点ID有效（原有逻辑保留）
                safe_nodes = input_nodes.clone().clamp(0, node_num - 1)
                batch["input_vg_nodes"][idx, :node_num] = safe_nodes.view(-1) + 1
                batch["vg_padding_mask"][idx, :node_num] = True  # 标记有效节点
                batch["vg_labels"][idx, :node_num] = safe_nodes

                # 生成掩码（核心改动：结构化筛选，替换原有perm随机采样）
                masked_node_num = max(1, int(node_num * self.align_mask_ratio))
                masked_node_num = min(masked_node_num, node_num)  # 防止超出节点总数

                # ----------------- 步骤1：加载QoR关键节点掩码（与MGM逻辑完全对齐） -----------------
                key_node_mask = g.get("key_node_mask_qor", torch.zeros(node_num, dtype=torch.bool))
                # 处理维度异常（如[N,1]转[N,]）
                if key_node_mask.dim() > 1:
                    key_node_mask = key_node_mask.squeeze(1)
                # 兜底：确保掩码长度与节点数一致（避免数据异常）
                if key_node_mask.shape[0] != node_num:
                    key_node_mask = torch.zeros(node_num, dtype=torch.bool)

                # ----------------- 步骤2：结构化筛选掩码节点（优先低度数非关键节点） -----------------
                # 分离非关键节点（关键节点不参与掩码）
                non_key_indices = torch.where(~key_node_mask)[0]
                if len(non_key_indices) == 0:
                    if node_num > 0:
                        mask_indices = torch.tensor([random.randint(0, node_num-1)], dtype=torch.long)
                else:
                    # 仅执行一次筛选逻辑
                    if total_degree.numel() == 0:
                        total_degree = torch.zeros(node_num, dtype=torch.long)  # 空度数视为0
                    low_degree_non_key = non_key_indices[total_degree[non_key_indices] <= 4]#之前的最佳是1，现在改成其他数
                    if len(low_degree_non_key) < masked_node_num:
                        remaining = masked_node_num - len(low_degree_non_key)
                        other_non_key = non_key_indices[total_degree[non_key_indices] > 1]
                        random_other = other_non_key[torch.randperm(len(other_non_key))[:remaining]] if len(other_non_key) > 0 else torch.tensor([], dtype=torch.long)
                        mask_indices = torch.cat([low_degree_non_key, random_other])
                    else:
                        mask_indices = low_degree_non_key[:masked_node_num]
                if len(mask_indices) == 0:
                    mask_indices = torch.tensor([non_key_indices[0]], dtype=torch.long) if len(non_key_indices) > 0 else torch.tensor([0], dtype=torch.long)
                # ----------------- 步骤3：生成最终掩码（替换原有perm逻辑） -----------------
                vg_node_mask = torch.zeros(node_num, dtype=torch.bool)
                if len(mask_indices) > 0:  # 避免空索引赋值错误
                    vg_node_mask[mask_indices] = True
                # 赋值到批次掩码中
                batch["vg_node_mask"][idx, :node_num] = vg_node_mask

            # 确保至少有一个掩码节点（原有兜底逻辑保留）
            if node_num > 0 and batch["vg_node_mask"][idx].sum() == 0:
                batch["vg_node_mask"][idx, 0] = True  # 至少掩码第一个节点

            # 处理Verilog数据（确保维度正确，原有逻辑保留）
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

            if not osp.exists(filePathArchive):
                print(f"[Warning] 文件 {filePathArchive} 不存在，跳过")
                continue

            try:
                with ZipFile(filePathArchive) as myzip:
                    if filePathName in myzip.namelist():
                        self.files.append(f)
                    else:
                        print(f"[Warning] {filePathName} 不在压缩包中，跳过")
            except Exception as e:
                print(f"[Warning] 打开 {filePathArchive} 失败：{e}，跳过")
        print(f"✅ GraphDataset 初始化完成：共保留 {len(self.files)} 个有效样本（原始 {len(self.all_files)} 个）")
                
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
        if hasattr(g, 'edge_type'):
            edge_type = g.edge_type.squeeze() if g.edge_type.dim() > 1 else g.edge_type
            # 处理空 edge_type（无有效边时）
            if edge_type.numel() == 0:
                mask = torch.tensor([], dtype=torch.bool)
            else:
                # 确保 mask 长度与边数一致（防止索引越界）
                mask = (edge_type == 1)[:g.edge_index.shape[1]]
        else:
            # 无 edge_type 字段→空 mask（无新增节点）
            mask = torch.tensor([], dtype=torch.bool)

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
                # 提取前缀（去掉_eq部分）
                base_name = design.split("_eq")[0]
                if base_name in self.verilog_list:
                    self.pyg_list.append(f)
                else:
                    print(f"[Warning] {design} 对应基名 {base_name} 不在 verilog_list 中，跳过")

        self.designs = [d.split(".")[0] for d in self.pyg_list]


    def __len__(self):
        return len(self.designs)

    def __getitem__(self, i):
        design = self.designs[i]
        pyg_file = osp.join(self.pyg_path, f"{design}.pt")
        g = torch.load(pyg_file, map_location="cpu")

        # ---------------------- 1. 处理 edge_index：确保格式为 (2, E)（避免空边维度异常） ----------------------
        if g.edge_index.dim() != 2 or g.edge_index.shape[0] != 2:
            g.edge_index = torch.empty((2, 0), dtype=torch.long)
        num_edges = g.edge_index.shape[1]

        # ---------------------- 2. 修复核心：统一 node_type 维度为 (N, 1)（二维） ----------------------
        # 处理原始节点类型 g.node_type：强制转为 (N, 1) 二维张量
        if hasattr(g, 'node_type'):
            # 情况1：原始 node_type 是 1 维（如 [0,1,2]）→ 转为 (N,1)
            if g.node_type.dim() == 1:
                node_type = g.node_type.view(-1, 1)  # 关键：1维→二维，保持与 new_node_type 维度一致
            # 情况2：原始 node_type 是 2 维但列数≠1（如 (N,2)）→ 取第一列并转为 (N,1)
            elif g.node_type.dim() == 2:
                if g.node_type.shape[1] != 1:
                    node_type = g.node_type[:, 0].view(-1, 1)
                else:
                    node_type = g.node_type  # 已符合 (N,1)，直接使用
            # 其他异常维度（如3维）→ 报错提示
            else:
                raise ValueError(f"原始 node_type 维度异常：{g.node_type.dim()} 维，需1或2维")
        else:
            # 若无 node_type 字段（极端情况）→ 用全0填充（需根据实际场景调整）
            num_nodes = g.x.shape[0] if hasattr(g, 'x') else 0
            node_type = torch.zeros((num_nodes, 1), dtype=torch.long)

        num_nodes = node_type.shape[0]  # 原始节点数（基于统一后的二维张量）

        # ---------------------- 3. 处理 edge_type 和 mask：确保 mask 长度与边数一致 ----------------------
        if hasattr(g, 'edge_type'):
            edge_type = g.edge_type
            # 统一 edge_type 为 1 维（如 (E,1)→(E,)）
            if edge_type.dim() == 2:
                edge_type = edge_type.squeeze(dim=1)
            # 生成与边数匹配的 mask（避免索引越界）
            if num_edges == 0 or edge_type.numel() == 0:
                mask = torch.tensor([], dtype=torch.bool)
            else:
                mask = (edge_type == 1)[:num_edges]  # 截断到边数长度
        else:
            # 若无 edge_type 字段→空 mask（无新增节点）
            mask = torch.tensor([], dtype=torch.bool)

        # 分离边：not_edge_index（生成新边）、buff_edge_index（直接保留）
        not_edge_index = g.edge_index[:, mask] if mask.numel() > 0 else torch.empty((2, 0), dtype=torch.long)
        buff_edge_index = g.edge_index[:, ~mask] if mask.numel() > 0 else g.edge_index
        new_nodes_count = mask.sum().item()  # 新增节点数（与 edge_type=1 的边数一致）

        # ---------------------- 4. 生成新边：用 stack 确保格式正确，避免 vstack 错误 ----------------------
        u_new = torch.empty((2, 0), dtype=torch.long)
        v_new = torch.empty((2, 0), dtype=torch.long)

        if new_nodes_count > 0 and not_edge_index.numel() > 0:
            u, v = not_edge_index  # u/v 均为 (new_nodes_count,) 一维
            new_nodes = num_nodes + torch.arange(new_nodes_count, dtype=torch.long)  # 新增节点索引
            # 直接生成 (2, new_nodes_count) 二维边索引
            u_new = torch.stack([u, new_nodes], dim=0)  # 原始节点→新增节点
            v_new = torch.stack([new_nodes, v], dim=0)  # 新增节点→原始节点

        # ---------------------- 5. 生成 new_node_type：强制为 (new_nodes_count, 1) 二维 ----------------------
        if new_nodes_count > 0:
            # 生成 [3,3,...] 并转为 (new_nodes_count, 1) 二维
            new_node_type = torch.tensor([3] * new_nodes_count, dtype=torch.long).view(-1, 1)
        else:
            # 空节点类型：保持 (0,1) 二维，避免拼接维度不匹配
            new_node_type = torch.empty((0, 1), dtype=torch.long)

        # ---------------------- 6. 拼接节点类型：此时二者均为二维，可正常拼接 ----------------------
        # 校验维度（提前暴露问题）
        assert node_type.dim() == 2 and node_type.shape[1] == 1, f"原始 node_type 需为 (N,1)，当前 {node_type.shape}"
        assert new_node_type.dim() == 2 and new_node_type.shape[1] == 1, f"new_node_type 需为 (M,1)，当前 {new_node_type.shape}"
        # 拼接（dim=0：按节点数方向拼接）
        node_type = torch.cat((node_type, new_node_type), dim=0)

        # ---------------------- 7. 拼接边索引：确保均为 (2, E) 二维 ----------------------
        new_edge_index = torch.cat([buff_edge_index, u_new, v_new], dim=1)
        assert new_edge_index.dim() == 2 and new_edge_index.shape[0] == 2, f"input_edges 需为 (2,E)，当前 {new_edge_index.shape}"

        # ---------------------- 8. 处理 input_verilogs：确保长度为 3584 一维 ----------------------
        max_len = 3584
        if design in self.verilog_list and self.verilog_list[design].size > 0:
            input_verilog = torch.tensor(self.verilog_list[design], dtype=torch.float).squeeze()
            # 统一为一维
            if input_verilog.dim() == 0:
                input_verilog = input_verilog.unsqueeze(0)
            # 截断/填充到 max_len
            if input_verilog.shape[0] < max_len:
                input_verilog = torch.cat([input_verilog, torch.zeros(max_len - input_verilog.shape[0], dtype=torch.float)], dim=0)
            else:
                input_verilog = input_verilog[:max_len]
        else:
            # 无 verilog 数据时用全0填充
            input_verilog = torch.zeros(max_len, dtype=torch.float)
        # 确保为一维（符合模型输入）
        input_verilog = input_verilog.view(-1)

        return dict(
            input_nodes=node_type,    # (N+M, 1) 二维
            input_edges=new_edge_index,  # (2, E_total) 二维
            input_verilogs=input_verilog  # (3584,) 一维
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