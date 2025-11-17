# dataset.py (已更新：增加-3.0V过滤)
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm
import config
import networkx as nx
from torch.nn.utils.rnn import pad_sequence

# 定义电压阈值
VOLTAGE_THRESHOLD = -3.0


# ... parse_current_info 和 get_all_current_info 函数保持不变 ...
def parse_current_info(line):
    try:
        board_id_match = re.search(r'NUM_(\d+):', line)
        if not board_id_match: return None
        board_id = int(board_id_match.group(1))
        vrm_match = re.search(r'vrm:\[(.*?), (.*?)], (\d+)A', line)
        if not vrm_match: return None
        vrm_x, vrm_y, vrm_current = vrm_match.groups()
        vrm_info = {'coord': (float(vrm_x), float(vrm_y)), 'current': int(vrm_current)}
        sinks_info = []
        sink_block_match = re.search(r'sink,(.*)', line)
        if sink_block_match:
            sink_block = sink_block_match.group(1)
            sinks_matches = re.findall(r'\[(.*?), (.*?)], (\d+)A', sink_block)
            for sink_x, sink_y, sink_current in sinks_matches:
                sinks_info.append({'coord': (float(sink_x), float(sink_y)), 'current': -int(sink_current)})
        return board_id, {'vrm': vrm_info, 'sinks': sinks_info}
    except Exception as e:
        print(f"解析行时发生错误: {line.strip()} -> {e}")
        return None


def get_all_current_info():
    all_info = {}
    if not os.path.exists(config.CURRENT_INFO_FILE):
        print(f"警告: 在路径 {config.CURRENT_INFO_FILE} 未找到电流信息文件。")
        return {}
    with open(config.CURRENT_INFO_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parsed = parse_current_info(line)
            if parsed:
                board_id, info = parsed
                all_info[board_id] = info
    return all_info


# --- 关键修改：get_global_stats 现在会过滤数据 ---
def get_global_stats(board_ids, all_current_info):
    # ... 路径诊断 (保持) ...
    print("\n--- 路径诊断 ---")
    cwd = os.getcwd();
    print(f"脚本当前工作目录: {cwd}")
    data_root_abs = os.path.abspath(config.DATA_ROOT_DIR);
    print(f"程序尝试访问的数据根目录: {data_root_abs}")
    if not os.path.exists(data_root_abs):
        raise FileNotFoundError(f"\n\n致命错误: 指定的数据目录不存在: {data_root_abs}")
    else:
        print("成功: 数据目录已找到。"); dir_contents = os.listdir(data_root_abs); print(
            f"在该目录中找到的前5个文件/文件夹: {dir_contents[:5]}")
    print("------------------------\n")

    print(f"正在计算全局统计数据... 将过滤掉 voltage < {VOLTAGE_THRESHOLD}V 的点")
    all_valid_voltages = []
    all_valid_resistances = []

    for board_id in tqdm(board_ids):
        voltage_file = os.path.join(config.VOLTAGE_DIR, f"voltage_with_mask_{board_id}.csv")
        possible_resistance_files = [os.path.join(config.RESISTANCE_DIR, f"resistance_{board_id}.csv"),
                                     os.path.join(config.RESISTANCE_DIR,
                                                  f"resistance_matrix_{board_id}_with_coords.csv")]
        resistance_file = next((f for f in possible_resistance_files if os.path.exists(f)), None)
        if not resistance_file or not os.path.exists(voltage_file): continue

        try:
            voltage_df = pd.read_csv(voltage_file)
            resistance_df = pd.read_csv(resistance_file)

            merged_df = pd.merge(resistance_df, voltage_df, on=['x', 'y'], how='left')

            # --- 核心过滤逻辑 ---
            valid_data_df = merged_df[
                (merged_df['mask'] == 1) &
                (merged_df['voltage'] >= VOLTAGE_THRESHOLD)
                ]
            # --------------------

            if valid_data_df.empty:
                continue

            all_valid_voltages.extend(valid_data_df['voltage'].values)

            res_cols = ['R_up', 'R_down', 'R_left', 'R_right']
            for col in res_cols:
                all_valid_resistances.extend(valid_data_df[col].values)

        except Exception as e:
            print(f"\n警告: 无法处理电路板 {board_id}。错误: {e}")

    if not all_valid_voltages: raise FileNotFoundError(
        "\n\n致命错误: 数据目录已找到, 但没有文件匹配预期的格式(或所有点都被过滤)。")

    all_valid_resistances = np.array(all_valid_resistances)
    all_valid_resistances = all_valid_resistances[all_valid_resistances < 50000.0]  # 阈值
    if all_valid_resistances.size == 0: raise ValueError(
        "已找到数据文件, 但在过滤后的有效铜皮区域中未找到任何有效的电阻值。")

    stats = {
        'r_mean': np.mean(all_valid_resistances),
        'r_std': np.std(all_valid_resistances),
        'v_min': np.min(all_valid_voltages),  # 这很可能就是 -3.0
        'v_max': np.max(all_valid_voltages)  # 这很可能是 0.0
    }
    print(f"全局电阻均值 (已过滤): {stats['r_mean']:.4f}, 标准差: {stats['r_std']:.4f}")
    print(f"全局电压最小值 (已过滤): {stats['v_min']:.4f}, 最大值: {stats['v_max']:.4f}")
    return stats


class PCBDataset(Dataset):
    def __init__(self, board_ids, all_current_info, global_stats):
        self.board_ids = board_ids
        self.all_current_info = all_current_info
        self.global_stats = global_stats
        self.v_min, self.v_max = global_stats['v_min'], global_stats['v_max']
        self.v_range = self.v_max - self.v_min if (self.v_max - self.v_min) > 0 else 1
        self.r_mean, self.r_std = global_stats['r_mean'], global_stats['r_std']
        self.RESISTANCE_THRESHOLD = 50000.0
        self.res_cols = ['R_up', 'R_down', 'R_left', 'R_right']

    def __len__(self):
        return len(self.board_ids)

    def __getitem__(self, idx):
        board_id = self.board_ids[idx]
        voltage_file = os.path.join(config.VOLTAGE_DIR, f"voltage_with_mask_{board_id}.csv")
        possible_resistance_files = [os.path.join(config.RESISTANCE_DIR, f"resistance_{board_id}.csv"),
                                     os.path.join(config.RESISTANCE_DIR,
                                                  f"resistance_matrix_{board_id}_with_coords.csv")]
        resistance_file = next((f for f in possible_resistance_files if os.path.exists(f)), None)
        if not resistance_file: raise FileNotFoundError(f"找不到电路板 {board_id} 的电阻文件")

        voltage_df = pd.read_csv(voltage_file)
        resistance_df = pd.read_csv(resistance_file)
        data_df = pd.merge(resistance_df, voltage_df, on=['x', 'y'], how='left')

        # --- 核心修改 1: 过滤 ---
        data_df = data_df[
            (data_df['mask'] == 1) &
            (data_df['voltage'] >= VOLTAGE_THRESHOLD)
            ].reset_index(drop=True)
        # ------------------------

        if data_df.empty:
            return torch.empty(0, config.INPUT_DIM), torch.empty(0)

        # 2. 建图 (只在过滤后的点上)
        G = nx.Graph()
        coord_to_node = {(r['x'], r['y']): i for i, r in data_df.iterrows()}
        for i, row in data_df.iterrows():
            neighbor_coord = (row['x'], row['y'] + 2);
            neighbor_node = coord_to_node.get(neighbor_coord)
            if neighbor_node is not None and row['R_up'] < self.RESISTANCE_THRESHOLD: G.add_edge(i, neighbor_node,
                                                                                                 weight=row['R_up'])
            neighbor_coord = (row['x'] + 2, row['y']);
            neighbor_node = coord_to_node.get(neighbor_coord)
            if neighbor_node is not None and row['R_right'] < self.RESISTANCE_THRESHOLD: G.add_edge(i, neighbor_node,
                                                                                                    weight=row[
                                                                                                        'R_right'])

        # 3. Dijkstra
        source_node = coord_to_node.get((0.0, 0.0))
        # 检查源端是否在过滤后的数据中
        if source_node is None or source_node not in G:
            # 如果源端(0,0)被过滤掉了(比如它的电压<-3V，或mask=0)，我们就没法排序了
            # 这是一个潜在问题，但我们暂时假设(0,0)总是在
            if data_df.empty: return torch.empty(0, config.INPUT_DIM), torch.empty(0)
            source_node = data_df.iloc[0].name  # 使用第一个点作为备用源

        try:
            lengths = nx.shortest_path_length(G, source=source_node, weight='weight')
            data_df['elec_dist'] = data_df.index.map(lengths).fillna(np.inf)
        except nx.NodeNotFound:
            data_df['elec_dist'] = np.inf

        # 4. 排序
        data_df_sorted = data_df.sort_values(by='elec_dist').reset_index(drop=True)

        # 5. 创建 9 维特征 (逻辑不变)
        N_i = len(data_df_sorted)
        features = np.zeros((N_i, config.INPUT_DIM), dtype=np.float32)
        features[:, 0] = data_df_sorted['x'].values
        features[:, 1] = data_df_sorted['y'].values

        current_col = np.zeros(N_i, dtype=np.float32)
        current_info = self.all_current_info.get(board_id)
        if current_info:
            vrm_idx = np.where((data_df_sorted['x'] == 0.0) & (data_df_sorted['y'] == 0.0))[0]
            if vrm_idx.size > 0:
                features[vrm_idx[0], 2] = 1.0
                current_col[vrm_idx[0]] = current_info['vrm']['current']
            for sink in current_info['sinks']:
                sink_idx = \
                np.where((data_df_sorted['x'] == sink['coord'][0]) & (data_df_sorted['y'] == sink['coord'][1]))[0]
                if sink_idx.size > 0:
                    features[sink_idx[0], 3] = 1.0
                    current_col[sink_idx[0]] = sink['current']

        non_zero_mask = (current_col != 0)
        if np.any(non_zero_mask):
            mean_current, std_current = np.mean(current_col[non_zero_mask]), np.std(current_col[non_zero_mask])
            if std_current > 0: current_col[non_zero_mask] = (current_col[non_zero_mask] - mean_current) / std_current
        features[:, 4] = current_col

        # 归一化电阻
        for i, col in enumerate(self.res_cols):
            res_values = data_df_sorted[col].values
            norm_res = np.zeros_like(res_values, dtype=np.float32)
            if self.r_std > 0:
                # 关键：我们只归一化真实电阻，>50000的仍然设为0或一个特殊值
                real_res_mask = (res_values < self.RESISTANCE_THRESHOLD)
                norm_res[real_res_mask] = (res_values[real_res_mask] - self.r_mean) / self.r_std

                # 对于那些 > 50000 的值 (非连通)，我们不应使用归一化
                # 我们可以设为0，或者设为一个固定的高值
                # 让我们设为一个固定的高Z-score值，比如 5.0
                norm_res[~real_res_mask] = 5.0
            features[:, 5 + i] = norm_res

        # --- 创建标签 ---
        voltages = data_df_sorted['voltage'].values
        normalized_voltages = (voltages - self.v_min) / self.v_range

        return torch.from_numpy(features), torch.from_numpy(normalized_voltages).float()


# collate_fn 保持不变
def pad_collate_fn(batch):
    batch = [b for b in batch if b[0].shape[0] > 0]
    if not batch:
        return torch.empty(0, 0, config.INPUT_DIM), torch.empty(0, 0), torch.empty(0, 0, dtype=torch.bool)
    features_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=0.0)
    lengths = [len(f) for f in features_list]
    max_len = features_padded.shape[1]
    validity_mask = torch.arange(max_len)[None, :] < torch.tensor(lengths)[:, None]
    return features_padded, labels_padded, validity_mask