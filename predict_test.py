# predict.py (适配“A*距离 + 9维特征 + 过滤”方案的正确版本)
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import pandas as pd
import config
from model import PCBTransformer
from dataset import PCBDataset, get_all_current_info, get_global_stats
from visualize import save_heatmap_plot
from itertools import product


def main():
    print(f"--- 运行预测与可视化 ({config.BEST_MODEL_PATH} 方案) ---")
    device = torch.device(config.DEVICE)
    all_board_ids = list(range(1, config.NUM_BOARDS + 1))
    np.random.seed(42);
    np.random.shuffle(all_board_ids)
    train_size = int(len(all_board_ids) * config.TRAIN_SPLIT);
    val_size = int(len(all_board_ids) * config.VAL_SPLIT)
    train_ids = all_board_ids[:train_size];
    test_ids = all_board_ids[train_size + val_size:]
    if not test_ids: raise ValueError("没有可用的测试电路板。")
    all_current_info = get_all_current_info()

    print("加载全局统计数据 (已过滤)...")
    global_stats = get_global_stats(train_ids, all_current_info)

    # --------------------------------------------------
    # --- 关键：在这里指定您想测试的电路板ID ---
    test_board_id = 19742  # <-- 请修改为您想预测的ID (例如 57)
    # --------------------------------------------------

    print(f"选定的测试电路板ID: {test_board_id}")

    try:
        test_dataset = PCBDataset([test_board_id], all_current_info, global_stats)
    except Exception as e:
        print(f"为电路板 {test_board_id} 创建数据集时失败: {e}")
        return

    # --- 关键修改：这里的 test_dataset[0] 返回 (features, labels_norm) ---
    features, labels_norm = test_dataset[0]  # [N_i, 9], [N_i]
    # -----------------------------------------------------------------

    if features.shape[0] == 0:
        print(f"错误：电路板 {test_board_id} 没有任何有效数据点 (mask=1 且 voltage >= -3.0V)。")
        return

    features_batch = features.unsqueeze(0).to(device)
    validity_mask = torch.ones(1, features.shape[0], dtype=torch.bool).to(device)

    model = PCBTransformer().to(device)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model.eval()
    print("模型已从", config.BEST_MODEL_PATH, "加载")

    with torch.no_grad():
        pred_voltages_norm = model(features_batch, validity_mask).squeeze(0).squeeze(-1).cpu().numpy()  # [N_i]

    # --- 关键修改：labels_norm 已经是我们需要的张量 ---
    true_voltages_norm = labels_norm.numpy()  # [N_i]
    # ----------------------------------------------

    xy_coords = features[:, 0:2].numpy()  # [N_i, 2]

    v_min = global_stats['v_min']
    v_range = global_stats['v_max'] - v_min
    pred_voltages_real = pred_voltages_norm * v_range + v_min
    true_voltages_real = true_voltages_norm * v_range + v_min

    # --- 1. 计算 "原始" 统计数据 ---
    abs_errors = np.abs(pred_voltages_real - true_voltages_real)
    mae_raw = np.mean(abs_errors)
    max_error_raw = np.max(abs_errors)
    min_error_raw = np.min(abs_errors)

    print(f"\n--- 电路板 {test_board_id} 的【原始】评估结果 (过滤后数据) ---")
    print(f"  节点数: {features.shape[0]}")
    print(f"  平均绝对误差 (MAE): {mae_raw:.6f} V")
    print(f"  最大绝对误差 (Max Error): {max_error_raw:.6f} V")
    print(f"  最小绝对误差 (Min Error): {min_error_raw:.6f} V")
    print(f"  预测电压范围: {np.min(pred_voltages_real):.4f} V  到  {np.max(pred_voltages_real):.4f} V")
    print(f"  真实电压范围: {np.min(true_voltages_real):.4f} V  到  {np.max(true_voltages_real):.4f} V")

    # --- 2. 打印前50个点的误差 ---
    print("\n--- 预测 vs. 真实 对比 (前50个点，按电学距离排序) ---")
    comparison_df = pd.DataFrame({
        'x': xy_coords[:50, 0],
        'y': xy_coords[:50, 1],
        '真实电压': true_voltages_real[:50],
        '预测电压': pred_voltages_real[:50],
        '绝对误差': abs_errors[:50]
    })
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.float_format', '{:.6f}'.format)
    print(comparison_df.to_string())

    print("\n正在重建 2D 热力图...")

    # --- 3. 重建 2D 网格 ---
    x_coords_np = config.X_COORDS.numpy()
    y_coords_np = config.Y_COORDS.numpy()
    full_x_coords = np.tile(x_coords_np, len(y_coords_np))
    full_y_coords = np.repeat(y_coords_np, len(x_coords_np))
    full_df = pd.DataFrame({'x': full_x_coords, 'y': full_y_coords})

    result_df = pd.DataFrame({
        'x': xy_coords[:, 0],
        'y': xy_coords[:, 1],
        'pred_v': pred_voltages_real,
        'true_v': true_voltages_real
    })

    merged_df = pd.merge(full_df, result_df, on=['x', 'y'], how='left')
    merged_df = merged_df.sort_values(by=['y', 'x']).reset_index(drop=True)

    voltage_map = merged_df['pred_v'].fillna(np.nan).values.reshape(config.GRID_SIZE)
    true_voltage_map = merged_df['true_v'].fillna(np.nan).values.reshape(config.GRID_SIZE)
    mask_map = ~merged_df['pred_v'].isna().values.reshape(config.GRID_SIZE)

    # --- 4. 后处理：强制修正源端 ---
    origin_y_idx = 10;
    origin_x_idx = 10

    if config.GRID_SIZE[0] > origin_y_idx and config.GRID_SIZE[1] > origin_x_idx:
        if mask_map[origin_y_idx, origin_x_idx]:
            voltage_map[origin_y_idx, origin_x_idx] = 0.0
            print(f"后处理：已将坐标 (0.0, 0.0) [索引 {origin_y_idx}, {origin_x_idx}] 的预测值强制设为 0.0V。")
        else:
            print(f"后处理：坐标 (0.0, 0.0) [索引 {origin_y_idx}, {origin_x_idx}] 不在有效数据点中，未修改。")
    else:
        print(f"警告：(0,0) 点的索引 [{origin_y_idx}, {origin_x_idx}] 超出了网格大小 {config.GRID_SIZE}。")

    # --- 5. 重新计算 "后处理" 统计数据 ---
    corrected_preds_flat = voltage_map[mask_map]
    true_vals_flat = true_voltage_map[mask_map]

    abs_errors_post = np.abs(corrected_preds_flat - true_vals_flat)
    mae_post_processed = np.mean(abs_errors_post)
    max_error_post = np.max(abs_errors_post)
    min_error_post = np.min(abs_errors_post)
    range_post_pred_min = np.min(corrected_preds_flat)
    range_post_pred_max = np.max(corrected_preds_flat)

    print(f"\n--- 电路板 {test_board_id} 的【后处理】评估结果 ---")
    print(f"  平均绝对误差 (MAE): {mae_post_processed:.6f} V")
    print(f"  最大绝对误差 (Max Error): {max_error_post:.6f} V")
    print(f"  最小绝对误差 (Min Error): {min_error_post:.6f} V")
    print(f"  预测电压范围: {range_post_pred_min:.4f} V  到  {range_post_pred_max:.4f} V")
    print(f"  (真实电压范围: {np.min(true_vals_flat):.4f} V  到  {np.max(true_vals_flat):.4f} V)")

    current_info = all_current_info.get(test_board_id)
    save_heatmap_plot(test_board_id, voltage_map, mask_map, current_info, config.HEATMAP_PLOT_PATH)


if __name__ == "__main__":
    main()