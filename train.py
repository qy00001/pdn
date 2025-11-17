# train.py (已更新为“可变长序列”方案)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from tqdm import tqdm

import config
from model import PCBTransformer
from dataset import PCBDataset, get_all_current_info, get_global_stats, pad_collate_fn  # 导入 collate_fn
from visualize import save_training_history_plot


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    # --- 核心修改：解包 (features, labels, mask) ---
    for features, labels, mask in tqdm(dataloader, desc="训练中"):
        features, labels, mask = features.to(device), labels.to(device), mask.to(device)

        optimizer.zero_grad()

        # --- 核心修改：传入 mask ---
        pred_voltages = model(features, mask).squeeze(-1)  # (B, Seq)

        # --- 核心修改：只在有效点上计算损失 ---
        if mask.sum() == 0: continue  # 跳过全空批次
        loss = loss_fn(pred_voltages[mask], labels[mask])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, loss_fn, device, v_min, v_range):
    model.eval()
    total_loss, total_mae = 0, 0
    total_valid_points = 0

    with torch.no_grad():
        for features, labels, mask in tqdm(dataloader, desc="验证中"):
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)

            if mask.sum() == 0: continue

            pred_voltages_norm = model(features, mask).squeeze(-1)

            loss = loss_fn(pred_voltages_norm[mask], labels[mask])
            total_loss += loss.item()

            pred_voltages_real = pred_voltages_norm * v_range + v_min
            true_voltages_real = labels * v_range + v_min

            mae = torch.abs(pred_voltages_real[mask] - true_voltages_real[mask]).sum()  # <--- 注意是 sum()
            total_mae += mae.item()
            total_valid_points += mask.sum().item()

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    avg_mae = total_mae / total_valid_points if total_valid_points > 0 else 0.0
    return avg_loss, avg_mae


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    all_board_ids = list(range(1, config.NUM_BOARDS + 1))
    np.random.seed(42);
    np.random.shuffle(all_board_ids)
    train_size, val_size = int(len(all_board_ids) * config.TRAIN_SPLIT), int(len(all_board_ids) * config.VAL_SPLIT)
    train_ids, val_ids = all_board_ids[:train_size], all_board_ids[train_size:train_size + val_size]
    all_current_info = get_all_current_info()
    global_stats = get_global_stats(train_ids, all_current_info)
    train_dataset, val_dataset = PCBDataset(train_ids, all_current_info, global_stats), PCBDataset(val_ids,
                                                                                                   all_current_info,
                                                                                                   global_stats)

    # --- 核心修改：添加 collate_fn ---
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)

    device = torch.device(config.DEVICE)
    model = PCBTransformer().to(device)
    print(f"模型已加载到 {device}")

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)

    best_val_mae = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_mae = validate(model, val_loader, loss_fn, device,
                                     global_stats['v_min'], global_stats['v_max'] - global_stats['v_min'])

        scheduler.step(val_mae)
        print(f"Epoch {epoch + 1} 总结:")
        print(f"  训练损失 (L1): {train_loss:.6f}")
        print(f"  验证损失 (L1): {val_loss:.6f}")
        print(f"  验证平均绝对误差 (MAE): {val_mae:.6f} V")

        history['train_loss'].append(train_loss);
        history['val_loss'].append(val_loss);
        history['val_mae'].append(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"新最佳模型已保存至 {config.BEST_MODEL_PATH}，MAE: {best_val_mae:.6f} V")

    save_training_history_plot(history, config.LOSS_PLOT_PATH)
    print("训练历史图已生成。")


if __name__ == "__main__":
    main()