# visualize.py
import matplotlib.pyplot as plt
import numpy as np
import config


# ... save_training_history_plot 函数保持不变 ...
def save_training_history_plot(history, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'bo-', label='训练损失 (L1 Loss)')
    ax1.plot(epochs, history['val_loss'], 'ro-', label='验证损失 (L1 Loss)')
    ax1.set_title('训练和验证损失');
    ax1.set_xlabel('Epochs');
    ax1.set_ylabel('L1 Loss')
    ax1.legend();
    ax1.grid(True)
    ax2.plot(epochs, history['val_mae'], 'go-', label='验证平均绝对误差 (MAE)')
    ax2.set_title('验证平均绝对误差');
    ax2.set_xlabel('Epochs');
    ax2.set_ylabel('MAE (V)')
    ax2.legend();
    ax2.grid(True)
    plt.tight_layout();
    plt.savefig(out_path);
    plt.close()
    print(f"训练历史图已保存至 {out_path}")


def save_heatmap_plot(board_id, voltage_map, mask_map, current_info, outdir):
    fig, ax = plt.subplots(figsize=(12, 10))
    x_grid, y_grid = config.X_COORDS.numpy(), config.Y_COORDS.numpy()
    all_pts = np.array([[x, y] for y in y_grid for x in x_grid])
    voltage_flat, mask_flat = voltage_map.flatten(), mask_map.flatten()
    valid_pts = all_pts[mask_flat == 1]
    valid_voltage = voltage_flat[mask_flat == 1]
    sc = ax.scatter(valid_pts[:, 0], valid_pts[:, 1], s=40, c=valid_voltage, cmap='coolwarm', alpha=1.0,
                    label='Voltage')
    xx, yy = np.meshgrid(x_grid, y_grid)
    ax.contour(xx, yy, mask_map, levels=[0.5], colors='blue', linewidths=2)
    invalid_pts = all_pts[mask_flat == 0]
    if len(invalid_pts) > 0:
        ax.scatter(invalid_pts[:, 0], invalid_pts[:, 1], s=10, c='black', alpha=0.1, label='Invalid Nodes')

    # --- 关键修改 ---
    # 直接在 (0,0) 绘制 VRM (源端)
    ax.scatter([0.0], [0.0], s=250, c='lime', marker='*',
               edgecolors='k', linewidths=1.5, label='VRM (Source)', zorder=5)

    # Sinks 的绘制逻辑保持不变
    if current_info and current_info['sinks']:
        sinks = np.array([s['coord'] for s in current_info['sinks']])
        if len(sinks) > 0:
            ax.scatter(sinks[:, 0], sinks[:, 1], s=180, c='deepskyblue', marker='X',
                       edgecolors='k', linewidths=1.5, label='Sinks (Loads)', zorder=5)

    ax.set_xlabel('X (mm)');
    ax.set_ylabel('Y (mm)')
    ax.set_title(f'Predicted Voltage Distribution for Board {board_id}')
    ax.axis('equal');
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.colorbar(sc, ax=ax, label='Predicted Voltage (V)')
    ax.legend();
    plt.tight_layout()
    # save_path = f"{outdir}_{board_id}.png"
    save_path = f"{outdir}_{board_id}_test.png"
    plt.savefig(save_path)
    plt.close()
    print(f"热力图已保存至 {save_path}")