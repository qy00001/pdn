# config.py (已更新为“可变长序列 + 9维特征”方案)
import torch

# --- 数据路径配置 ---
DATA_ROOT_DIR = "../holes_grid_data_2mm_new7"
RESISTANCE_DIR = DATA_ROOT_DIR
VOLTAGE_DIR = DATA_ROOT_DIR
CURRENT_INFO_FILE = f"{DATA_ROOT_DIR}/board_point_current_info.txt"

# --- 输出路径配置 ---
OUTPUT_DIR = "outputs_via"
BEST_MODEL_PATH = f"{OUTPUT_DIR}/best_regression_model_dijkstra_9feat_via.pth" # 新的模型名
LOSS_PLOT_PATH = f"{OUTPUT_DIR}/regression_training_history_dijkstra_9feat_via.png"
HEATMAP_PLOT_PATH = f"{OUTPUT_DIR}/regression_heatmap_dijkstra_9feat_via"

# --- 模型超参数 ---
# (x, y, is_vrm, is_sink, norm_current) = 5
# (norm_R_up, norm_R_down, norm_R_left, norm_R_right) = 4
# TOTAL = 5 + 4 = 9
INPUT_DIM = 9           # <--- 关键修改：特征维度变为 9
D_MODEL = 256
N_HEAD = 8
NUM_ENCODER_LAYERS = 6
DROPOUT = 0.1

# --- 训练超参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LEARNING_RATE = 0.00001
EPOCHS = 80
NUM_BOARDS = 20000
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

# --- 网格信息 ---
X_COORDS = torch.arange(-20, 22, 2)
Y_COORDS = torch.arange(-20, 22, 2)
GRID_SIZE = (len(Y_COORDS), len(X_COORDS))
SEQ_LENGTH = GRID_SIZE[0] * GRID_SIZE[1]