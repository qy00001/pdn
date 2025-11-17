# model.py (已更新为“可变长序列”方案)
import torch
import torch.nn as nn
import math
import config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=config.SEQ_LENGTH):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]  # <--- 关键：PositionalEncoding 现在编码的是电学距离排名
        return self.dropout(x)


class PCBTransformer(nn.Module):
    def __init__(self, input_dim=config.INPUT_DIM, d_model=config.D_MODEL, nhead=config.N_HEAD,
                 num_encoder_layers=config.NUM_ENCODER_LAYERS, dropout=config.DROPOUT):
        super(PCBTransformer, self).__init__()
        self.d_model = d_model

        # 输入层 (5维 -> d_model)
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=d_model * 4,
                                                   dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_regressor = nn.Linear(d_model, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_regressor.bias.data.zero_()
        self.output_regressor.weight.data.uniform_(-initrange, initrange)

    # --- 核心修改：forward 函数现在接受掩码 ---
    def forward(self, src, validity_mask):
        # src shape: (B, Seq, Dim) e.g. (16, 950, 5)
        # validity_mask shape: (B, Seq) e.g. (16, 950), True for valid

        src = self.input_embedding(src)

        src = src.permute(1, 0, 2)  # (Seq, B, Dim)

        src = self.pos_encoder(src)

        # --- 关键：转换掩码 ---
        # Transformer 需要的掩码是 (B, Seq), 且 True 代表 "被遮盖" (无效)
        padding_mask = ~validity_mask

        output = self.transformer_encoder(src, src_key_padding_mask=padding_mask)

        output = output.permute(1, 0, 2)

        output = self.output_regressor(output)

        return output