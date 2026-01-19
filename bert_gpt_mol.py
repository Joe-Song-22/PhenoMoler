import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F


def create_prefix_look_ahead_mask(seq_len, prefix_len, device):
    # 1. 创建原有的上三角掩码 (右上角全为 1，代表遮掩)
    # [[0, 1, 1],
    #  [0, 0, 1],
    #  [0, 0, 0]]
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)

    # 2. 将前 prefix_len 列全部强制置为 0 (代表可见)
    # 无论当前在哪个位置，前 prefix_len 个 token 永远不会被遮掩
    mask[:, :prefix_len] = 0

    return mask
def create_look_ahead_mask(seq_len, device):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask  # (seq_len, seq_len)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x = x + self.pe[:, :seq_len, :]
        return x

class FourierPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(FourierPositionalEncoding, self).__init__()
        self.d_model = d_model

        # 构建 (d_model, d_model) 矩阵
        W = torch.zeros(d_model, d_model)

        # 将 [0, d_model-1] 映射到 [-1, 1]
        t = torch.linspace(-1, 1, steps=d_model)

        # 不同频率（傅里叶基底：1,2,3,...,d_model）
        freqs = torch.arange(1, d_model + 1, dtype=torch.float)

        for i, f in enumerate(freqs):
            if i % 2 == 0:
                # 偶数列 -> cos
                W[:, i] = torch.cos(np.pi * f * t)
            else:
                # 奇数列 -> sin
                W[:, i] = torch.sin(np.pi * f * t)

        # 注册 buffer (固定不训练)
        self.register_buffer('W', W)

    def forward(self, x):
        # x: (batch, seq, d_model)
        batch, seq, d_model = x.shape
        assert d_model == self.d_model, "d_model mismatch"

        # 展平成 (batch*seq, d_model)
        x_flat = x.reshape(-1, d_model)

        # 投影到傅里叶基底
        x_proj = torch.matmul(x_flat, self.W)

        # reshape 回去
        return x_proj.view(batch, seq, d_model)

# class Graphormer(nn.Module):
#     def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, num_layers, heads):
#         super(Graphormer, self).__init__()
#         self.node_feature_transform = nn.Linear(node_feature_dim, hidden_dim)
#         self.edge_feature_transform = nn.Linear(edge_feature_dim, hidden_dim)
#         self.layers = nn.ModuleList([
#             GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
#         ])
#         self.dropout = nn.Dropout(0.1)
#         self.hidden_dim = hidden_dim
#
#     def forward(self, x, edge_index, batch):
#         """
#         x: 节点特征 (num_nodes, node_feature_dim)
#         edge_index: 边索引 (2, num_edges)
#         batch: 节点所属子图的批量索引 (num_nodes,)
#         """
#         # 升维节点特征
#         x = self.node_feature_transform(x)  # (num_nodes, node_feature_dim) -> (num_nodes, hidden_dim)
#
#         # 按照 GCN 层处理所有子图
#         for conv in self.layers:
#             x = F.relu(conv(x, edge_index))  # 图卷积
#             x = self.dropout(x)
#
#         # 将节点特征重新组织为 (batch_size, max_nodes, hidden_dim)
#         x, mask = to_dense_batch(x, batch)  # 转为稠密表示，方便后续与 Transformer 结合
#
#         return x # 返回稠密节点嵌入和子图掩码

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.depth)
        if mask is not None:
            # 扩展掩码以适配多头注意力
            mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len_q, seq_len_k)
            mask = mask.to(scores.device)
            scores = scores.masked_fill(mask == 1, -1e9)
        attention = torch.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        x = torch.matmul(attention, v)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        return self.fc(x), attention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, attention_encoder = self.mha(x, x, x, mask)
        out1 = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output))
        return out2


# class DecoderEmbedding(nn.Module):
#     def __init__(self, d_model, max_len):
#         super(DecoderEmbedding, self).__init__()
#         # 定义一个线性层，将每个维度映射到 d_model（512）维
#         self.embedding = nn.Linear(1, d_model)
#         self.pos_encoding = FourierPositionalEncoding(d_model)
#     def forward(self, tgt, d_model):
#         # tgt shape: (batch_size, 79)
#         # 1. 将输入扩展维度，以便每个维度都能通过线性层进行处理
#         tgt = tgt.unsqueeze(-1)  # (batch_size, 79) -> (batch_size, 79, 1)
#
#         # 2. 使用线性层将每个维度映射到 512 维
#         tgt = tgt.float()
#         tgt_emb = self.embedding(tgt)*np.sqrt(d_model)  # (batch_size, 79, 1) -> (batch_size, 79, 512)
#         tgt_emb = self.pos_encoding(tgt_emb)
#         return tgt_emb
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, max_len, shared_embedding, dropout=0.0):
        super().__init__()
        self.shared_embedding = shared_embedding
        # self.pos_encoding = FourierPositionalEncoding(d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.enc_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (batch, 97)
        x = self.shared_embedding(x)             # (batch, 97, d_model)
        x = self.pos_encoding(x)           # 添加位置编码
        x = self.dropout(x)
        batch_size, seq_len, _ = x.size()
        zero_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=x.device)
        for layer in self.enc_layers:
            x = layer(x, zero_mask)             # (batch, 97, d_model)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)  # 自注意力
        self.mha2 = MultiHeadAttention(d_model, num_heads)  # 编码-解码注意力

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output=None, tgt_mask=None, use_cross_attention=True):
        """
        Parameters:
            x: 解码器输入 (batch_size, seq_len, d_model)
            enc_output: 编码器输出 (batch_size, enc_seq_len, d_model)，如果不需要编码器-解码器注意力，可传入 None
            tgt_mask: 目标序列的掩码
            use_cross_attention: 是否使用编码器-解码器注意力，默认为 True
        """
        if enc_output is None:
            # 只使用自注意力
            attn1, attn_weights1 = self.mha1(x, x, x, tgt_mask)
            out = self.layernorm1(x + self.dropout1(attn1))
            attn_weights2 = None  # 不使用交叉注意力
        else:
            # 先自注意力
            attn1, attn_weights1 = self.mha1(x, x, x, tgt_mask)
            out1 = self.layernorm1(x + self.dropout1(attn1))

            # 再交叉注意力
            attn2, attn_weights2 = self.mha2(out1, enc_output, enc_output)
            out = self.layernorm2(out1 + 0.3*self.dropout2(attn2))

        # 前馈神经网络
        ffn_output = self.ffn(out)
        out = self.layernorm3(out + self.dropout3(ffn_output))

        return out, attn_weights1, attn_weights2
class ScaffoldDecoder(nn.Module):
    def __init__(self, scaffold_layers, d_model, num_heads, d_ff, max_len, shared_embedding, dropout=0.1):
        super().__init__()
        self.embedding = shared_embedding
        self.pos_encoding = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos_encoding.weight, mean=0.0, std=0.02)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(scaffold_layers)
        ])

    def forward(self, x, enc_output=None, tgt_mask=None):
        # x: [batch, seq_len]
        x = self.embedding(x) + self.pos_encoding(torch.arange(x.size(1), device=x.device))
        attn_weights = []
        for layer in self.layers:
            x, attn1, _ = layer(x, enc_output, tgt_mask=tgt_mask)
            attn_weights.append(attn1)
        return x, attn_weights
# class TransformerDecoder(nn.Module):
#     def __init__(self, full_layers, d_model, num_heads, d_ff, max_len, shared_embedding, dropout=0.1):
#         super(TransformerDecoder, self).__init__()
#         self.embedding = shared_embedding
#         # self.pos_encoding = FourierPositionalEncoding(d_model)
#         self.pos_encoding = PositionalEncoding(d_model, max_len)
#         self.layers = nn.ModuleList([
#             TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
#             for _ in range(full_layers)
#         ])
#
#     def forward(self, x, enc_output=None, tgt_mask=None):
#         attn_weights_all = []
#         x = self.embedding(x)
#         x = self.pos_encoding(x)
#         for layer in self.layers:
#             x, attn1, attn2 = layer(x, enc_output, tgt_mask)
#             attn_weights_all.append((attn1, attn2))
#         return x, attn_weights_all

class TransformerDecoder(nn.Module):
    def __init__(self, full_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(full_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, hidden, enc_output=None, tgt_mask=None):
        """
        hidden: (batch, seq_len, d_model)
        enc_output: (batch, src_len, d_model)
        """
        attn_weights_all = []

        x = hidden
        for layer in self.layers:
            x, attn1, attn2 = layer(
                x,
                enc_output,
                tgt_mask
            )
            attn_weights_all.append((attn1, attn2))

        x = self.norm(x)
        return x, attn_weights_all

class TransLSTMEncoderDecoder(nn.Module):
    def __init__(self, scaffold_layers, full_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_len, dropout):
        super(TransLSTMEncoderDecoder, self).__init__()
        # self.graph_encoder = Graphormer(node_feature_dim=1, edge_feature_dim=0, hidden_dim=d_model, num_layers=3, heads=4)
        # Encoder 部分
        # self.encoder_embedding = nn.Linear(input_vocab_size, d_model)
        # encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
        self.shared_embedding = nn.Embedding(target_vocab_size, d_model)
        # self.encoder = TransformerEncoder(full_layers, d_model, num_heads, d_ff,  max_len, self.shared_embedding, dropout=dropout)

        # Decoder 部分
        self.scaffold = ScaffoldDecoder(scaffold_layers, d_model, num_heads, d_ff, max_len, self.shared_embedding, dropout=dropout)
        self.decoder = TransformerDecoder(full_layers, d_model, num_heads, d_ff, dropout=dropout)
        # self.lstm = nn.LSTM(d_model, d_model, num_layers, batch_first=True)
        # self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        # self.crf = CRF(num_tags=target_vocab_size, batch_first=True)
        # 输出层
        self.fc_out = nn.Linear(d_model, target_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, src, tgt, d_model, hidden=None, mask=None):
        # src 是基因扰动数据，形状 (batch_size, input_vocab_size)
        # tgt 是药物字符串的输入，形状 (batch_size, seq_len)

        # Encoder 部分
        # enc_src = self.encoder(src)  # Transformer Encoder 输出 (batch_size, seq_len, d_model)

        # 如果未传入 mask，则使用默认的 look-ahead mask
        if mask is None:
            mask = create_look_ahead_mask(tgt.shape[1])
            mask = mask.unsqueeze(0).expand(tgt.shape[0], -1, -1)
        scaffold_tgt, attn_scaffold = self.scaffold(tgt, src, mask)
        scaffold_tgt = self.norm(scaffold_tgt)
        output, attn_weights1 = self.decoder(scaffold_tgt, src, mask)  # 使用 encoder 的输出作为条件信息
        # output = output + lstm_out
        # 生成输出，预测下一个字符
        output = self.norm(output)
        output = self.fc_out(output)  # (batch_size, max_len, target_vocab_size,)

        return output, attn_weights1
class GeneEncoderConvBlock(nn.Module):
    def __init__(self):
        super(GeneEncoderConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, stride=10, padding=0)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 978) -> (batch, 1, 978)
        x = self.conv(x)    # -> (batch, 1, 97)
        x = x.squeeze(1)    # -> (batch, 97)
        return x
class FullyConnectedCompressBlock(nn.Module):
    def __init__(self, input_dim, seq_len, d_model):
        super(FullyConnectedCompressBlock, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.GELU()
        # 输出维度 = seq_len * d_model
        self.fc2 = nn.Linear(512, seq_len * d_model)

    def forward(self, x):
        # 输入: (batch_size, input_dim)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)   # (batch_size, seq_len * d_model)
        x = x.view(x.size(0), self.seq_len, self.d_model)  # reshape
        return x  # (batch_size, seq_len, d_model)
