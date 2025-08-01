import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
# from torch_geometric.nn import Graphormer
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torchcrf import CRF
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch
def create_look_ahead_mask(seq_len):
    mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)  # 上三角矩阵
    return mask  # 形状为 (seq_len, seq_len)
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


class Graphormer(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, num_layers, heads):
        super(Graphormer, self).__init__()
        self.node_feature_transform = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_feature_transform = nn.Linear(edge_feature_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.1)
        self.hidden_dim = hidden_dim

    def forward(self, x, edge_index, batch):
        """
        x: 节点特征 (num_nodes, node_feature_dim)
        edge_index: 边索引 (2, num_edges)
        batch: 节点所属子图的批量索引 (num_nodes,)
        """
        # 升维节点特征
        x = self.node_feature_transform(x)  # (num_nodes, node_feature_dim) -> (num_nodes, hidden_dim)

        # 按照 GCN 层处理所有子图
        for conv in self.layers:
            x = F.relu(conv(x, edge_index))  # 图卷积
            x = self.dropout(x)

        # 将节点特征重新组织为 (batch_size, max_nodes, hidden_dim)
        x, mask = to_dense_batch(x, batch)  # 转为稠密表示，方便后续与 Transformer 结合

        return x # 返回稠密节点嵌入和子图掩码

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
            nn.ReLU(),
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


class DecoderEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super(DecoderEmbedding, self).__init__()
        # 定义一个线性层，将每个维度映射到 d_model（512）维
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
    def forward(self, tgt, d_model):
        # tgt shape: (batch_size, 79)
        # 1. 将输入扩展维度，以便每个维度都能通过线性层进行处理
        tgt = tgt.unsqueeze(-1)  # (batch_size, 79) -> (batch_size, 79, 1)

        # 2. 使用线性层将每个维度映射到 512 维
        tgt = tgt.float()
        tgt_emb = self.embedding(tgt)*np.sqrt(d_model)  # (batch_size, 79, 1) -> (batch_size, 79, 512)
        tgt_emb = self.pos_encoding(tgt_emb)
        return tgt_emb
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, max_len, dropout=0.0):
        super(TransformerEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5, stride=5, padding=0)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=d_model, kernel_size=3, stride=2, padding=0)
        self.conv_activation = nn.ReLU()
        # self.embedding = nn.Linear(d_model, d_model)
        self.enc_layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
                                         for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, d_model, mask=None):
        x = x.unsqueeze(-1)  # (batch, 978) -> (batch, 978, 1)
        x = x.transpose(1, 2)  # (batch, 978, 1) -> (batch, 1, 978)
        x = self.conv_activation(self.conv1(x))  # (batch, 1, 978) -> (batch, 128, new_seq_len)
        x = self.conv_activation(self.conv2(x))  # (batch, 128, new_seq_len) -> (batch, d_model, new_seq_len)

        x = x.transpose(1, 2)  # (batch, d_model, new_seq_len) -> (batch, new_seq_len, d_model)
        # x = self.embedding(x) * np.sqrt(d_model)
        # x = self.pos_encoding(x)
        # x = x.unsqueeze(1)
        x = self.dropout(x)
        batch_size, seq_len, _ = x.size()
        zero_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=x.device)
        for layer in self.enc_layers:
            x = layer(x, zero_mask)
        # x = x.mean(dim=1, keepdim=True)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)  # 自注意力
        self.mha2 = MultiHeadAttention(d_model, num_heads)  # 编码-解码注意力

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
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

        # 自注意力
        attn1, attn_weights1 = self.mha1(x, x, x, tgt_mask)
        out1 = self.layernorm1(x + self.dropout1(attn1))

        # 编码器-解码器注意力（可选）
        if use_cross_attention and enc_output is not None:
            attn2, attn_weights2 = self.mha2(out1, enc_output, enc_output)
            out2 = self.layernorm2(out1 + self.dropout2(attn2))
        else:
            out2 = out1  # 如果不使用编码器-解码器注意力，直接传递 out1
            attn_weights2 = None
        # 前馈神经网络
        ffn_output = self.ffn(out2)
        out3 = self.layernorm3(out2 + self.dropout3(ffn_output))

        return out3, attn_weights1, attn_weights2

class TransLSTMEncoderDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_len, dropout):
        super(TransLSTMEncoderDecoder, self).__init__()
        # self.graph_encoder = Graphormer(node_feature_dim=1, edge_feature_dim=0, hidden_dim=d_model, num_layers=3, heads=4)
        # Encoder 部分
        # self.encoder_embedding = nn.Linear(input_vocab_size, d_model)
        # encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, max_len, dropout=0.0)

        # Decoder 部分
        self.decoder_embedding = DecoderEmbedding(d_model, max_len)
        self.decoder_layer = TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
        self.lstm = nn.LSTM(d_model, d_model, num_layers, batch_first=True)
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
        # enc_src = self.encoder_embedding(src)  # (batch_size, input_vocab_size) -> (batch_size, d_model)
        enc_src = self.encoder(src, d_model)  # Transformer Encoder 输出 (batch_size, seq_len, d_model)

        # Decoder 部分
        # graph_embedding = self.graph_encoder(graph.x, graph.edge_index, graph.batch)
        tgt_emb = self.decoder_embedding(tgt, d_model)  # 将药物的输入序列转换为嵌入 (batch_size, seq_len, d_model)
        lstm_out, hidden = self.lstm(tgt_emb, hidden)
        # seq_len_graph = graph_embedding.size(1)
        # seq_len_tgt = tgt_emb.size(1)

        # 计算需要补零的长度
        # padding_len = seq_len_tgt - seq_len_graph

        # 如果 graph_embedding 比 tgt_emb 短，补零
        # if padding_len > 0:
            # 在第二维（seq_len）补零
            # graph_embedding = F.pad(graph_embedding, (0, 0, 0, padding_len, 0, 0))

        # 如果未传入 mask，则使用默认的 look-ahead mask
        if mask is None:
            mask = create_look_ahead_mask(tgt_emb.shape[1])
            mask = mask.unsqueeze(0).expand(tgt.shape[0], -1, -1)
        output, attn_weights1, attn_weights2 = self.decoder_layer(tgt_emb, enc_src, mask)  # 使用 encoder 的输出作为条件信息
        output = output + lstm_out
        # 生成输出，预测下一个字符
        output = self.norm(output)
        output = self.fc_out(output)  # (batch_size, max_len, target_vocab_size,)

        return output, attn_weights2
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
    def __init__(self):
        super(FullyConnectedCompressBlock, self).__init__()
        self.fc1 = nn.Linear(978, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 97)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))