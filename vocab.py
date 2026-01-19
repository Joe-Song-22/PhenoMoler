import sys
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from rdkit import Chem, DataStructs
import selfies as sf
# 1. 定义字符映射字典（包含起始、终止、PAD）
vocab = {
    'PAD': 0, 'START': 1, 'END': 2,  # 特殊符号
    'H': 3, 'B': 4, 'C': 5, 'c': 6, 'N': 7, 'n': 8, 'O': 9, 'o': 10,
    'P': 11, 'S': 12, 's': 13, 'F': 14, 'I': 15,
    'Q': 16, 'R': 17, 'V': 18, 'Y': 19, 'Z': 20, 'G': 21, 'T': 22, 'U': 23,
    '[': 24, ']': 25, '+': 26, 'W': 27, 'X': 28,
    '-': 29, '=': 30, '#': 31, '.': 32, '/': 33, '@': 34, '\\': 35,
    '(': 36, ')': 37, '1': 38, '2': 39, '3': 40, '4': 41, '5': 42, '6': 43, '7': 44, '8': 45, '9': 46
}
# def ids_batch_to_selfies(generated_ids, id2token):
#     """
#     将形如 (batch, seq) 的 token id tensor 转成 SELFIES 字符串列表
#     """
#     selfies_list = []
#     for sequence in generated_ids:
#         tokens = [id2token[int(idx)] for idx in sequence.cpu().numpy()]
#         # 移除特殊 token
#         tokens = [t for t in tokens if t not in ("[START]", "[END]", "[PAD]")]
#         selfies_str = "".join(tokens)
#         selfies_list.append(selfies_str)
#     return selfies_list
def ids_batch_to_selfies(generated_ids, id2token):
    """
    将 (batch, seq) 的 tensor 或 CRF decode 输出的 List[List[int]] 转成 SELFIES 字符串列表
    """
    selfies_list = []
    for sequence in generated_ids:
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.cpu().numpy()
        tokens = [id2token[int(idx)] for idx in sequence]
        tokens = [t for t in tokens if t not in ("[START]", "[END]", "[PAD]")]
        selfies_str = "".join(tokens)
        selfies_list.append(selfies_str)
    return selfies_list


def selfies_to_smiles(selfies_list):
    smiles_list = []
    for s in selfies_list:
        try:
            smiles = sf.decoder(s)
            smiles_list.append(smiles)
        except:
            smiles_list.append(None)
    return smiles_list
def decode_smiles(encoded_seqs, vocab):
    """解码 batch 里的多个 SMILES"""
    inv_vocab = {v: k for k, v in vocab.items()}  # 反向映射
    decoded_smiles = []

    for seq in encoded_seqs:
        # 处理单个序列
        tokens = [inv_vocab[idx.item()] for idx in seq if idx.item() in inv_vocab]  # 确保 idx.item() 取值

        # 去掉特殊标记
        tokens = [tok for tok in tokens if tok not in ['START', 'END', 'PAD']]

        # 还原替换字符
        smiles = ''.join(tokens)
        smiles = smiles.replace('Q', 'Si').replace('R', 'Cl').replace('V', 'Br') \
            .replace('W', '[H2]').replace('X', '[H3]')

        decoded_smiles.append(smiles)

    return decoded_smiles  # 返回 batch 里的所有 SMILES


def smiles_to_encoded(smiles, vocab):
    """将 SMILES 转换为编码序列"""
    smiles = smiles.replace('Si', 'Q').replace('Cl', 'R').replace('Br', 'V') \
        .replace('[H2]', 'W').replace('[H3]', 'X')

    tokens = ['START']  # 添加起始标记
    i = 0
    while i < len(smiles):
        if i < len(smiles) - 1 and smiles[i:i + 2] in vocab:
            tokens.append(smiles[i:i + 2])
            i += 2
        else:
            tokens.append(smiles[i])
            i += 1
    tokens.append('END')  # 添加终止标记

    return [vocab[tok] for tok in tokens if tok in vocab]
class MyDataset2(Dataset):
    def __init__(self, input_data):
        self.input_data = input_data  # 保留原始不等长的序列，不转为 tensor

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx]  # 返回不等长的原始序列
def collate_fn(batch):
    processed_batch = []
    for item in batch:
        if isinstance(item, str):  # 检查是否为字符串
            try:
                # 如果是字符串，解析为浮点数列表
                num_list = [float(num) for num in item.split(",")]
                processed_batch.append(torch.tensor(num_list, dtype=torch.float32))
            except ValueError:
                raise ValueError(f"无法将字符串解析为浮点数列表: {item}")
        elif isinstance(item, (list, np.ndarray)):  # 如果是列表或 NumPy 数组
            try:
                processed_batch.append(torch.tensor(item, dtype=torch.float32))
            except ValueError:
                raise ValueError(f"无法将列表或数组转换为张量: {item}")
        else:
            raise TypeError(f"无效的数据类型: {type(item)}，数据内容: {item}")

    # 使用 pad_sequence 进行填充
    padded_batch = pad_sequence(processed_batch, batch_first=True, padding_value=0.0)
    return padded_batch


def process_input_data(input_data):
    # 对 input_data 进行任何操作，比如标准化或归一化
    input_data = torch.tensor(input_data, dtype=torch.float32)
    # 可以添加额外的处理逻辑
    return input_data


class MyDataset(Dataset):
    def __init__(self, input_data):
        # 只对 input_data 进行操作
        self.input_data = process_input_data(input_data)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx]
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
def is_valid_smiles(smiles):
    if isinstance(smiles, list):  # 处理列表
        smiles = ''.join(smiles)
    elif isinstance(smiles, torch.Tensor):  # 处理 PyTorch Tensor
        smiles = ''.join([str(s) for s in smiles.tolist()])

    mol = Chem.MolFromSmiles(smiles)
    return mol is not None  # 返回 True 表示是有效 SMILES
def get_reward(smiles, target=None):
    # 奖励机制
    if not is_valid_smiles(smiles):
        return -1  # 无效分子惩罚，负奖励
    # 你可以加入更多的结构合理性检查，来对符合目标的分子给予额外奖励
    # 例如：检查分子的某些特征是否符合目标要求（比如某些功能团、药理活性等）
    # 这里假设我们只检查有效性
    return 1  # 有效分子奖励，正奖励
def compute_loss(outputs, decoder_inputs, target_vocab_size, smiles_generated, target_smiles=None):
    # 常规的交叉熵损失
    loss = F.cross_entropy(outputs.view(-1, target_vocab_size), decoder_inputs.view(-1))

    # 获取奖励/惩罚
    reward = get_reward(smiles_generated, target_smiles)

    # 将奖励加入到损失中
    loss -= reward * 0.1  # 通过调整系数控制奖励/惩罚的强度

    return loss
