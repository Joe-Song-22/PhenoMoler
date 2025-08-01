import torch
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem, DataStructs
import selfies as sf
from rdkit.Chem import AllChem, DataStructs

import torch.nn as nn
def compute_tanimoto_similarity(smiles1, smiles2):
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return 0.0
def selfies_scaffold_mask_v2(decoder_inputs, vocab, pad_token=0):
    """
    生成 SELFIES 掩码矩阵，掩盖非骨架部分（骨架位置为0，其他为1）

    参数:
        decoder_inputs: Tensor, shape (batch_size, seq_len)，token ids
        vocab: Dict[str, int]，SELFIES token → id 的词表
        pad_token: int，用于填充的 token id

    返回:
        scaffold_mask: Tensor, shape (batch_size, seq_len, seq_len)
    """
    inv_vocab = {v: k for k, v in vocab.items()}
    batch_size, seq_len = decoder_inputs.shape
    scaffold_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.float32)

    for i in range(batch_size):
        # 还原 SELFIES 字符串（去除 pad 和特殊 token）
        token_ids = decoder_inputs[i].tolist()
        selfies_tokens = [inv_vocab.get(tok, '') for tok in token_ids if tok != pad_token and tok in inv_vocab]
        selfies_str = ''.join([t for t in selfies_tokens if t not in ('[START]', '[END]')])

        try:
            smiles = sf.decoder(selfies_str)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # 提取骨架并得到其在原分子中的原子索引
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            matches = mol.GetSubstructMatches(scaffold)
            scaffold_atoms = set(matches[0]) if matches else set()

            # 遍历 SELFIES token，确定其对应原子是否属于骨架
            token_mask = [1] * seq_len
            tokens = sf.split_selfies(selfies_str)
            for j, tok in enumerate(tokens):
                try:
                    sub_selfies = tok
                    sub_smiles = sf.decoder(sub_selfies)
                    sub_mol = Chem.MolFromSmiles(sub_smiles)
                    if sub_mol is None:
                        continue

                    match = mol.GetSubstructMatch(sub_mol)
                    if match and all(idx in scaffold_atoms for idx in match):
                        token_mask[j] = 0  # 属于骨架
                except:
                    continue

            # 将 1D 掩码扩展为 2D 掩码矩阵：骨架列不进行掩码（为0）
            for j in range(min(len(token_mask), seq_len)):
                if token_mask[j] == 0:
                    scaffold_mask[i, :, j] = 1  # j列属于骨架，不能被掩码

        except Exception:
            continue

    return scaffold_mask
def selfies_sidechain_mask(decoder_inputs, vocab, pad_token=0):
    """
    生成 SELFIES 掩码矩阵，掩盖骨架部分（骨架为0，其余为1，即侧链可被mask）

    参数:
        decoder_inputs: Tensor, shape (batch_size, seq_len)，token ids
        vocab: Dict[str, int]，SELFIES token → id 的词表
        pad_token: int，用于填充的 token id

    返回:
        sidechain_mask: Tensor, shape (batch_size, seq_len, seq_len)
    """

    inv_vocab = {v: k for k, v in vocab.items()}
    batch_size, seq_len = decoder_inputs.shape
    sidechain_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.float32)

    for i in range(batch_size):
        token_ids = decoder_inputs[i].tolist()
        selfies_tokens = [inv_vocab.get(tok, '') for tok in token_ids if tok != pad_token and tok in inv_vocab]
        selfies_str = ''.join([t for t in selfies_tokens if t not in ('[START]', '[END]')])

        try:
            smiles = sf.decoder(selfies_str)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # 提取骨架原子索引
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            matches = mol.GetSubstructMatches(scaffold)
            scaffold_atoms = set(matches[0]) if matches else set()

            # 构造 token mask：1表示“可以被mask”的侧链
            token_mask = [0] * seq_len  # 初始化为0：表示骨架（不可被mask）
            tokens = sf.split_selfies(selfies_str)

            for j, tok in enumerate(tokens):
                try:
                    sub_smiles = sf.decoder(tok)
                    sub_mol = Chem.MolFromSmiles(sub_smiles)
                    if sub_mol is None:
                        continue
                    match = mol.GetSubstructMatch(sub_mol)
                    if not match or not all(idx in scaffold_atoms for idx in match):
                        token_mask[j] = 1  # 属于侧链，可被mask
                except:
                    continue

            # 构造 mask 矩阵：列为1表示“可以被mask”（侧链）
            for j in range(min(len(token_mask), seq_len)):
                if token_mask[j] == 1:
                    sidechain_mask[i, :, j] = 1  # 允许mask该位置

        except Exception:
            continue

    return sidechain_mask

def selfies_linker_mask(decoder_inputs, vocab, pad_token=0):
    """
    生成 SELFIES 掩码矩阵，掩盖连接子部分（连接子为1，其余为0）

    参数:
        decoder_inputs: Tensor, shape (batch_size, seq_len)，token ids
        vocab: Dict[str, int]，SELFIES token → id 的词表
        pad_token: int，用于填充的 token id

    返回:
        linker_mask: Tensor, shape (batch_size, seq_len, seq_len)
    """
    inv_vocab = {v: k for k, v in vocab.items()}
    batch_size, seq_len = decoder_inputs.shape
    linker_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.float32)

    for i in range(batch_size):
        token_ids = decoder_inputs[i].tolist()
        selfies_tokens = [inv_vocab.get(tok, '') for tok in token_ids if tok != pad_token and tok in inv_vocab]
        selfies_str = ''.join([t for t in selfies_tokens if t not in ('[START]', '[END]')])

        try:
            smiles = sf.decoder(selfies_str)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # 使用标准化和分子碎片拆分，识别连接子部分
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if len(frags) != 1:
                continue  # 跳过非单体结构

            # 使用 Murcko Scaffold 得到锚定结构，linker 是非骨架部分连接两端的结构
            scaffold = Chem.MolToSmiles(Chem.MolFromSmarts(Chem.MolToSmarts(Chem.DeleteSubstructs(mol, Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))))))

            scaffold_atoms = set()
            try:
                scaffold_mol = Chem.MolFromSmiles(scaffold)
                match = mol.GetSubstructMatch(scaffold_mol)
                if match:
                    scaffold_atoms = set(match)
            except:
                pass

            # 使用连接子识别逻辑（非 scaffold 部分 + 与两个锚定区域相连）
            linker_atoms = set(i for i in range(mol.GetNumAtoms()) if i not in scaffold_atoms)

            # 构造 token mask：1表示“可以被mask”的连接子部分
            token_mask = [0] * seq_len
            tokens = sf.split_selfies(selfies_str)

            for j, tok in enumerate(tokens):
                try:
                    sub_smiles = sf.decoder(tok)
                    sub_mol = Chem.MolFromSmiles(sub_smiles)
                    if sub_mol is None:
                        continue
                    match = mol.GetSubstructMatch(sub_mol)
                    if match and all(idx in linker_atoms for idx in match):
                        token_mask[j] = 1  # 属于连接子
                except:
                    continue

            # 构造 mask 矩阵：列为1表示“允许被mask”
            for j in range(min(len(token_mask), seq_len)):
                if token_mask[j] == 1:
                    linker_mask[i, :, j] = 1  # 允许 mask linker 位置

        except Exception:
            continue

    return linker_mask

def selfies_scaffold_mask(decoder_inputs, vocab, pad_token=0):
    """
    对 SELFIES 索引序列进行主干识别并生成掩码矩阵。

    参数:
        decoder_inputs: (batch_size, seq_len) 的 Tensor，每个元素为 token id
        vocab: 字典 {token: id}
        pad_token: int，用于 padding 的 token id

    返回:
        scaffold_mask: (batch_size, seq_len, seq_len)，骨架部分为0，其他为1
    """
    inv_vocab = {v: k for k, v in vocab.items()}
    batch_size, seq_len = decoder_inputs.shape
    scaffold_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.float32)

    for i in range(batch_size):
        # 将 token id 序列转为 SELFIES token
        token_ids = decoder_inputs[i].tolist()
        tokens = [inv_vocab.get(idx, '') for idx in token_ids if idx != pad_token]
        selfies_str = ''.join(tokens).replace('[START]', '').replace('[END]', '')

        try:
            smiles = sf.decoder(selfies_str)
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                scaffold_selfies = sf.encoder(scaffold)
                scaffold_tokens = sf.split_selfies(scaffold_selfies)
                scaffold_token_ids = {vocab[tok] for tok in scaffold_tokens if tok in vocab}

                # 掩盖主干部分
                for j in range(seq_len):
                    if decoder_inputs[i, j].item() in scaffold_token_ids:
                        scaffold_mask[i, :, j] = 1
        except Exception as e:
            # 无法解析就跳过（保留默认mask）
            continue

    return scaffold_mask
def extract_scaffold_mask(decoder_inputs, vocab, pad_token=0):
    """
    计算 decoder_inputs 的核心骨架，并创建掩码（适用于分子片段级别的 vocab）。

    参数:
        decoder_inputs: (batch_size, seq_len)，索引化的分子片段序列
        vocab: 字典 {片段: index}，片段级的 vocab
        pad_token: int, 用于填充的 token（默认 0）

    返回:
        scaffold_mask: (batch_size, seq_len, seq_len)，其中 0 代表掩盖，1 代表可见
    """
    batch_size, seq_len = decoder_inputs.shape
    scaffold_mask = torch.ones((batch_size, seq_len, seq_len), dtype=torch.float32)  # 1 代表可见

    inv_vocab = {v: k for k, v in vocab.items()}  # 反转字典，索引 -> 片段

    for i in range(batch_size):
        # 1️⃣ 将索引转换回 SMILES 片段
        smiles_fragments = [inv_vocab[idx] for idx in decoder_inputs[i].tolist() if idx in inv_vocab]
        smiles_str = "".join(smiles_fragments)  # 连接成 SMILES
        smiles_str = smiles_str.replace("START", "").replace("END", "").replace("PAD", "").strip()

        # 2️⃣ 计算核心骨架
        mol = Chem.MolFromSmiles(smiles_str)
        if mol:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)  # 生成 Murcko Scaffold
            scaffold_tokens = list(scaffold)  # 拆分为字符列表

            # 3️⃣ 转换回索引
            scaffold_indices = {vocab[ch] for ch in scaffold_tokens if ch in vocab}  # 找到骨架对应的索引

            # 4️⃣ 掩盖骨架部分
            for j in range(seq_len):
                if decoder_inputs[i, j].item() in scaffold_indices and decoder_inputs[i, j].item() != pad_token:
                    scaffold_mask[i, :, j] = 0  # 0 代表屏蔽

    return scaffold_mask
def apply_grammar_mask(logits, current_states, grammar_masks):
    """
    - logits: (batch, seq_len, vocab_size) -> Transformer 输出的 logits
    - current_states: (batch, seq_len) -> 记录当前时间步，每个样本的 CFG 状态
    - grammar_masks: (num_nonterminals, vocab_size) -> 语法约束掩码
    """
    batch_size, seq_len, vocab_size = logits.shape
    masks = torch.zeros_like(logits)  # (batch, seq_len, vocab_size)

    # 遍历 batch 和 seq_len，逐步掩码
    for b in range(batch_size):
        for t in range(seq_len):
            state_idx = current_states[b, t]  # 取当前 token 的 CFG 状态索引
            masks[b, t] = grammar_masks[state_idx]  # 取该状态下允许的 token 掩码

    # **屏蔽非法 token（设为 -inf）**
    masked_logits = logits + (masks - 1) * 1e9  # 只允许 mask=1 的 token 被选择

    return masked_logits
# 进行 padding
def pad_tensor(tensor, max_len, pad_value):
    pad_size = max_len - tensor.shape[1]
    if pad_size > 0:
        pad_tensor = torch.full((tensor.shape[0], pad_size), pad_value, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, pad_tensor], dim=1)
    return tensor



# class DynamicCRF(CRF):
#     def __init__(self, num_tags, token_c_id, max_c=4, batch_first=True):
#         super().__init__(num_tags, batch_first=batch_first)
#         self.token_c_id = token_c_id
#         self.max_c = max_c
#
#     def decode(self, emissions, mask=None):
#         """
#         支持 batch_size > 1 的动态 CRF 解码，每个样本限制最多 max_c 个 [C]。
#         emissions: (batch, seq_len, num_tags)
#         mask: (batch, seq_len)
#         return: List[List[int]]  # 每个 batch 的 token 序列
#         """
#         if self.batch_first:
#             emissions = emissions.transpose(0, 1)  # -> (seq_len, batch, num_tags)
#             if mask is not None:
#                 mask = mask.transpose(0, 1)        # -> (seq_len, batch)
#
#         seq_len, batch_size, num_tags = emissions.size()
#         mask = torch.ones(seq_len, batch_size, dtype=torch.bool, device=emissions.device) if mask is None else mask
#
#         decoded_sequences = []
#
#         for b in range(batch_size):
#             emission_b = emissions[:, b, :]      # (seq_len, num_tags)
#             mask_b = mask[:, b]                  # (seq_len,)
#             score = self.start_transitions + emission_b[0]  # (num_tags,)
#             paths = [[i] for i in range(num_tags)]
#             c_counts = [1 if i == self.token_c_id else 0 for i in range(num_tags)]
#
#             for t in range(1, seq_len):
#                 if not mask_b[t]:
#                     break  # padding
#
#                 next_score = torch.full_like(score, -1e9)
#                 next_paths = [[] for _ in range(num_tags)]
#                 next_c_counts = [0 for _ in range(num_tags)]
#
#                 for to_tag in range(num_tags):
#                     for from_tag in range(num_tags):
#                         if c_counts[from_tag] >= self.max_c and to_tag == self.token_c_id:
#                             continue  # 超过 [C] 限制，跳过该转移
#
#                         s = score[from_tag] + self.transitions[to_tag, from_tag] + emission_b[t, to_tag]
#                         if s > next_score[to_tag]:
#                             next_score[to_tag] = s
#                             next_paths[to_tag] = paths[from_tag] + [to_tag]
#                             next_c_counts[to_tag] = c_counts[from_tag] + (1 if to_tag == self.token_c_id else 0)
#
#                 score = next_score
#                 paths = next_paths
#                 c_counts = next_c_counts
#
#             final_score = score + self.end_transitions
#             best_tag = torch.argmax(final_score).item()
#             decoded_sequences.append(paths[best_tag])
#
#         return decoded_sequences
