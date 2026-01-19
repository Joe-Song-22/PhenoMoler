import pandas as pd
import torch
import csv
device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
from vocab import *
from bert_gpt_mol import *
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from mask import *
from rdkit import Chem
import json
from torch.utils.data import Subset
from torch.nn.functional import softmax
# from torchcrf import CRF
token2id_path = "/data/sr/train_smiles/token2id.json"

# 加载 token -> id 映射
with open(token2id_path, "r", encoding="utf-8") as f:
    token2id = json.load(f)
# 构建 id -> token 映射
id2token = {v: k for k, v in token2id.items()}

# 1. 读取 CSV 文件
# df = pd.read_csv("/data/sr/train_smiles/encoded(1)_smiles.csv")  # 替换为你的 CSV 文件路径
# smiles_list = df.iloc[:, 0].astype(str).tolist()  # 读取第一列
# encoded_data = [smiles_to_encoded(smiles, vocab) for smiles in smiles_list]
# max_len = max(len(seq) for seq in encoded_data)  # 找到最长序列
# pad_token = vocab['PAD']
# tensor_data = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in encoded_data], batch_first=True, padding_value=pad_token)
# torch.save(tensor_data, "/data/sr/train_smiles/encoded_smiles.pt")  # 保存为 PyTorch 格式
# tensor_data = torch.load("/data/sr/train_smiles/encoded_smiles.pt")
# all_smiles = torch.load("/data/sr/train_smiles/encoded_all_smiles.pt")
# PAD_IDX = vocab['PAD']  # 获取 PAD 符号的索引

# 找到最长序列长度
# max_len = max(tensor_data.shape[1], all_smiles.shape[1])
# tensor_data = pad_tensor(tensor_data, max_len, PAD_IDX)
# merged_data = torch.cat([tensor_data, all_smiles], dim=0)
# scaffold = np.load(
#     "/data/sr/train_smiles/selfies_scaffold_tokenized_indexed_padded97.npy",
#     mmap_mode="r"
# )  # (N, 97)
drug = np.load("/data/sr/train_smiles/selfies_tokenized_indexed74740.npy",
               mmap_mode="r")
drug_data = torch.from_numpy(drug).long()

# drug_all = drug_all.tolist()
# drug_data = drug_data.tolist()
gene_data = pd.read_csv('/data/sr/foldchange_cellline.csv').values
cell_label = gene_data[:, 1]
gene_data = gene_data[:, 3:]
cell_label = np.array(cell_label, dtype=np.float32)
gene_data = np.array(gene_data, dtype=np.float32)
# gene = pd.read_csv('/data/sr/MCF7_AKT1.csv').values
# ctl = pd.read_csv('/data/sr/ctl_MCF7.csv').values
# gene = gene[:, 1:]
# ctl = ctl[:, 2:].mean()
# gene = gene - ctl
# gene = np.array(gene, dtype=np.float32)
# drug_all = torch.tensor(drug_data)
# drug_data = torch.tensor(drug_data)
scaffold_layers = 4
full_layers = 2
max_len = drug_data.size(1)
d_model = 256
num_heads = 8  # 这里的 num_heads 一般是 8 而不是 64，因为 64 可能是维度大小错误
d_ff = 1024
# d_model = 12
# num_heads = 1
# d_ff = 24
input_vocab_size = 978
target_vocab_size = len(token2id)
dropout = 0.1
dataset = MyDataset(gene_data)
# test = MyDataset(gene)
encoder_drug0 = MyDataset2(drug_data)
# encoder_drug1 = MyDataset2(drug_data)
cell_label = CustomDataset(cell_label)
# dataset = CustomDataset(dataset)
cell_label_array = np.array(cell_label)  # 转换为NumPy数组方便索引操作
# train_dataset1 = dataset
# label_train = cell_label
# train_dataset2 = encoder_drug1
# test = CustomDataset(test)
test_indices = np.where(cell_label_array == 8)[0]  # 获取 cell_label == 12 的索引
train_indices = np.where(cell_label_array != 10)[0]
# train_indices = np.where(np.isin(cell_label_array, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 58.0]))[0]
  # 获取 cell_label != 12 的索引
train_dataset1 = torch.utils.data.Subset(dataset, train_indices)
train_dataset2 = torch.utils.data.Subset(encoder_drug0, train_indices)
# print(len(train_dataset2))
# print(len(train_dataset1))
# train_indices_all = list(range(len(train_dataset1)))
# 随机选出 1% 的索引
# subset_size = max(1, int(0.01 * len(train_dataset1)))  # 至少选一个
# subset_indices = random.sample(train_indices_all, subset_size)
# train_subset_1percent = Subset(train_dataset1, subset_indices)
# label_train = torch.utils.data.Subset(cell_label, train_indices)
# test_dataset1 = torch.utils.data.Subset(dataset, test_indices)
# label_test = torch.utils.data.Subset(cell_label, test_indices)
# test_dataset2 = torch.utils.data.Subset(encoder_drug1, test_indices)
train_dataloader0 = DataLoader(encoder_drug0, batch_size=32, shuffle=False, drop_last=True)
train_dataloader1 = DataLoader(train_dataset1, batch_size=32, shuffle=False, drop_last=True)
# train_label = DataLoader(label_train, batch_size=16, shuffle=False, drop_last=True)
# train_dataloader2 = DataLoader(train_dataset2, batch_size=16, shuffle=False, drop_last=True)
# test_dataloader1 = DataLoader(test_dataset1, batch_size=16, shuffle=False, drop_last=True)
# test_label = DataLoader(label_test, batch_size=16, shuffle=False, drop_last=False)
# test_dataloader2 = DataLoader(test_dataset2, batch_size=16, shuffle=False, drop_last=True)
# test_dataloader1 = DataLoader(train_subset_1percent, batch_size=32, shuffle=False, drop_last=False)

# model1 = TransLSTMEncoderDecoder(scaffold_layers, full_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_len, dropout=0.1).to(device)
model2 = TransLSTMEncoderDecoder(scaffold_layers, full_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_len, dropout=0.1).to(device)
criterion = nn.CrossEntropyLoss()
# crf = CRF(num_tags=target_vocab_size, batch_first=True).to(device)
# crf = DynamicCRF(num_tags=target_vocab_size, token_c_id=token2id['[C]'], max_c=4).to(device)
# conv_block = GeneEncoderConvBlock().to(device)
conv_block = FullyConnectedCompressBlock(input_dim=978, seq_len=1, d_model=d_model).to(device)


# optimizer1 = optim.Adam(model1.parameters(), lr=1e-4, weight_decay=1e-6)
# optimizer2 = optim.Adam(list(model2.parameters()) + list(conv_block.parameters()), lr=1e-4, weight_decay=1e-6)
# optimizer3 = optim.Adam(list(model2.parameters()) + list(crf.parameters()), lr=1e-5, weight_decay=1e-6)
MSE = nn.MSELoss()
# optimizer3 = optim.Adam(conv_block.parameters(),lr=1e-5, weight_decay=1e-6)
num_epoch = 2
best_loss0 = float('inf')
best_loss1 = float('inf')
losses0 = []
losses1 = []
start_token = token2id["[START]"]
end_token = token2id["[END]"]
pad_token = token2id["[PAD]"]
# state_dict0 = torch.load(
#     "/data/sr/train_smiles/tcga_fourier_druglstmformer9_3.pth",
#     map_location="cpu"
# )
# model1.load_state_dict(state_dict0)
# for param in model1.encoder.parameters():
#     param.requires_grad = False
# model2.load_state_dict(torch.load('/data/sr/train_smiles/tcga_fourier_druglstmformer9_3.pth'))
# ===== 1. 加载模型参数 =====
ckpt_path = "/data/sr/train_smiles/full_layers_gelu_druglstmformer.pth"
# ckpt_path = "/data/sr/train_smiles/full_layers_gelu_druglstmformer.pth"
state_dict = torch.load(ckpt_path, map_location=device)
model2.load_state_dict(state_dict, strict=False)

# ===== 2. 冻结所有参数 =====
for param in model2.parameters():
    param.requires_grad = False

# ===== 3. 只解冻 decoder =====
for layer in model2.decoder.layers:
    for p in layer.mha2.parameters():
        p.requires_grad = True
for layer in model2.scaffold.layers:
    for p in layer.mha2.parameters():
        p.requires_grad = True

# # ===== 4. 构建优化器 =====
# optimizer1 = torch.optim.Adam(
#     filter(lambda p: p.requires_grad, model1.parameters()),
#     lr=1e-4
# )
optimizer2 = torch.optim.Adam(
    filter(lambda p: p.requires_grad, list(model2.parameters()) + list(conv_block.parameters())),
    lr=1e-4
)
# ===== 5. 验证 =====
print("Trainable params:")
for name, p in model2.named_parameters():
    if p.requires_grad:
        print(name)
for name, p in conv_block.named_parameters():
    if p.requires_grad:
        print(name)
# for epoch in range(num_epoch):
#     model2.train()
#     running_loss = 0.0
#     num_batches = 0
#     outputs_smiles_epoc = []
#     progress_bar = tqdm(train_dataloader0, desc=f"Epoch {epoch + 1}/{num_epoch}")
#     total_batches = len(progress_bar)
#     # for decoder_inputs in progress_bar:
#     for batch_idx, decoder_inputs in enumerate(progress_bar):
#         num_batches += 1
#         data_size = len(decoder_inputs)  # 数据大小
#         # indices = list(range(data_size))  # 索引列表
#         # random.shuffle(indices)  # 随机打乱索引
#         # decoder_inputs = decoder_inputs[indices] if isinstance(decoder_inputs, torch.Tensor) else [decoder_inputs[i] for
#         #                                                                                            i in indices]
#         batch_size = decoder_inputs.shape[0]
#         # decoder_inputs = decoder_inputs.to(device)
#         # decoder_inputs_used = torch.full_like(decoder_inputs, pad_token)
#         # decoder_inputs_used[:, 0] = start_token  # 仅第一个位置为 [START]
#         # encoder_inputs = decoder_inputs.float()
#         # ===== 自回归准备：错位 =====
#         decoder_inputs_in = decoder_inputs[:, :-1].to(device)  # 去掉最后一个 token，用作 decoder 输入
#         decoder_targets = decoder_inputs[:, 1:].to(device)  # 去掉第一个 token，用作目标输出
#         # ===== 掩码生成 =====
#         # mask = selfies_scaffold_mask(decoder_inputs_in, vocab)
#         seq_len = decoder_inputs_in.size(1)
#         mask = create_look_ahead_mask(seq_len, device=decoder_inputs_in.device)
#         mask = mask.unsqueeze(0).expand(decoder_inputs_in.shape[0], -1, -1)
#         output, _ = model2.scaffold(decoder_inputs_in, None, mask)
#         output, _ = model2.decoder(model2.norm(output), None, mask)
#         output = model2.fc_out(model2.norm(output))
#
#         # scaffold_mask = extract_scaffold_mask(decoder_inputs, vocab)
#         # scaffold_mask = selfies_scaffold_mask(decoder_inputs, token2id, pad_token=token2id["[PAD]"])
#         # tgt_emb = model1.decoder_embedding(decoder_inputs_in, d_model)
#         # lstm_out, hidden = model1.lstm(tgt_emb, None)
#         # mask = create_look_ahead_mask(tgt_emb.shape[1])
#         # mask = mask.unsqueeze(0).expand(decoder_inputs.shape[0], -1, -1)
#         # output, attn_weight1, attn_weight2 = model1.decoder_layer(tgt_emb, enc_output=None, tgt_mask=None, use_cross_attention=False)
#         # mask = create_look_ahead_mask(tgt_emb.shape[1]).unsqueeze(0).expand(batch_size, -1, -1)
#         # output = output + lstm_out
#         # output = model1.norm(output)
#         # output = model1.fc_out(output)
#         decoder_targets = decoder_targets.contiguous().view(-1)  # (B * L)
#         outputs = output.view(-1, target_vocab_size)  # (B * L, vocab)
#
#         loss = criterion(outputs, decoder_targets)
#         # decoder_inputs = decoder_inputs.view(-1)
#         # loss = criterion(output, decoder_inputs_one_hot)
#         # pad_token_id = token2id["[PAD]"]
#         # loss_mask = decoder_targets != pad_token_id              # (B * L)
#         # filtered_output = output[loss_mask]
#         # filtered_target = decoder_targets[loss_mask]
#
#         # loss = criterion(output.view(-1, target_vocab_size), decoder_inputs.view(-1))
#         # ce_loss = criterion(filtered_output, filtered_target)
#         # loss = ce_loss
#         optimizer2.zero_grad()
#         loss.backward()
#         optimizer2.step()
#         running_loss += loss.item()
#         progress_bar.set_postfix(loss=loss.item())
#
#         # generated_selfies = torch.argmax(output, dim=-1)
#         # generated_selfies = ids_batch_to_selfies(generated_selfies, id2token)
#         # outputs_smiles = selfies_to_smiles(generated_selfies)
#         # outputs_smiles_epoc.append(outputs_smiles)
for epoch in range(num_epoch):
    model2.train()
    conv_block.train()
    running_loss = 0.0
    total_train_samples = 0
    num_batches = 0
    outputs_smiles_epoc = []
    progress_bar = tqdm(zip(train_dataloader1, train_dataloader0, ), desc=f"Epoch {epoch + 1}/{num_epoch}")
    # 判断本epoch是argmax还是采样
    # use_sampling = (epoch // 10) % 2 == 0  # 每10轮切换一次模式
    for (encoder_inputs), (decoder_inputs) in progress_bar:
        num_batches += 1
        batch_size = encoder_inputs.size(0)
        total_train_samples += batch_size
        data_size = len(encoder_inputs)  # 数据大小
        # indices = list(range(data_size))  # 索引列表
        # random.shuffle(indices)  # 随机打乱索引
        # encoder_inputs = encoder_inputs[indices] if isinstance(encoder_inputs, torch.Tensor) else [encoder_inputs[i] for
        #                                                                                            i in indices]
        # decoder_inputs = decoder_inputs[indices] if isinstance(decoder_inputs, torch.Tensor) else [decoder_inputs[i] for
        #                                                                                            i in indices]
        encoder_inputs = encoder_inputs.to(device)
        decoder_inputs = decoder_inputs.to(device)
        decoder_inputs_in = decoder_inputs[:, :-1]
        # decoder_inputs_0 = decoder_inputs.float()
        decoder_targets = decoder_inputs[:, 1:]  # 去掉第一个 token，用作目标输出
        encoder_inputs = conv_block(encoder_inputs)
        seq_len = decoder_inputs_in.size(1)
        mask = create_look_ahead_mask(seq_len, device=decoder_inputs_in.device)
        mask = mask.unsqueeze(0).expand(decoder_inputs_in.shape[0], -1, -1)
        output, _ = model2.scaffold(decoder_inputs_in, encoder_inputs, mask)
        output, _ = model2.decoder(model2.norm(output), encoder_inputs, mask)
        output = model2.fc_out(model2.norm(output))
        loss = criterion(output.view(-1, target_vocab_size), decoder_targets.contiguous().view(-1))
        #         loss_align = F.mse_loss(encoder_inputs, enc_logits)
        #         loss = loss0 + loss_align
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

        # # ✅ 只保存前500或后500个 batch 的生成分子
        # if batch_idx < 500 or batch_idx >= total_batches - 500:
        #     generated_selfies = torch.argmax(output, dim=-1)
        #     generated_selfies = ids_batch_to_selfies(generated_selfies, id2token)
        #     outputs_smiles = selfies_to_smiles(generated_selfies)
        #     outputs_smiles_epoc.append(outputs_smiles)
        # if batch_idx >= total_batches - 100:
        if num_batches >= 4600:
            generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
            for _ in range(max_len - 1):  # 已经有1个token了，还要生成max_len-1个
                # ===== 模型前向 =====
                # logits, _ = model1(encoder_inputs, generated, d_model, None, None)  # [B, L, V]
                logits, _ = model2.scaffold(generated, encoder_inputs, None)
                logits, _ = model2.decoder(model2.norm(logits), encoder_inputs, None)
                logits = model2.fc_out(model2.norm(logits))
                next_token_logits = logits[:, -1, :]  # 取最后一步预测 [B, V]

                # ===== 采样策略 =====
                # greedy search（取最大概率的token）
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)  # [B, 1]

                # 如果你想要采样而不是贪心，可以用 multinomial：
                # probs = F.softmax(next_token_logits, dim=-1)
                # next_token = torch.multinomial(probs, num_samples=1)
                # ===== 拼接到已生成序列 =====
                generated = torch.cat([generated, next_token], dim=1)

                # ===== 如果全batch都预测到 pad_id，可以提前停止 =====
                if (next_token == end_token).all():
                    break
            generated_selfies = ids_batch_to_selfies(generated, id2token)
            outputs_smiles = selfies_to_smiles(generated_selfies)
            outputs_smiles_epoc.append(outputs_smiles)
    avg_loss = running_loss
    losses0.append(avg_loss)
    if running_loss < best_loss0:
        best_loss0 = running_loss
        torch.save(model2.state_dict(), '/data/sr/train_smiles/full_layers_gelu_druglstmformer.pth')
        torch.save(conv_block.state_dict(), '/data/sr/train_smiles/gene_block.pth')
        print(f'Best model saved at epoch {epoch + 1} with Test Loss: {best_loss0}')
    for smiles in outputs_smiles_epoc:
        print(smiles)