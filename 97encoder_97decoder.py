import pandas as pd
import torch
import csv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from vocab import *
from trans_lstm import *
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from mask import *
from rdkit import Chem
import json
from torch.utils.data import Subset
from torch.nn.functional import softmax
from torchcrf import CRF
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
# drug_all = pd.read_csv('/data/sr/train_smiles/selfies_tokenized_indexed.csv').values
drug_data = pd.read_csv('/data/sr/train_smiles/selfies_tokenized_indexed74740.csv').values
# drug_all = drug_all.tolist()
drug_data = drug_data.tolist()
gene_data = pd.read_csv('/data/sr/foldchange_cellline.csv').values
cell_label = gene_data[:, 1]
gene_data = gene_data[:, 3:]
cell_label = np.array(cell_label, dtype=np.float32)
gene_data = np.array(gene_data, dtype=np.float32)
gene = pd.read_csv('/data/sr/MCF7_AKT1.csv').values
ctl = pd.read_csv('/data/sr/ctl_MCF7.csv').values
gene = gene[:, 1:]
ctl = ctl[:, 2:].mean()
gene = gene - ctl
gene = np.array(gene, dtype=np.float32)
# drug_all = torch.tensor(drug_all)
drug_data = torch.tensor(drug_data)
num_layers = 6
max_len = drug_data.size(1)
d_model = 256
num_heads = 8  # 这里的 num_heads 一般是 8 而不是 64，因为 64 可能是维度大小错误
d_ff = 1024
input_vocab_size = 978
target_vocab_size = len(token2id)
dropout = 0.0
dataset = MyDataset(gene_data)
test = MyDataset(gene)
# encoder_drug0 = MyDataset2(drug_all)
encoder_drug1 = MyDataset2(drug_data)
cell_label = CustomDataset(cell_label)
dataset = CustomDataset(dataset)
cell_label_array = np.array(cell_label)  # 转换为NumPy数组方便索引操作
# train_dataset1 = dataset
label_train = cell_label
# train_dataset2 = encoder_drug1
test = CustomDataset(test)
test_indices = np.where(cell_label_array == 8)[0]  # 获取 cell_label == 12 的索引
train_indices = np.where(cell_label_array != 10)[0]
# train_indices = np.where(np.isin(cell_label_array, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 58.0]))[0]
  # 获取 cell_label != 12 的索引
train_dataset1 = torch.utils.data.Subset(dataset, train_indices)
train_dataset2 = torch.utils.data.Subset(encoder_drug1, train_indices)
train_indices_all = list(range(len(train_dataset1)))
# 随机选出 1% 的索引
subset_size = max(1, int(0.01 * len(train_dataset1)))  # 至少选一个
subset_indices = random.sample(train_indices_all, subset_size)
train_subset_1percent = Subset(train_dataset1, subset_indices)
# label_train = torch.utils.data.Subset(cell_label, train_indices)
test_dataset1 = torch.utils.data.Subset(dataset, test_indices)
label_test = torch.utils.data.Subset(cell_label, test_indices)
test_dataset2 = torch.utils.data.Subset(encoder_drug1, test_indices)
# train_dataloader0 = DataLoader(encoder_drug0, batch_size=128, shuffle=False, drop_last=False)
train_dataloader1 = DataLoader(train_dataset1, batch_size=16, shuffle=False, drop_last=True)
train_label = DataLoader(label_train, batch_size=16, shuffle=False, drop_last=True)
train_dataloader2 = DataLoader(train_dataset2, batch_size=16, shuffle=False, drop_last=True)
test_dataloader1 = DataLoader(test_dataset1, batch_size=16, shuffle=False, drop_last=False)
test_label = DataLoader(label_test, batch_size=16, shuffle=False, drop_last=False)
test_dataloader2 = DataLoader(test_dataset2, batch_size=16, shuffle=False, drop_last=False)
# test_dataloader1 = DataLoader(train_subset_1percent, batch_size=32, shuffle=False, drop_last=False)

model1 = TransLSTMEncoderDecoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_len, dropout=0.1).to(device)
model2 = TransLSTMEncoderDecoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_len, dropout=0.1).to(device)
criterion = nn.CrossEntropyLoss()
# crf = CRF(num_tags=target_vocab_size, batch_first=True).to(device)
# crf = DynamicCRF(num_tags=target_vocab_size, token_c_id=token2id['[C]'], max_c=4).to(device)
conv_block = GeneEncoderConvBlock().to(device)
optimizer1 = optim.Adam(model1.parameters(), lr=1e-5, weight_decay=0.0)
optimizer2 = optim.Adam(model2.parameters(), lr=1e-5, weight_decay=1e-6)
# optimizer3 = optim.Adam(list(model2.parameters()) + list(crf.parameters()), lr=1e-5, weight_decay=1e-6)

num_epoch = 200
best_loss0 = float('inf')
best_loss1 = float('inf')
losses0 = []
losses1 = []
# model1.load_state_dict(torch.load('/data/sr/train_smiles/gpt_druglstmformer_mcf7.pth'))
# for epoch in range(num_epoch):
#     model1.train()
#     running_loss = 0.0
#     num_batches = 0
#     outputs_smiles_epoc = []
#     progress_bar = tqdm(train_dataloader0, desc=f"Epoch {epoch + 1}/{num_epoch}")
#     for decoder_inputs in progress_bar:
#         num_batches += 1
#         data_size = len(decoder_inputs)  # 数据大小
#         indices = list(range(data_size))  # 索引列表
#         random.shuffle(indices)  # 随机打乱索引
#         decoder_inputs = decoder_inputs[indices] if isinstance(decoder_inputs, torch.Tensor) else [decoder_inputs[i] for
#                                                                                                    i in indices]
#         batch_size = decoder_inputs.shape[0]
#         decoder_inputs = decoder_inputs.to(device)
#         # ===== 自回归准备：错位 =====
#         decoder_inputs_in = decoder_inputs[:, :-1]  # 去掉最后一个 token，用作 decoder 输入
#         decoder_targets = decoder_inputs[:, 1:]  # 去掉第一个 token，用作目标输出
#         # scaffold_mask = extract_scaffold_mask(decoder_inputs, vocab)
#         # scaffold_mask = selfies_scaffold_mask(decoder_inputs, token2id, pad_token=token2id["[PAD]"])
#         tgt_emb = model1.decoder_embedding(decoder_inputs_in, d_model)
#         lstm_out, hidden = model1.lstm(tgt_emb, None)
#         mask = create_look_ahead_mask(tgt_emb.shape[1])
#         mask = mask.unsqueeze(0).expand(decoder_inputs.shape[0], -1, -1)
#         output, attn_weight1, attn_weight2 = model1.decoder_layer(tgt_emb, enc_output=None, tgt_mask=None, use_cross_attention=False)
#         # mask = create_look_ahead_mask(tgt_emb.shape[1]).unsqueeze(0).expand(batch_size, -1, -1)
#         output = output + lstm_out
#         output = model1.norm(output)
#         output = model1.fc_out(output)
#         decoder_targets = decoder_targets.contiguous().view(-1)  # (B * L)
#         output = output.view(-1, target_vocab_size)  # (B * L, vocab)
#         # decoder_inputs = decoder_inputs.long()
#         # decoder_inputs_one_hot = F.one_hot(decoder_inputs, num_classes=target_vocab_size).float()
#         # loss = criterion(outputs.view(-1, target_vocab_size), decoder_inputs.view(-1))
#         # decoder_inputs = decoder_inputs.view(-1)
#         # loss = criterion(output, decoder_inputs_one_hot)
#         pad_token_id = token2id["[PAD]"]
#         loss_mask = decoder_targets != pad_token_id              # (B * L)
#         filtered_output = output[loss_mask]
#         filtered_target = decoder_targets[loss_mask]
#
#         # loss = criterion(output.view(-1, target_vocab_size), decoder_inputs.view(-1))
#         ce_loss = criterion(filtered_output, filtered_target)
#         loss = ce_loss
#         optimizer1.zero_grad()
#         loss.backward()
#         optimizer1.step()
#         running_loss += loss.item()
#         progress_bar.set_postfix(loss=loss.item())
#     model1.eval()
#     with torch.no_grad():
#         test_start = token2id["[START]"]
#         test_pad = token2id["[PAD]"]
#         start_pad_batch = torch.full((128, max_len), fill_value=test_pad, dtype=torch.long, device=device)
#         start_pad_batch[:, 0] = test_start  # 仅第一位为 [START]
#
#         generated = start_pad_batch.clone()
#         for t in range(1, max_len):
#             decoder_inputs_infer = generated[:, :t]  # 截取已生成的前 t 个 token
#             tgt_emb = model1.decoder_embedding(decoder_inputs_infer, d_model)
#             lstm_out, _ = model1.lstm(tgt_emb)
#
#
#             mask = create_look_ahead_mask(tgt_emb.shape[1]).to(device)
#             mask = mask.unsqueeze(0).expand(tgt_emb.shape[0], -1, -1)
#
#             output, _, _ = model1.decoder_layer(tgt_emb, enc_output=None, tgt_mask=mask, use_cross_attention=False)
#             output = output + lstm_out
#             output = model1.norm(output)
#             output = model1.fc_out(output)  # (B, t, vocab)
#
#             next_token_logits = output[:, -1, :]  # 只取最后一个位置的预测
#             next_token = torch.argmax(next_token_logits, dim=-1)  # greedy search
#
#             generated[:, t] = next_token  # 放入下一位
#
#         generated_selfies = ids_batch_to_selfies(generated, id2token)
#         outputs_smiles = selfies_to_smiles(generated_selfies)
#         outputs_smiles_epoc.append(outputs_smiles)
#
#         print(f"\n[Epoch {epoch + 1}] Generated SMILES from [START]+[PAD] input:")
#         for i, s in enumerate(outputs_smiles):
#             print(f"[Gen {i + 1:03d}] {s}")
#
#         # outputs_smiles = torch.argmax(output, dim=-1)
#         # outputs_smiles = decode_smiles(outputs_smiles, vocab)
#         # if epoch == num_epoch - 1:
#         #     generated_selfies = torch.argmax(output, dim=-1)
#         #     generated_selfies = ids_batch_to_selfies(generated_selfies, id2token)
#         #     outputs_smiles = selfies_to_smiles(generated_selfies)
#         #     outputs_smiles_epoc.append(outputs_smiles)
#     avg_loss = running_loss
#     losses0.append(avg_loss)
#     if running_loss < best_loss0:
#         best_loss0 = running_loss
#         torch.save(model1.state_dict(), '/data/sr/train_smiles/gpt_druglstmformer.pth')
#         print(f'Best model saved at epoch {epoch + 1} with Test Loss: {best_loss0}')
    # for smiles in outputs_smiles_epoc:
    #     print(smiles)
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, num_epoch + 1), losses0, label="Training Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss Curve")
# plt.legend()
# # plt.savefig('/data/sr/drugformer_new.jpg')
# plt.show()
# model2.load_state_dict(torch.load('/data/sr/train_smiles/scaffold_druglstmformer.pth'))
# model2.load_state_dict(torch.load('/data/sr/train_smiles/new_train_smiles.pth'))
# model2.encoder.load_state_dict(torch.load('/data/sr/drugformer_encoder.pth'))
# model2.load_state_dict(torch.load('/data/sr/train_smiles/gpt_druglstmformer.pth'))

# 1. 加载预训练模型（安全模式）
try:
    pretrained = torch.load(
        '/data/sr/train_smiles/gpt_druglstmformer.pth',
        map_location='cpu',
        weights_only=True
    )
except:
    pretrained = torch.load(
        '/data/sr/train_smiles/gpt_druglstmformer.pth',
        map_location='cpu',
        weights_only=False
    )

# 2. 直接提取目标组件参数（假设pretrained是state_dict）
components = {
    'decoder_embedding': model2.decoder_embedding,
    'decoder_layer': model2.decoder_layer,
    'lstm': model2.lstm,
    'norm': model2.norm,
    'fc_out': model2.fc_out
}

# 3. 暴力加载匹配参数
for name, component in components.items():
    try:
        # 尝试直接加载完整组件
        component.load_state_dict(pretrained[name])
        print(f"✅ {name} 加载成功")
    except:
        # 失败时尝试部分匹配
        current_sd = component.state_dict()
        pretrained_sd = {k: v for k, v in pretrained.items() if k.startswith(name)}
        if pretrained_sd:
            current_sd.update(pretrained_sd)
            component.load_state_dict(current_sd, strict=False)
            print(f"⚠️ {name} 部分参数加载成功")
        else:
            print(f"❌ {name} 无匹配参数")
# 假设 model2 里有这些模块
modules_to_freeze = [
    model2.decoder_embedding,
    model2.decoder_layer,
    model2.lstm,
    model2.norm,
    model2.fc_out
]

for module in modules_to_freeze:
    for param in module.parameters():
        param.requires_grad = False

optimizer = torch.optim.Adam(
    list(filter(lambda p: p.requires_grad, model2.parameters())) +
    list(conv_block.parameters()),
    lr=1e-5,
    weight_decay=1e-6
)


best_loss = float('inf')
losses = []
for epoch in range(num_epoch):
    model2.train()
    conv_block.train()
    running_loss = 0.0
    num_batches = 0
    outputs_smiles_epoc = []
    progress_bar = tqdm(zip(train_dataloader1, train_dataloader2,), desc=f"Epoch {epoch + 1}/{num_epoch}")
    # 判断本epoch是argmax还是采样
    # use_sampling = (epoch // 10) % 2 == 0  # 每10轮切换一次模式
    for (encoder_inputs), (decoder_inputs) in progress_bar:
        num_batches += 1
        data_size = len(encoder_inputs)  # 数据大小
        indices = list(range(data_size))  # 索引列表
        random.shuffle(indices)  # 随机打乱索引
        encoder_inputs = encoder_inputs[indices] if isinstance(encoder_inputs, torch.Tensor) else [encoder_inputs[i] for
                                                                                                   i in indices]
        decoder_inputs = decoder_inputs[indices] if isinstance(decoder_inputs, torch.Tensor) else [decoder_inputs[i] for
                                                                                                   i in indices]
        encoder_inputs = encoder_inputs.to(device)
        decoder_inputs = decoder_inputs.to(device)
        decoder_inputs = decoder_inputs.long()
        # decoder_targets = decoder_inputs[:, 1:]  # 去掉第一个 token，用作目标输出
        encoder_inputs = conv_block(encoder_inputs)
        tgt_emb = model2.decoder_embedding(encoder_inputs, d_model)
        lstm_out, _ = model2.lstm(tgt_emb)
        output, attn_weight1, attn_weight2 = model2.decoder_layer(tgt_emb, enc_output=None, tgt_mask=None,
                                                                  use_cross_attention=False)
        output = output + lstm_out
        output = model2.norm(output)
        output = model2.fc_out(output)
        loss = criterion(output.view(-1, target_vocab_size), decoder_inputs.contiguous().view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        generated_ids = torch.argmax(output, dim=-1)
        generated_selfies = ids_batch_to_selfies(generated_ids, id2token)
        outputs_smiles = selfies_to_smiles(generated_selfies)
        outputs_smiles_epoc.append(outputs_smiles)
        # scaffold_mask = selfies_linker_mask(decoder_inputs, token2id, pad_token=token2id["[PAD]"])
        # output,  _ = model2(encoder_inputs, decoder_inputs, d_model, None, scaffold_mask)
        # loss = criterion(output.view(-1, target_vocab_size), decoder_inputs.view(-1))
        # 不打乱数据
        # encoder_inputs = encoder_inputs.to(device)
        # decoder_inputs = decoder_inputs.to(device).long()
        #
        # # 构造全为 [PAD] 的输入序列，只保留第一个 [START]
        # pad_token = token2id["[PAD]"]
        # start_token = token2id["[START]"]
        #
        # # decoder_inputs_used = torch.full_like(decoder_inputs, pad_token)
        # # decoder_inputs_used[:, 0] = start_token  # 仅第一个位置为 [START]
        # if epoch % 2 == 1:
        #     # 奇数 epoch：使用正常 decoder_inputs + look-ahead mask（即自回归训练）
        #     decoder_inputs_used = decoder_inputs.clone()
        #     output, _ = model2(encoder_inputs, decoder_inputs_used, d_model, None, None)
        #     loss = -crf(output, decoder_inputs, mask=(decoder_inputs != pad_token), reduction='mean')
        #     print(f"Epoch {epoch + 1}: using look-ahead mask for autoregressive decoding.")
        #
        # else:
        #     # 偶数 epoch：使用全 [PAD] + [START]，即纯 phenotype 引导 + C 限制
        #     decoder_inputs_used = torch.full_like(decoder_inputs, pad_token)
        #     decoder_inputs_used[:, 0] = start_token
        #     output, _ = model2(encoder_inputs, decoder_inputs_used, d_model, None, None)
        #
        #     mask = (decoder_inputs != pad_token)
        #     loss = -crf(output, decoder_inputs, mask=mask, reduction='mean')
        #
        # # 前向传播，mask=None
        # mask = (decoder_inputs != pad_token)
        # # loss = -crf(output, decoder_inputs, mask=mask, reduction='mean')
        # optimizer3.zero_grad()
        # loss.backward()
        # optimizer3.step()
        # running_loss += loss.item()
        # preds = crf.decode(output, mask=mask)
        # generated_selfies = ids_batch_to_selfies(preds, id2token)
        # outputs_smiles = selfies_to_smiles(generated_selfies)
        # outputs_smiles_epoc.append(outputs_smiles)
        # # ===================== 高斯权重 + 多样性采样 =====================
        # vocab_size = output.size(-1)
        # batch_size, seq_len = output.size(0), output.size(1)
        #
        # token_ids = torch.arange(vocab_size, device=device)
        # mean = vocab_size / 2
        # std = vocab_size / 4
        # gaussian_weights = torch.exp(-0.5 * ((token_ids - mean) / std) ** 2)  # (vocab,)
        # gaussian_weights = gaussian_weights / gaussian_weights.sum()  # 归一化
        #
        # # softmax + 高斯调节
        # probs = softmax(output / 1.0, dim=-1)  # 可调 temperature
        # probs = probs * gaussian_weights  # 加权
        # probs = probs / probs.sum(dim=-1, keepdim=True)  # 再归一化 (batch, seq, vocab)
        #
        # # 每个位置采样一个 token
        # sampled_ids = torch.multinomial(probs.view(-1, vocab_size), num_samples=1).squeeze(1)
        # sampled_ids = sampled_ids.view(batch_size, seq_len)
        # # ================================================================
        # loss = criterion(output.view(-1, target_vocab_size), decoder_inputs.view(-1))
        # ========== 1. Softmax 得到概率分布 ==========
        # probs = torch.softmax(output, dim=-1)  # (batch, seq, vocab)
        #
        # # ========== 2. 计算每个位置的期望 token 索引（expected token id） ==========
        # vocab_ids = torch.arange(target_vocab_size, device=device).float()  # (vocab,)
        # expected_token_ids = torch.sum(probs * vocab_ids, dim=-1)  # (batch, seq)
        #
        # # ========== 3. 构造真实 token 索引 ==========
        # true_token_ids = decoder_inputs.float()  # (batch, seq)
        #
        # # ========== 4. 构造 PAD 掩码（1 表示非 PAD）==========
        # pad_token = token2id["[PAD]"]
        # mask = (decoder_inputs != pad_token).float()  # (batch, seq)
        #
        # # ========== 5. 计算 masked MSE Loss ==========
        # # 逐元素平方误差
        # mse_loss = (expected_token_ids - true_token_ids) ** 2  # (batch, seq)
        #
        # # 只在非PAD位置计算 loss
        # masked_loss = mse_loss * mask
        #
        # # 防止除以0，加 epsilon
        # loss = masked_loss.sum() / (mask.sum() + 1e-8)
        # optimizer2.zero_grad()
        # loss.backward()
        # optimizer2.step()
        # running_loss += loss.item()
        # progress_bar.set_postfix(loss=loss.item())
        #
        # # # if epoch == num_epoch - 1:
        # # generated_selfies = torch.argmax(output, dim=-1)
        # # generated_selfies = ids_batch_to_selfies(generated_selfies, id2token)
        # # outputs_smiles = selfies_to_smiles(generated_selfies)
        # # outputs_smiles_epoc.append(outputs_smiles)
        # rounded_ids = torch.round(expected_token_ids).long().clamp(0, target_vocab_size - 1)  # (batch, seq)
        # # 将 sampled_ids 转成 SMILES
        # generated_selfies = ids_batch_to_selfies(rounded_ids, id2token)
        # outputs_smiles = selfies_to_smiles(generated_selfies)
        # outputs_smiles_epoc.append(outputs_smiles)
    avg_loss = running_loss / num_batches
    losses1.append(avg_loss)
    if running_loss < best_loss1:
        best_loss1 = running_loss
        torch.save(model2.state_dict(), '/data/sr/train_smiles/TCGA_druglstmformer.pth')
        # torch.save(conv_block.state_dict(), '/data/sr/train_smiles/TCGA_conv.pth')
        print(f'Best model saved at epoch {epoch + 1} with Test Loss: {best_loss1}')
    for smiles in outputs_smiles_epoc:
        print(smiles)
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epoch + 1), losses1, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.show()
# test_smiles = []
# true_smiles = []
#
# with torch.no_grad():
#     for encoder_inputs, decoder_inputs in tqdm(zip(test_dataloader1, test_dataloader2), desc="Generating Molecules"):
#         encoder_inputs = encoder_inputs.to(device)
#         decoder_inputs = decoder_inputs.to(device)
#         decoder_inputs = decoder_inputs.long()
#
#         # 构造骨架mask
#         scaffold_mask = selfies_scaffold_mask(decoder_inputs, token2id, pad_token=token2id["[PAD]"])
#
#         # 模型生成
#         optimizer2.zero_grad()
#         output, _ = model2(encoder_inputs, decoder_inputs, d_model, None, scaffold_mask)
#         generated_ids = torch.argmax(output, dim=-1)
#
#         # 转换为SELFIES和SMILES
#         generated_selfies = ids_batch_to_selfies(generated_ids, id2token)
#         outputs_smiles = selfies_to_smiles(generated_selfies)
#         test_smiles.extend(outputs_smiles)  # list of strings
#
#         # 保存真实的SMILES
#         true_selfies = ids_batch_to_selfies(decoder_inputs, id2token)
#         true_smiles_batch = selfies_to_smiles(true_selfies)
#         true_smiles.extend(true_smiles_batch)
#
# # 保存为 CSV，包含生成和真实 SMILES
# df = pd.DataFrame({'Generated_SMILES': test_smiles, 'True_SMILES': true_smiles})
# df.to_csv("/data/sr/train_smiles/test_scaffold.csv", index=False)

# 将模型移至GPU（可选）
model2 = model2.to('cuda')
for epoch in range(num_epoch):
    model2.train()
    running_loss = 0.0
    outputs_smiles_epoc = []
    progress_bar = tqdm(zip(train_dataloader1, train_dataloader2), desc=f"Epoch {epoch + 1}/{num_epoch}")

    use_sampling = epoch % 2 == 0  # 每10轮切换一次模式

    for encoder_inputs, decoder_inputs in progress_bar:
        # 随机打乱顺序
        data_size = len(encoder_inputs)
        indices = list(range(data_size))
        random.shuffle(indices)
        if isinstance(encoder_inputs, torch.Tensor):
            encoder_inputs = encoder_inputs[indices]
        else:
            encoder_inputs = [encoder_inputs[i] for i in indices]
        if isinstance(decoder_inputs, torch.Tensor):
            decoder_inputs = decoder_inputs[indices]
        else:
            decoder_inputs = [decoder_inputs[i] for i in indices]

        encoder_inputs = encoder_inputs.to(device)
        decoder_inputs = decoder_inputs.to(device).long()

        optimizer2.zero_grad()

        batch_size = decoder_inputs.size(0)
        max_len = decoder_inputs.size(1)

        decoder_inputs_in = decoder_inputs[:, :-1]  # 去掉末尾 token
        decoder_targets = decoder_inputs[:, 1:]     # 去掉开头 token

        generated = decoder_inputs_in[:, :1]  # 初始仅含 [START]，形状 (B, 1)
        logits_list = []

        for t in range(max_len - 1):  # 因为 decoder_inputs_in 长度是 max_len - 1
            decoder_input_step = generated  # 当前已生成序列
            logits_t, _ = model2(encoder_inputs, decoder_input_step, d_model, None, None)  # logits_t: (B, t+1, V)
            logits_t = logits_t[:, -1, :]  # 取当前 step 的输出 (B, V)
            logits_list.append(logits_t)

            if use_sampling:
                temperature = 1.2
                probs = torch.softmax(logits_t / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                next_token = torch.argmax(logits_t, dim=-1, keepdim=True)  # (B, 1)

            generated = torch.cat([generated, next_token], dim=1)  # 拼接生成的 token

        # 拼接所有 logits: (B, T, V)
        logits_all = torch.stack(logits_list, dim=1)
        target = decoder_targets  # (B, T)

        loss = criterion(logits_all.view(-1, target_vocab_size), target.contiguous().view(-1))
        loss.backward()
        optimizer2.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

        # 保存 SMILES 序列用于最后显示
        with torch.no_grad():
            pred_ids = generated.detach().cpu()
            outputs_smiles = selfies_to_smiles(ids_batch_to_selfies(pred_ids, id2token))
            outputs_smiles_epoc.append(outputs_smiles)

    avg_loss = running_loss
    losses.append(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model2.state_dict(), '/data/sr/train_smiles/97en97de_druglstmformer_mcf7+.pth')
        print(f"✅ Saved best model with loss {best_loss:.4f} at epoch {epoch + 1}")

    if epoch == num_epoch - 1:
        print("\n✅ Final SMILES from last epoch:")
        for smiles_batch in outputs_smiles_epoc:
            for smiles in smiles_batch:
                print(smiles)

# 绘制热力图
# plt.figure(figsize=(20, 20))
# sns.heatmap(visualized_fragments, cmap="viridis", cbar=True, yticklabels=visualized_labels)
# plt.title("Attention Heatmap: Molecular Fragments to Genes (Last Epoch)")
# plt.xlabel("Genes")
# plt.ylabel("Molecular Fragments")
# plt.show()
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, num_epoch + 1), losses, label="Training Loss", color="blue", linestyle="-")
# # plt.plot(range(1, num_epochs2 + 1), losses1, label="Test Loss",  color="red", linestyle="--")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss Curve")
# plt.legend(fontsize=12)
# plt.savefig("loss_curve_comparison.png", dpi=300)
# plt.show()
# model2.load_state_dict(torch.load('/data/sr/train_smiles/gpt_druglstmformer_0.pth'))
model2.load_state_dict(torch.load('/data/sr/train_smiles/97en97de_druglstmformer_mcf7+.pth'))
num_samples = 50
temperature = 1.2
all_predicted_smiles_list = []
with torch.no_grad():
    for encoder_inputs in tqdm(test_dataloader1, desc="Generating Molecules"):
        encoder_inputs = encoder_inputs.to(device)
        batch_size = encoder_inputs.shape[0]
        start_token = token2id["[START]"]

        # 每个样本生成 num_samples 个分子
        predicted_smiles_list = [[] for _ in range(batch_size)]

        for _ in range(num_samples):
            # 初始化生成序列，仅含 [START]，其余为 [PAD]
            generated = torch.full((batch_size, max_len), fill_value=token2id["[PAD]"], dtype=torch.long, device=device)
            generated[:, 0] = start_token

            for t in range(1, max_len):
                # 调用模型预测当前所有位置的 logits（注意这里只看当前 t 前的 token）
                outputs, _ = model2(encoder_inputs, generated, d_model, mask=None)  # (batch, seq, vocab)
                if t >= 2:
                    probs = torch.softmax(outputs[:, t - 1, :] / temperature, dim=-1)  # 当前步的概率

                # 每个样本采样一个 token，填入第 t 个位置
                    sampled_ids = torch.multinomial(probs, num_samples=1).squeeze(1)  # (batch,)
                else:
                    sampled_ids = torch.argmax(outputs[:, t - 1, :], dim=-1)
                generated[:, t] = sampled_ids

            # 转换为 SMILES
            generated_selfies = ids_batch_to_selfies(generated, id2token)
            generated_smiles = selfies_to_smiles(generated_selfies)

            for b in range(batch_size):
                predicted_smiles_list[b].append(generated_smiles[b])

        all_predicted_smiles_list.extend(predicted_smiles_list)
# 保存补全后的预测 SMILES
predicted_df = pd.DataFrame(all_predicted_smiles_list)
predicted_df.to_csv("/data/sr/train_smiles/pred_gptAKT1.csv", index=False, header=False)
all_smiles = []

# 遍历 test_dataloader2
for batch in tqdm(test_dataloader2, desc="Converting SELFIES to SMILES"):
    # 假设 batch 是 token IDs 张量，形状为 (batch_size, seq_len)

    # 1. 将token IDs批量转换为SELFIES字符串
    batch_selfies = ids_batch_to_selfies(batch, id2token)  # 返回一个SELFIES字符串列表

    # 2. 将每个SELFIES转换为SMILES
    batch_smiles = selfies_to_smiles(batch_selfies)

    all_smiles.extend(batch_smiles)

# 保存为CSV
smiles_df = pd.DataFrame(all_smiles, columns=["SMILES"])
smiles_df.to_csv("/data/sr/train_smiles/test_dataloader2_smiles.csv", index=False)
print(f"成功保存 {len(all_smiles)} 个SMILES到CSV文件")