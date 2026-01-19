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
token2id_path = "/data/sr/train_smiles/token2id.json"
moses_train = np.load("/data/sr/train_smiles/moses_train.npy",
                      mmap_mode="r")
# moses_test = np.load("/data/sr/train_smiles/moses_test.npy",
#                      mmap_mode="r")
moses_test = np.load("/data/sr/train_smiles/selfies_tokenized_indexed74740.npy",
                     mmap_mode="r")
drug = torch.from_numpy(moses_train).long()
drug = MyDataset2(drug)
test = torch.from_numpy(moses_test).long()
test = MyDataset2(test)
train_dataloader0 = DataLoader(drug, batch_size=32, shuffle=False, drop_last=False)
test_dataloader0 = DataLoader(test, batch_size=32, shuffle=False, drop_last=False)
# 加载 token -> id 映射
with open(token2id_path, "r", encoding="utf-8") as f:
    token2id = json.load(f)
# 构建 id -> token 映射
id2token = {v: k for k, v in token2id.items()}
# gene_data = pd.read_csv('/data/sr/foldchange_cellline.csv').values
# cell_label = gene_data[:, 1]
# gene_data = gene_data[:, 3:]
# cell_label = np.array(cell_label, dtype=np.float32)
# gene_data = np.array(gene_data, dtype=np.float32)
# num_samples = gene_data.shape[0]
# num_select = 10_000
#
# assert num_samples >= num_select, "基因样本数不足 10k"
#
# indices = np.random.choice(num_samples, size=num_select, replace=False)
#
# gene_data_10k = gene_data[indices]        # [10000, G]
# cell_label_10k = cell_label[indices]      # [10000]
# gene_data_10k = torch.tensor(
#     gene_data_10k,
#     dtype=torch.float32,
#     device=device
# )  # [10000, G]

scaffold_layers = 4
full_layers = 2
max_len = 97
d_model = 256
num_heads = 8  # 这里的 num_heads 一般是 8 而不是 64，因为 64 可能是维度大小错误
d_ff = 1024
input_vocab_size = 978
target_vocab_size = len(token2id)
model2 = TransLSTMEncoderDecoder(scaffold_layers, full_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_len, dropout=0.1).to(device)
conv_block = FullyConnectedCompressBlock(input_dim=978, seq_len=1, d_model=d_model).to(device)
start_token = token2id["[START]"]
end_token = token2id["[END]"]
pad_token = token2id["[PAD]"]
# ckpt_path = "/data/sr/train_smiles/full_layers_gelu_druglstmformer.pth"
# ckpt_path = "/data/sr/train_smiles/full_layers_gelu_druglstmformer.pth"
# ckpt_path1 = "/data/sr/train_smiles/gene_block.pth"
# state_dict = torch.load(ckpt_path, map_location=device)
# state_dict1 = torch.load(ckpt_path1, map_location=device)
# model2.load_state_dict(state_dict, strict=False)
# conv_block.load_state_dict(state_dict1, strict=False)

ckpt_path = "/data/sr/train_smiles/moses_train_druglstmformer.pth"
state_dict = torch.load(ckpt_path, map_location=device)
model2.load_state_dict(state_dict, strict=False)
outputs_smiles_epoc = []
# for param in model2.parameters():
#     param.requires_grad = True
for param in model2.shared_embedding.parameters():
    param.requires_grad = False
# num_generate_per_round = 1000
# num_rounds = 10
# max_len = max_len  # 你已有
# all_generated_smiles = []
criterion = nn.CrossEntropyLoss()
# ===== 4. 构建优化器 =====
optimizer2 = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model2.parameters()),
    lr=1e-4
)
num_epoch = 5
best_loss0 = float('inf')
# best_loss0 = 9728.0
# for epoch in range(num_epoch):
#     model2.train()
#     running_loss = 0.0
#     num_batches = 0
#     outputs_smiles_epoc = []
#     progress_bar = tqdm(test_dataloader0, desc=f"Epoch {epoch + 1}/{num_epoch}")
#     total_batches = len(progress_bar)
#     for batch_idx, decoder_inputs in enumerate(progress_bar):
#         num_batches += 1
#         data_size = len(decoder_inputs)  # 数据大小
#         batch_size = decoder_inputs.shape[0]
#         # ===== 自回归准备：错位 =====
#         decoder_inputs_in = decoder_inputs[:, :-1].to(device)  # 去掉最后一个 token，用作 decoder 输入
#         decoder_targets = decoder_inputs[:, 1:].to(device)  # 去掉第一个 token，用作目标输出
#
#         seq_len = decoder_inputs_in.size(1)
#         mask = create_look_ahead_mask(seq_len, device=decoder_inputs_in.device)
#         mask = mask.unsqueeze(0).expand(decoder_inputs_in.shape[0], -1, -1)
#         # zero_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=decoder_inputs.device)
#         output, _ = model2.scaffold(decoder_inputs_in, None, mask)
#         # output, _ = model2.decoder(model2.norm(output), None, zero_mask)
#         output = model2.fc_out(model2.norm(output))
#         decoder_targets = decoder_targets.contiguous().view(-1)  # (B * L)
#         outputs = output.view(-1, target_vocab_size)  # (B * L, vocab)
#
#         loss = criterion(outputs, decoder_targets)
#         optimizer2.zero_grad()
#         loss.backward()
#         optimizer2.step()
#         running_loss += loss.item()
#         progress_bar.set_postfix(loss=loss.item())
#     if running_loss < best_loss0:
#         best_loss0 = running_loss
#         torch.save(model2.state_dict(), '/data/sr/train_smiles/moses_train_druglstmformer_ultimate.pth')
#         print(f'Best model saved at epoch {epoch + 1} with Test Loss: {best_loss0}')

        # generated_selfies = torch.argmax(output, dim=-1)
        # generated_selfies = ids_batch_to_selfies(generated_selfies, id2token)
        # outputs_smiles = selfies_to_smiles(generated_selfies)
        # outputs_smiles_epoc.append(outputs_smiles)
# with torch.no_grad():
#     for r in range(num_rounds):
#         print(f"🚀 Generating round {r+1}/{num_rounds}")
#
#         batch_size = num_generate_per_round
#         start = r * num_generate_per_round
#         end = (r + 1) * num_generate_per_round
#
#         gene_batch = gene_data_10k[start:end]
#         encoder_inputs = conv_block(gene_batch)
#         # 建议 enc_output.shape = [B, 1, D] 或 [B, L, D]
#
#         generated = torch.full(
#                 (batch_size, 1),
#                 start_token,
#                 dtype=torch.long,
#                 device=device
#             )
#
#         for _ in range(max_len - 1):
#             logits, _ = model2.scaffold(generated, encoder_inputs, None)
#             logits, _ = model2.decoder(model2.norm(logits), encoder_inputs, None)
#             logits = model2.fc_out(model2.norm(logits))
#
#             next_token_logits = logits[:, -1, :]
#             # next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
#             probs = torch.softmax(next_token_logits / 1.0, dim=-1)
#             next_token = torch.multinomial(probs, num_samples=1)
#
#             generated = torch.cat([generated, next_token], dim=1)
#
#             # 如果整个 batch 都结束了，可以提前 break
#             if (next_token == end_token).all():
#                 break
#
#         # ===== 转成 SMILES =====
#         generated_selfies = ids_batch_to_selfies(generated, id2token)
#         generated_smiles = selfies_to_smiles(generated_selfies)
#
#         all_generated_smiles.extend(generated_smiles)
#
# # ===== 保存 =====
# df = pd.DataFrame({"smiles": all_generated_smiles})
# df.to_csv("/data/sr/train_smiles/moses_@withgene10k.csv", index=False)
#
# print(f"🎉 Generated {len(all_generated_smiles)} molecules → moses_@10k.csv")
ckpt_path = "/data/sr/train_smiles/moses_train_druglstmformer.pth"
state_dict = torch.load(ckpt_path, map_location=device)
model2.load_state_dict(state_dict, strict=False)

all_generated_selfies = []
all_generated_smiles = []

with torch.no_grad():
    model2.eval()
    for batch_idx, decoder_inputs in enumerate(
        tqdm(test_dataloader0, desc="Masked Decoding (Non-AR)")
    ):
        """
        decoder_inputs:
            shape = [B, L]
            含 [START] ... [END] [PAD]
        """

        decoder_inputs = decoder_inputs.to(device)
        batch_size = decoder_inputs.size(0)

        # =========================
        # 1. 和训练一致的错位
        # =========================
        decoder_inputs_in = decoder_inputs[:, :-1]   # [B, L-1]

        seq_len = decoder_inputs_in.size(1)
        zero_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=decoder_inputs.device)
        # mask = create_look_ahead_mask(seq_len, device=decoder_inputs_in.device)
        # mask = mask.unsqueeze(0).expand(decoder_inputs_in.shape[0], -1, -1)
        # --- 应用到你的代码逻辑中 ---
        prefix_len = 15  # 比如你想让模型看到前 5 个字符
        seq_len = decoder_inputs_in.shape[1]
        device = decoder_inputs_in.device

        mask = create_prefix_look_ahead_mask(seq_len, prefix_len, device)
        mask = mask.unsqueeze(0).expand(decoder_inputs_in.shape[0], -1, -1)
        # =========================
        # 3. forward（完全对齐训练）
        # =========================
        output, _ = model2.scaffold(
            decoder_inputs_in,
            None,
            mask
        )
        # output, _ = model2.decoder(
        #     model2.norm(output),
        #     None,
        #     zero_mask
        # )
        logits = model2.fc_out(model2.norm(output))  # [B, L-1, vocab]

        # =========================
        # 4. 直接取 argmax（非自回归）
        # =========================
        pred_ids = torch.argmax(logits, dim=-1)  # [B, L-1]

        # =========================
        # 5. 拼回 START（方便解析）
        # =========================
        pred_full = torch.cat(
            [
                decoder_inputs[:, :1],  # 原始 START
                pred_ids
            ],
            dim=1
        )  # [B, L]

        # =========================
        # 6. ids → selfies → smiles
        # =========================
        generated_selfies = ids_batch_to_selfies(pred_full, id2token)
        generated_smiles = selfies_to_smiles(generated_selfies)

        all_generated_selfies.extend(generated_selfies)
        all_generated_smiles.extend(generated_smiles)

# =========================
# 7. 保存 CSV
# =========================
df = pd.DataFrame({
    "selfies": all_generated_selfies,
    "smiles": all_generated_smiles
})

save_path = "/data/sr/train_smiles/generated_masked_parallel.csv"
df.to_csv(save_path, index=False)

print(f"🎉 Generated {len(df)} molecules (non-autoregressive)")
print(f"📁 Saved to {save_path}")