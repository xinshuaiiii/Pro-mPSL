from transformers import T5Tokenizer, T5EncoderModel
import torch
import numpy as np
from tqdm import tqdm
import re
import torch.nn as nn

batch_size = 10
model_path = "../prot5"
filename = "/home/zhaozhimiao/xs/pythonproject/train/17_mutated_sequence_S553P.txt"
all_embeddings = []
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# 加载本地的分词器
tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)

# 加载本地的模型
model = T5EncoderModel.from_pretrained(model_path).to(device)

# 设置模型为float32
model.to(torch.float32) if device == torch.device("cpu") else None

protein_seq = []
with open(filename) as fin:
    for line in fin:
        if not line.startswith('>'):
            protein_seq.append(line.strip())  # 修改此行

# 替换稀有/模糊的氨基酸并分词
sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in protein_seq]

# 使用tqdm来显示进度
with tqdm(total=len(sequence_examples), desc="Processing batches") as pbar:
    for i in range(0, len(sequence_examples), batch_size):
        batch_sequences = sequence_examples[i:i + batch_size]

        # 对批次进行编码
        encoding = tokenizer.batch_encode_plus(
            batch_sequences,
            add_special_tokens=True,
            padding='longest',
            return_tensors='pt',
            truncation=True,
            max_length=1000  # 可选：如果需要，设置最大序列长度
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # 生成嵌入
        with torch.no_grad():
            embeddings = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

            # 提取每个序列的残基嵌入
            per_residue_embeddings = [
                embedding[:sum(attention_mask[j])]
                for j, embedding in enumerate(embeddings)
            ]

            # 如果你想要为每个蛋白质生成一个单一的表示
            per_protein_embeddings = [
                embedding.mean(dim=0)
                for embedding in per_residue_embeddings
            ]
            all_embeddings.extend(per_protein_embeddings)
            # 更新进度条
            pbar.update(len(batch_sequences))

# 转换为numpy数组
all_embeddings_numpy = torch.stack(all_embeddings).cpu().numpy()

# 保存所有嵌入到一个文件中
np.save('/home/zhaozhimiao/xs/pythonproject/code1/test3.npy', all_embeddings_numpy)
