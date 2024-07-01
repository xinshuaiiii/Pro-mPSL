import argparse
from transformers import T5Tokenizer, T5EncoderModel
import torch
import numpy as np
from tqdm import tqdm
import re
import torch.nn as nn

parser = argparse.ArgumentParser(description='Generate protein embeddings.')
parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model.')
parser.add_argument('--filename', type=str, required=True, help='Path to the protein sequence file.')
parser.add_argument('--output_path', type=str, required=True, help='Path to save the output embeddings.')
args = parser.parse_args()

batch_size = 10
model_path = args.model_path
filename = args.filename
output_path = args.output_path
all_embeddings = []
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)

model = T5EncoderModel.from_pretrained(model_path).to(device)

model.to(torch.float32) if device == torch.device("cpu") else None

protein_seq = []
with open(filename) as fin:
    for line in fin:
        if not line.startswith('>'):
            protein_seq.append(line.strip()) 

sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in protein_seq]

with tqdm(total=len(sequence_examples), desc="Processing batches") as pbar:
    for i in range(0, len(sequence_examples), batch_size):
        batch_sequences = sequence_examples[i:i + batch_size]

        encoding = tokenizer.batch_encode_plus(
            batch_sequences,
            add_special_tokens=True,
            padding='longest',
            return_tensors='pt',
            truncation=True,
            max_length=1000  
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            embeddings = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

            per_residue_embeddings = [
                embedding[:sum(attention_mask[j])]
                for j, embedding in enumerate(embeddings)
            ]

            per_protein_embeddings = [
                embedding.mean(dim=0)
                for embedding in per_residue_embeddings
            ]
            all_embeddings.extend(per_protein_embeddings)

            pbar.update(len(batch_sequences))

all_embeddings_numpy = torch.stack(all_embeddings).cpu().numpy()

np.save(output_path, all_embeddings_numpy)
