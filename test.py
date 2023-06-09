import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from config import CHECKPOINT_PATH, SOURCE_PATH, DUMP_PATH

def generate(model, tokenizer, prompt, entry_count=10, entry_length=300, top_p=0.8, temperature=1.,):
    model.eval()

    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():

        for entry_idx in trange(entry_count):

            entry_finished = False

            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            for i in range(entry_length):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                if next_token in tokenizer.encode("<|endoftext|>"):
                    entry_finished = True

                if entry_finished:

                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    generated_list.append(output_text)
                    break
            
            if not entry_finished:
              output_list = list(generated.squeeze().numpy())
              output_text = f"{tokenizer.decode(output_list)}<|endoftext|>" 
              generated_list.append(output_text)
                
    return generated_list

model = GPT2LMHeadModel.from_pretrained('gpt2')
weights = torch.load(CHECKPOINT_PATH)
# weights = torch.load('/home/bigthinx1/research/taylorllm/wreckgar-49.pt')
model.load_state_dict(weights)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

with open(SOURCE_PATH, "r") as fp:
    source_text = fp.read()
prompt = "<|startoftext|> " + source_text

text = generate(model.to('cpu'), tokenizer, prompt, entry_count=1)

# print(text)
with open(DUMP_PATH, 'w') as fp:
    fp.write(text[0])


