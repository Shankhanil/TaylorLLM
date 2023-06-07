from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

class SongLyrics(Dataset):
    
    def __init__(self, df, control_code='<|startoftext|>', truncate=False, gpt2_type="gpt2", max_length=1024):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.lyrics = []

        for index, row in df.iterrows():
            # print(row.file)
            # exit()
            with open(row.file , 'r') as fp:
                text = fp.read()
            self.lyrics.append(torch.tensor(
                # self.tokenizer.encode(f"<|{control_code}|>{text[:max_length]}<|endoftext|>")
                self.tokenizer.encode(f"<|{control_code}|>{text}<|endoftext|>")[:max_length]
            ))
            # print(text[:max_length])
            # print(len(self.lyrics[0]))
            # exit()
        if truncate:
            self.lyrics = self.lyrics[:20000]
        self.lyrics_count = len(self.lyrics)
        
    def __len__(self):
        return self.lyrics_count

    def __getitem__(self, item):
        return self.lyrics[item]