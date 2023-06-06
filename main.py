import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv
from dataset import SongLyrics
from utils import train

if __name__ == "__main__":
    data = pd.read_csv('dump.csv')
    print(data.shape)

    test_set = data.sample(n = 100)
    data = data.loc[~data.index.isin(test_set.index)]

    test_set = test_set.reset_index()
    data = data.reset_index()

    dataset = SongLyrics(data, truncate=True, gpt2_type="gpt2")

    print(dataset)

    #Train the model on the specific data we have
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    _model = train(dataset, model, tokenizer)
    torch.save(_model, 'model.pt')
    pass
