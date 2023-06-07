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
from train import train
import argparse
import os

def parse_args():

    parser = argparse.ArgumentParser(description='Taylor Swift style song lyrics generation LLM')

    parser.add_argument('--model', default='gpt2',
                        help='Specify which pretrained model to use')

    # task specifications
    parser.add_argument('--task', required=True, 
                        help='Training or testing')

    # training specifications
    parser.add_argument('--epoch', type=int, default = 20, 
                        help='How many epochs to train for')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Specify the batch size (default: 256).')
    parser.add_argument('--lr', type=float, default=1e-05,
                        help='Specify initial learning rate (default: 1e-05).')
    
    parser.add_argument('--save-model', action='store_true', help='Model save path')
    parser.add_argument('--save-path', type=str, default='checkpoints/',
                        help='Model save path')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Model gets saved after how many epochs')
                        
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    args = parse_args()

    data = pd.read_csv('dump.csv')
    print(data.shape)

    test_set = data.sample(n = 20)
    data = data.loc[~data.index.isin(test_set.index)]

    test_set = test_set.reset_index()
    data = data.reset_index()

    dataset = SongLyrics(data, truncate=True, gpt2_type="gpt2")

    print(dataset)

    #Train the model on the specific data we have
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    if args.save_model and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    _model = train(dataset, model, tokenizer, args)

    # torch.save(_model, 'model.pt')
    pass
