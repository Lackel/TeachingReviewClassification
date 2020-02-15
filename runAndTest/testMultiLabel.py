#   _*_coding:utf-8 _*_
#   author:wenbinan
#   time:2020/2/14 19:09
#   filename:testMultiLabel.py
#   product:PyCharm

import time
import torch
import numpy as np
from utils.utils_multiLabel.train_eval import train, init_network
from importlib import import_module
import argparse
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import os


os.chdir("..")   # 修改当前工作目录
model_name = "TextCNN"
dataset = 'data/multiLabel'
path_state_dict = "./data/multiLabel/saved_dict/TextCNN.ckpt"
embedding = 'embedding_SougouNews.npz'
state_dict_load = torch.load(path_state_dict)

if model_name == 'FastText':
    from utils.utils_multiLabel.utils_fasttext import build_dataset, build_iterator, get_time_dif

    embedding = 'random'
else:
    from utils.utils_multiLabel.utils import build_dataset, build_iterator, get_time_dif

x = import_module('models.multiLabel.' + model_name)
config = x.Config(dataset, embedding)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

vocab, train_data, dev_data, test_data = build_dataset(config, False)
config.n_vocab = len(vocab)
test_iter = build_iterator(test_data, config)
model = x.Model(config).to(config.device)

model.load_state_dict(state_dict_load)

# MultiLabel
for texts, labels in test_iter:
    outputs = model(texts)
    predic = (outputs.data > 0.5).cpu().numpy()
    print(predic)


