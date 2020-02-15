# coding: UTF-8
import time
import torch
import numpy as np
from utils.utils_multiClass.train_eval import train, init_network
from importlib import import_module
import argparse
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import os


if __name__ == '__main__':
    os.chdir("..")  # 修改当前工作目录
    model_name = "TextCNN"  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    dataset = 'data/multiClass'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if model_name == 'FastText':
        from utils.utils_multiClass.utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils.utils_multiClass.utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.multiClass.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, False)  # True for word, False for char
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    # trains, labels = next(iter(train_iter))
    # with SummaryWriter(comment='model') as w:
    #     w.add_graph(model, (trains, ))
    train(config, model, train_iter, dev_iter, test_iter)
