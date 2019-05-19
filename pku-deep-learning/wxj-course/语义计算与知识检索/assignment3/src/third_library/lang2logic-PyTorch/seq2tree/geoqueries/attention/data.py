import time
import pickle as pkl
import torch
from util import SymbolsManager
from sys import path
import argparse
import random
import numpy as np
from util import convert_to_tree




def test_vocab(opt):
    word_manager = SymbolsManager(True)
    word_manager.init_from_file(
        "{}/train_vocab_q.txt".format(opt.data_dir), opt.min_freq, opt.max_vocab_size)
    #word_manager.init_from_file("{}/vocab.q.txt".format(opt.data_dir), opt.min_freq, opt.max_vocab_size)
    form_manager = SymbolsManager(True)
    form_manager.init_from_file(
        "{}/train_vocab_f.txt".format(opt.data_dir), 0, opt.max_vocab_size)
    #form_manager.init_from_file("{}/vocab.f.txt".format(opt.data_dir), 0, opt.max_vocab_size)
    print(word_manager.vocab_size)
    print(form_manager.vocab_size)
    print(form_manager.get_symbol_idx('chris_pine'))
    try:
        print(form_manager.get_symbol_idx('the_walking_dead_theme_song'))
    except:
        print('Not found')
    word_token_set = set()
    form_token_set = set()
    with open('../../../../../../data/MSParS.entity.train','r',encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            words,forms = line[0].split(' '), line[1].split(' ')
            for word in words:
                word_token_set.add(word)
            for form in forms:
                form_token_set.add(form)
    print('word token num:', len(word_token_set))
    print('form token num:', len(form_token_set))


def process_train_data(opt):
    time_start = time.time()
    word_manager = SymbolsManager(True)
    word_manager.init_from_file(
        "{}/train_vocab_q.txt".format(opt.data_dir), opt.min_freq, opt.max_vocab_size)
    #word_manager.init_from_file("{}/vocab.q.txt".format(opt.data_dir), opt.min_freq, opt.max_vocab_size)
    form_manager = SymbolsManager(True)
    form_manager.init_from_file(
        "{}/train_vocab_f.txt".format(opt.data_dir), 0, opt.max_vocab_size)
    #form_manager.init_from_file("{}/vocab.f.txt".format(opt.data_dir), 0, opt.max_vocab_size)
    print(word_manager.vocab_size)
    print(form_manager.vocab_size)
    data = []
    with open("{}/{}".format(opt.data_dir, opt.train), "r") as f:
        for line in f:
            l_list = line.split("\t")
            w_list = word_manager.get_symbol_idx_for_list(l_list[0].strip().split(' '))
            r_list = form_manager.get_symbol_idx_for_list(l_list[1].strip().split(' '))
            cur_tree = convert_to_tree(r_list, 0, len(r_list), form_manager)
            #print(w_list)
            #print(r_list)
            #print(cur_tree.to_string())
            data.append((w_list, r_list, cur_tree))
    out_mapfile = "{}/map.pkl".format(opt.data_dir)
    out_datafile = "{}/train.pkl".format(opt.data_dir)
    with open(out_mapfile, "wb") as out_map:
        pkl.dump([word_manager, form_manager], out_map)
    with open(out_datafile, "wb") as out_data:
        pkl.dump(data, out_data)

def serialize_data(opt):
    data = []
    managers = pkl.load( open("{}/map.pkl".format(opt.data_dir), "rb" ) )
    word_manager, form_manager = managers
    with open("{}/{}".format(opt.data_dir, opt.test), "r") as f:
        for line in f:
            l_list = line.split("\t")
            w_list = word_manager.get_symbol_idx_for_list(l_list[0].strip().split(' '))
            r_list = form_manager.get_symbol_idx_for_list(l_list[1].strip().split(' '))
            cur_tree = convert_to_tree(r_list, 0, len(r_list), form_manager)
            data.append((w_list, r_list, cur_tree))
    out_datafile = "{}/test.pkl".format(opt.data_dir)
    with open(out_datafile, "wb") as out_data:
        pkl.dump(data, out_data)
    
   
if __name__ == "__main__":   
    main_arg_parser = argparse.ArgumentParser(description="parser")
    main_arg_parser.add_argument("-data_dir", type=str, default="../data/",
                                    help="data dir")
    main_arg_parser.add_argument("-train", type=str, default="train",
                                    help="train dir")
    main_arg_parser.add_argument("-test", type=str, default="test",
                                    help="test dir")
    main_arg_parser.add_argument("-min_freq", type=int, default=0,
                                    help="minimum word frequency")
    main_arg_parser.add_argument("-max_vocab_size", type=int, default=50000,
                                    help="max vocab size")
    main_arg_parser.add_argument('-seed',type=int,default=123,help='torch manual random number generator seed')

    args = main_arg_parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    process_train_data(args)
    serialize_data(args)
    # test_vocab(args)
