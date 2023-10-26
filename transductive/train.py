import random
import os
import argparse
import torch
import numpy as np
from load_data import DataLoader
from base_model import BaseModel
from utils import *

''' main script of AdaProp'''
parser = argparse.ArgumentParser(description="Parser for AdaProp")
parser.add_argument('--data_path', type=str, default='data/family/')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--topk', type=int, default=-1)
parser.add_argument('--layers', type=int, default=-1)
parser.add_argument('--sampling', type=str, default='incremental')
parser.add_argument('--weight', type=str, default=None)
parser.add_argument('--tau', type=float, default=1.0)
parser.add_argument('--loss_in_each_layer', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--HPO', action='store_true')
parser.add_argument('--eval_with_node_usage', action='store_true')
parser.add_argument('--scheduler', type=str, default='exp')
parser.add_argument('--remove_1hop_edges', action='store_true')
parser.add_argument('--fact_ratio', type=float, default=0.9)
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--eval_interval', type=int, default=1)
args = parser.parse_args()

if __name__ == '__main__':
    opts = args
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    
    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]
    
    torch.cuda.set_device(opts.gpu)
    print('==> gpu:', opts.gpu)
    loader = DataLoader(opts)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel
    
    if dataset == 'family':
        opts.lr = 0.0036
        opts.decay_rate = 0.999
        opts.lamb = 0.000017
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.dropout = 0.29
        opts.act = 'relu'
        opts.n_node_topk = [opts.topk] * opts.layers
        opts.n_edge_topk = -1
        opts.n_layer = opts.layers
        opts.n_batch = opts.n_tbatch = 20
        
    elif dataset == 'umls':
        opts.lr = 0.0012
        opts.decay_rate = 0.998
        opts.lamb = 0.00014
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.dropout = 0.01
        opts.act = 'tanh'
        opts.n_node_topk = [opts.topk] * opts.layers
        opts.n_edge_topk = -1
        opts.n_layer = opts.layers
        opts.n_batch = opts.n_tbatch = 10
        
    elif dataset == 'WN18RR':
        opts.lr = 0.0030
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.n_node_topk = [opts.topk] * opts.layers
        opts.n_edge_topk = -1
        opts.n_layer = opts.layers
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = opts.n_tbatch = 50
        
    elif dataset == 'fb15k-237':
        opts.lr = 0.0009
        opts.decay_rate = 0.9938
        opts.lamb = 0.000080
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_node_topk = [opts.topk] * opts.layers
        opts.n_edge_topk = -1
        opts.n_layer = opts.layers
        opts.dropout = 0.0391
        opts.act = 'idd'
        opts.n_batch = opts.n_tbatch = 10
        
    elif dataset == 'nell':
        opts.lr = 0.0011
        opts.decay_rate = 0.9938
        opts.lamb = 0.000089
        opts.hidden_dim = 128
        opts.attn_dim = 64
        opts.dropout = 0.2593
        opts.act = 'idd'
        opts.n_node_topk = [opts.topk] * opts.layers
        opts.n_edge_topk = -1
        opts.n_layer = opts.layers
        opts.n_batch = opts.n_tbatch = 10
        
    elif dataset == 'YAGO':
        opts.lr = 0.001
        opts.decay_rate = 0.9429713470775948
        opts.lamb = 0.000946516892415447
        opts.hidden_dim = 64
        opts.attn_dim = 2
        opts.dropout = 0.19456805575101324
        opts.act = 'relu'
        opts.n_node_topk = [opts.topk] * opts.layers
        opts.n_edge_topk = -1
        opts.n_layer = opts.layers
        opts.n_batch = opts.n_tbatch = 5
    
    # check all output paths
    checkPath('./results/')
    checkPath(f'./results/{dataset}/')
    checkPath(f'{loader.task_dir}/saveModel/')

    model = BaseModel(opts, loader)
    opts.perf_file = f'results/{dataset}/{model.modelName}_perf.txt'
    print(f'==> perf_file: {opts.perf_file}')
    
    config_str = '%.4f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s\n' % (opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act)
    print(config_str)
    with open(opts.perf_file, 'a+') as f:
        f.write(config_str)  

    if args.weight != None:
        model.loadModel(args.weight)
        model._update()
        model.model.updateTopkNums(opts.n_node_topk)

    if opts.train:
        # training mode
        best_v_mrr = 0
        for epoch in range(opts.epoch):
            model.train_batch()
            # eval on val/test set
            if (epoch+1) % args.eval_interval == 0:
                result_dict, out_str = model.evaluate(eval_val=True, eval_test=True)
                v_mrr, t_mrr = result_dict['v_mrr'], result_dict['t_mrr']
                print(out_str)
                with open(opts.perf_file, 'a+') as f:
                    f.write(out_str)
                if v_mrr > best_v_mrr:
                    best_v_mrr = v_mrr
                    best_str = out_str
                    print(str(epoch) + '\t' + best_str)
                    BestMetricStr = f'ValMRR_{str(v_mrr)[:5]}_TestMRR_{str(t_mrr)[:5]}'
                    model.saveModelToFiles(BestMetricStr, deleteLastFile=False)
        
        # show the final result
        print(best_str)
        
    if opts.eval:
        # evaluate on test set with loaded weight to save time
        result_dict, out_str = model.evaluate(eval_val=False, eval_test=True, verbose=True)
        print(result_dict, '\n', out_str)
        
