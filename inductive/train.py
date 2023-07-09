import argparse
import torch
import numpy as np
from load_data import DataLoader
from base_model import BaseModel
from utils import select_gpu

parser = argparse.ArgumentParser(description="Parser for AdaProp")
parser.add_argument('--data_path', type=str, default='./data/fb237_v1')
parser.add_argument('--seed', type=str, default=1234)
args = parser.parse_args()

class Options(object):
    pass

dataset = args.data_path
dataset = dataset.split('/')
if len(dataset[-1]) > 0:
    dataset = dataset[-1]
else:
    dataset = dataset[-2]

opts = Options
opts.hidden_dim = 64
opts.init_dim = 10
opts.attn_dim = 5
opts.n_layer = 3
opts.n_batch = 50
opts.lr = 0.001
opts.decay_rate = 0.999
opts.perf_file = './results.txt'
gpu = select_gpu()
torch.cuda.set_device(gpu)
print('==> selected GPU id: ', gpu)

loader = DataLoader(args.data_path, n_batch=opts.n_batch)
opts.n_ent = loader.n_ent
opts.n_rel = loader.n_rel

def run_model(params):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    opts.lr = params['lr']
    opts.lamb = params["lamb"]
    opts.decay_rate = params['decay_rate']
    opts.hidden_dim = params['hidden_dim']
    opts.init_dim = params['hidden_dim']
    opts.attn_dim = params['attn_dim']
    opts.dropout = params['dropout']
    opts.act = params['act']
    opts.n_layer = params['n_layer']
    opts.n_batch = params['n_batch']
    opts.topk = params['topk']
    opts.increase = params['increase']

    config_str = '%.4f, %.4f, %.6f,  %d, %d, %d, %d, %d, %.4f, %s  %d, %s\n' % (opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.init_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act, opts.topk, str(opts.increase))
    print(args.data_path)
    print(config_str)
    
    try:
        model = BaseModel(opts, loader)
        best_mrr = 0
        best_tmrr = 0
        early_stop = 0
        for epoch in range(30):
            mrr, t_mrr, out_str = model.train_batch()
            if mrr > best_mrr:
                best_mrr = mrr
                best_tmrr = t_mrr
                best_str = out_str
                early_stop = 0
            else:
                early_stop += 1

        with open(opts.perf_file, 'a') as f:
            f.write(args.data_path + '\n')
            f.write(config_str)
            f.write(best_str + '\n')
            print('\n\n')
            
    except RuntimeError:
        best_tmrr = 0
    
    print('self.time_1, self.time_2, time_3, v_mrr, v_mr, v_h1, v_h3, v_h10, v_h1050, t_mrr, t_mr, t_h1, t_h3, t_h10, t_h1050')
    print(best_str)
    return


params = {}
if 'WN18RR_v1' in args.data_path:
    params['lr'], params['decay_rate'], params["lamb"], params['hidden_dim'],  params['init_dim'], params['attn_dim'], params['n_layer'], params['n_batch'], params['dropout'], params['act'],  params['topk'], params['increase'] = 0.0058, 0.9912, 0.000023,  48, 48, 3, 4, 50, 0.4770, 'idd', 200, True

if 'WN18RR_v2' in args.data_path:
    params['lr'], params['decay_rate'], params["lamb"], params['hidden_dim'],  params['init_dim'], params['attn_dim'], params['n_layer'], params['n_batch'], params['dropout'], params['act'],  params['topk'], params['increase'] = 0.0021, 0.9968, 0.000018,  64, 64, 3, 7, 20, 0.4237, 'relu', 100, True

if 'WN18RR_v3' in args.data_path:
    params['lr'], params['decay_rate'], params["lamb"], params['hidden_dim'],  params['init_dim'], params['attn_dim'], params['n_layer'], params['n_batch'], params['dropout'], params['act'],  params['topk'], params['increase'] = 0.0057, 0.9940, 0.000035,  48, 48, 5, 7, 50, 0.2335, 'relu', 100, True

if 'WN18RR_v4' in args.data_path:
    params['lr'], params['decay_rate'], params["lamb"], params['hidden_dim'],  params['init_dim'], params['attn_dim'], params['n_layer'], params['n_batch'], params['dropout'], params['act'],  params['topk'], params['increase'] = 0.0063, 0.9970, 0.000414,  64, 64, 3, 6, 50, 0.4571, 'idd', 200, True

if 'fb237_v1' in args.data_path:
    params['lr'], params['decay_rate'], params["lamb"], params['hidden_dim'],  params['init_dim'], params['attn_dim'], params['n_layer'], params['n_batch'], params['dropout'], params['act'],  params['topk'], params['increase'] = 0.0005, 0.9968, 0.000081,  32, 32, 5, 3, 100, 0.4137, 'relu', 100, True

if 'fb237_v2' in args.data_path:
    params['lr'], params['decay_rate'], params["lamb"], params['hidden_dim'],  params['init_dim'], params['attn_dim'], params['n_layer'], params['n_batch'], params['dropout'], params['act'],  params['topk'], params['increase'] = 0.0087, 0.9937, 0.000025,  16, 16, 5, 5, 20, 0.3265, 'relu', 200, True

if 'fb237_v3' in args.data_path:
    params['lr'], params['decay_rate'], params["lamb"], params['hidden_dim'],  params['init_dim'], params['attn_dim'], params['n_layer'], params['n_batch'], params['dropout'], params['act'],  params['topk'], params['increase'] = 0.0079, 0.9934, 0.000187,  48, 48, 5, 7, 20, 0.4632, 'relu', 200, True

if 'fb237_v4' in args.data_path:
    params['lr'], params['decay_rate'], params["lamb"], params['hidden_dim'],  params['init_dim'], params['attn_dim'], params['n_layer'], params['n_batch'], params['dropout'], params['act'],  params['topk'], params['increase'] = 0.0010, 0.9997, 0.000186,  16, 16, 5, 7, 50, 0.4793, 'relu', 500, True

if 'nell_v1' in args.data_path:
    params['lr'], params['decay_rate'], params["lamb"], params['hidden_dim'],  params['init_dim'], params['attn_dim'], params['n_layer'], params['n_batch'], params['dropout'], params['act'],  params['topk'], params['increase'] = 0.0046, 0.9902, 0.000220,  32, 32, 5, 6, 20, 0.3268, 'relu', 200, True

if 'nell_v2' in args.data_path:
    params['lr'], params['decay_rate'], params["lamb"], params['hidden_dim'],  params['init_dim'], params['attn_dim'], params['n_layer'], params['n_batch'], params['dropout'], params['act'],  params['topk'], params['increase'] = 0.0049, 0.9948, 0.000072,  16, 16, 5, 3, 50, 0.3247, 'relu', 200, True

if 'nell_v3' in args.data_path:
    params['lr'], params['decay_rate'], params["lamb"], params['hidden_dim'],  params['init_dim'], params['attn_dim'], params['n_layer'], params['n_batch'], params['dropout'], params['act'],  params['topk'], params['increase'] = 0.0090, 0.9986, 0.000298,  16, 16, 3, 8, 20, 0.1336, 'relu', 500, True

if 'nell_v4' in args.data_path:
    params['lr'], params['decay_rate'], params["lamb"], params['hidden_dim'],  params['init_dim'], params['attn_dim'], params['n_layer'], params['n_batch'], params['dropout'], params['act'],  params['topk'], params['increase'] = 0.0006, 0.9958, 0.000060,  16, 16, 5, 3, 50, 0.2411, 'relu', 100, True
    
print(params)
run_model(params)
