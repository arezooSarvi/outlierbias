import metrics

import pickle
import time
import os
import numpy as np
import json

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from allrank.lambdaLoss import lambdaLoss as lambdaLoss


from correction import *
from allrank.model import *
from data_util import outlier2group


def get_torch_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def log_epoch(epoch, epochs):
    if epoch < 0:
        return False
    if epoch == epochs - 1 or epoch < 5:
        return True
    steps = [[100,5],[500,10],[1000,20],[10000,50]]
    for step, div in steps:
        if epoch < step and (epoch+1) % div == 0:
            return True
    return False

class LTRData(Dataset):
    def __init__(self, fm, dlr):
        
        self.fm_by_qid = np.split(fm, dlr[1:-1])
        self.predicted = [np.ones(dlr[qid+1] - dlr[qid])*0.5 for qid in range(dlr.shape[0] - 1)]
        self.lv = [None for _ in range(dlr.shape[0] - 1)]
        self.dev = get_torch_device()
        if torch.cuda.is_available():
            self.torch_ = torch.cuda
        else:
            self.torch_ = torch
            
    def update_labels(self, labels, qids):
        for qid, label in zip(qids, labels):
            self.lv[qid] = label[:]
        
    def update_predicted(self, ys, qids):
        for qid, y in zip(qids, ys):
            self.predicted[qid] = y.cpu().data[:self.lv[qid].shape[0]].numpy()
            
    def __len__(self):
        return len(self.fm_by_qid)
    
    def __getitem__(self, qid):
        feature = self.torch_.FloatTensor(self.fm_by_qid[qid], device=self.dev)
        lv = self.torch_.FloatTensor(self.lv[qid], device=self.dev) if self.lv[qid] is not None else None
        return feature, lv, qid
    

def collate_LTR(batch):
    batch_lens = [feature.shape[0] for feature, lv, qid in batch]
    max_len = max(batch_lens)
    X = torch.stack([torch.nn.functional.pad(feature,pad=(0,0,0,max_len-feature.shape[0])) for feature, lv, qid in batch])
    Y = torch.stack([torch.nn.functional.pad(lv,pad=(0,max_len-lv.shape[0]), value=-1) for feature, lv, qid in batch]) if batch[0][1] is not None else None
    qids = [qid for feature, lv, qid in batch]
    indices = torch.stack([torch.LongTensor(np.pad(np.arange(0, sample_size), (0, max_len - sample_size), "constant", constant_values=-1)) for sample_size in batch_lens], dim=0)
    return X, Y, indices, qids
    

def set_seed(random_seed):
    import random
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
def predict(net, fm, dlr, lv):
    qgdata = LTRData(fm, dlr)
    valid_dl = DataLoader(qgdata, batch_size=1, shuffle=False, collate_fn = collate_LTR)
    net.eval()
    
    for qid in range(dlr.shape[0] - 1):
        qgdata.update_labels([lv[dlr[qid]:dlr[qid]+1]], [qid])
            
    preds = []
    with torch.no_grad():
        for (x, y, indices, _) in valid_dl:
            mask = (y == -1)
            output = net(x, mask, indices).squeeze(dim=0)
#             y.append(np.mean(output.cpu().data.numpy(), 1))
            preds.append(output.cpu().data.numpy())
    return np.concatenate(preds)
    

def cltr(jobid, dataset_name, correction_method,
         net_config, dataset,
         epochs, learning_rate, rseed, 
         bernoulli, 
         is_rbem,
         results_file, verbose=False):
    
    if 'oracle' in correction_method:
        is_rbem = False
    if 'outlier' not in correction_method:
        outlierness = np.zeros_like(dataset.trfm[:,0])
        dataset.group_ids, dataset.biglist_index = outlier2group(outlierness, dataset.sessions, dataset.trdlr)
    
    set_seed(rseed)
    
    
    qgdata = LTRData(dataset.trfm, dataset.trdlr)
    train_dl = DataLoader(qgdata, batch_size=1, shuffle=True, collate_fn = collate_LTR)

    net = make_model(**net_config['model'], n_features=dataset.trfm.shape[1])
    if torch.cuda.is_available():
        net.cuda(get_torch_device())
    print(net)
    
    train_clicks = 0
    for c in dataset.clicks:
        train_clicks += c.sum()
    
    net.opt = torch.optim.Adagrad(net.parameters(), lr=learning_rate)
    
    correction_params = None
    pr1 = None
    if not is_rbem:
        if 'oracle' in correction_method:
            param_path = correction_method.split('_',1)[1]
            correction_method = 'oracle'

            with open(param_path, 'rb') as f:
                correction_params = pickle.load(f)
        else:
            correction_params = cltr(jobid, dataset_name, correction_method,
                                     net_config, dataset,
                                     epochs, learning_rate, rseed, 
                                     bernoulli, 
                                     is_rbem = True,
                                     results_file = results_file, verbose = verbose)
            return
       
        correction_op = Correction(correction_method.replace('outlier_', ''), 0.)
    #     First we need to call this module to read the shapes
        correction_op.init_params(dataset.clicks, dataset.trdlr, dataset.group_ids)
        correction_op.load_oracle_values(correction_params)
        
#         loss_fn = lambdaLoss
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
        
        results_file = os.path.join(results_file, jobid + '.json')
        
    else:
        correction_op = Correction(correction_method.replace('outlier_', ''), 0.6)

        correction_op.init_params(dataset.clicks, dataset.trdlr, dataset.group_ids)

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
        results_file = os.path.join(results_file, jobid + '_rbem.json')
    
    
    
    losses = []
    
    if is_rbem:
        pr1 = correction_op.expmax(qgdata.predicted, dataset.clicks, dataset.sessions, dataset.trdlr)
    
    for qid in range(dataset.trdlr.shape[0] - 1):
        lv = correction_op.debias(dataset.clicks[qid], dataset.sessions[qid], dataset.biglist_index[qid])
        qgdata.update_labels([lv], [qid])
    
    for epoch in range(epochs):
        if pr1 is not None:
            for qid in range(dataset.trdlr.shape[0] - 1):
                lv = pr1(dataset.clicks[qid], dataset.sessions[qid], dataset.biglist_index[qid])

                if bernoulli:
                    lv = np.random.binomial(1, lv)

                qgdata.update_labels([lv], [qid])
    
        net.train()
        for (x, y, indices, qids) in train_dl:
            net.opt.zero_grad()
            mask = (y == -1)
            out = net(x, mask, indices)
            out[mask] = -1e6
            y[mask] = 0
            loss = loss_fn(out, y)
            losses.append(loss.data)
            loss.backward()
            net.opt.step()
            qgdata.update_predicted(out, qids)
        
        if is_rbem:
            pr1 = correction_op.expmax(qgdata.predicted, dataset.clicks, dataset.sessions, dataset.trdlr)
        
        if log_epoch(epoch, epochs):
            train_ndcg = metrics.LTRMetrics(dataset.trlv,np.diff(dataset.trdlr),np.concatenate(qgdata.predicted, 0)).NDCG(10)
            valid_ndcg = -1
            if hasattr(dataset, 'vafm'):
                valid_ndcg = metrics.LTRMetrics(dataset.valv,np.diff(dataset.vadlr),predict(net,dataset.vafm, dataset.vadlr, dataset.valv)).NDCG(10)
            
            if verbose:
                print({'epoch': epoch+1, 'train': train_ndcg, 'valid': valid_ndcg, 'loss': float(np.array(losses).mean())})
            with open(results_file, 'a+') as f:
                json.dump({
                    'jobid': jobid, 'dataset': dataset_name, 'train_docs': dataset.trfm.shape[0], 
                           'train_size': dataset.trdlr.shape[0]-1, 'train_clicks': int(train_clicks),
                           'epoch': epoch+1, 'learning_rate': learning_rate, 
                           'config': net_config,
                           'train': train_ndcg, 'valid': valid_ndcg, 'loss': float(np.array(losses).mean()),
                           'correction': correction_method, 'bernoulli': bernoulli,
                           'correction_params': correction_op.get_params()
                          }, f)
                f.write('\n')
         
    if hasattr(dataset, 'tefm'):
        test_ndcg = metrics.LTRMetrics(dataset.telv,np.diff(dataset.tedlr),predict(net,dataset.tefm, dataset.tedlr, dataset.telv)).NDCG(10)

        with open(results_file, 'a+') as f:
            json.dump({
                'jobid': jobid, 'dataset': dataset_name, 'train_docs': dataset.trfm.shape[0], 
                       'train_size': dataset.trdlr.shape[0]-1, 'train_clicks': int(train_clicks),
                       'epoch': epoch+1, 'learning_rate': learning_rate,
                       'config': net_config,
                       'train': train_ndcg, 'valid': valid_ndcg, 'test': test_ndcg, 'loss': float(np.array(losses).mean()),
                       'correction': correction_method, 'bernoulli': bernoulli,
                       'correction_params': correction_op.get_params()
                      }, f)
            f.write('\n')
    
    if is_rbem:
        return correction_op.get_params()
    else:
        return net

