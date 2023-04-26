import time
import lightgbm as lgb
import metrics
import os
import numpy as np
import json

import pickle

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def _subsample_by_ids(dlr, fm, lv, qids):
    feature_matrix = []
    label_vector = []
    doclist_ranges = [0]
    for qid in qids:
        s_i = dlr[qid]
        e_i = dlr[qid+1]
        feature_matrix.append(fm[s_i:e_i, :])
        label_vector.append(lv[s_i:e_i])
        doclist_ranges.append(e_i - s_i)
    
    doclist_ranges = np.cumsum(np.array(doclist_ranges), axis=0)
    feature_matrix = np.concatenate(feature_matrix, axis=0)
    label_vector = np.concatenate(label_vector, axis=0)
    return doclist_ranges, feature_matrix, label_vector
    
def _random_subsample(dlr, fm, lv, subsample_size, seed):
    np.random.seed(seed)
    qids = np.random.randint(0, dlr.shape[0]-1, subsample_size)
    
    return _subsample_by_ids(dlr, fm, lv, qids)
    

def train_random_subsample(dataset, subsample_size, seed):
    np.random.seed(seed)
    qids = np.random.randint(0, dataset.trdlr.shape[0]-1, subsample_size)
    
    sub_ds = type('', (), {})()
    sub_ds.trdlr, sub_ds.trfm, sub_ds.trlv = _subsample_by_ids(dataset.trdlr, dataset.trfm, dataset.trlv, qids)
    return sub_ds
    
    
def subsample_splits(dataset, subsample_size, seed):
    sdataset = type('', (), {})()
    sdataset.trdlr, sdataset.trfm, sdataset.trlv = _random_subsample(dataset.trdlr, 
                                                                     dataset.trfm, 
                                                                     dataset.trlv, 
                                                                     subsample_size, seed)
    vasize = int(subsample_size * (dataset.vadlr.shape[0]-1) / (dataset.trdlr.shape[0]-1))
    sdataset.vadlr, sdataset.vafm, sdataset.valv = _random_subsample(dataset.vadlr, 
                                                                     dataset.vafm, 
                                                                     dataset.valv, 
                                                                     vasize, seed)
    tesize = int(subsample_size * (dataset.tedlr.shape[0]-1) / (dataset.trdlr.shape[0]-1))
    sdataset.tedlr, sdataset.tefm, sdataset.telv = _random_subsample(dataset.tedlr, 
                                                                     dataset.tefm, 
                                                                     dataset.telv, 
                                                                     tesize, seed)
    return sdataset
    

def kfold_train(dataset, k=5):
    folds = [type('', (), {})() for _ in range(k)]
    
    
    docs_len = dataset.trdlr.shape[0] - 1
    step_len = int(docs_len/k)
    
    for i in range(len(folds)):
        trqids = []
        trqids += list(range(i*step_len))
        trqids += list(range((i+1)*step_len, docs_len))
        
        folds[i].trdlr, folds[i].trfm, folds[i].trlv = _subsample_by_ids(dataset.trdlr, dataset.trfm, dataset.trlv, trqids)
        folds[i].vadlr, folds[i].vafm, folds[i].valv = _subsample_by_ids(dataset.trdlr, dataset.trfm, dataset.trlv, range(i*step_len, (i+1)*step_len))

        
    return folds
           
def remove_duplicates(dataset):
    dlr, fm, lv = [0], [], []
    for qid in range(dataset.trdlr.shape[0] - 1):
        s_i, e_i = dataset.trdlr[qid:qid+2]
        x = dataset.trfm[s_i:e_i,:]
        y = dataset.trlv[s_i:e_i]
        diff = x[:,None,:] - x[None,:,:]
        diff = diff.sum(2)
        diff += np.tril(np.ones(diff.shape))
        uniques = np.ones(diff.shape[0], dtype=np.bool8)
        uniques[np.where(diff==0)[0]] = False
#         print(uniques)
        fm.append(x[uniques,:])
        lv.append(y[uniques])
        dlr.append(sum(uniques))
    return np.concatenate(fm,0), np.concatenate(lv,0), np.cumsum(np.array(dlr))

def read_pkl(pkl_path, toy_size=-1, subsample_rseed=0):
    loaded_data = np.load(pkl_path, allow_pickle=True)
#     feature_map = loaded_data['feature_map'].item()
    train_feature_matrix = loaded_data['train_feature_matrix']
    train_doclist_ranges = loaded_data['train_doclist_ranges']
    train_label_vector   = loaded_data['train_label_vector']
    valid_feature_matrix = loaded_data['valid_feature_matrix']
    valid_doclist_ranges = loaded_data['valid_doclist_ranges']
    valid_label_vector   = loaded_data['valid_label_vector']
    test_feature_matrix  = loaded_data['test_feature_matrix']
    test_doclist_ranges  = loaded_data['test_doclist_ranges']
    test_label_vector    = loaded_data['test_label_vector']
    dataset = type('', (), {})()
#     dataset.fmap = feature_map
    dataset.trfm = train_feature_matrix
    dataset.tefm = test_feature_matrix
    dataset.vafm = valid_feature_matrix
    dataset.trdlr = train_doclist_ranges
    dataset.tedlr = test_doclist_ranges
    dataset.vadlr = valid_doclist_ranges
    dataset.trlv = train_label_vector
    dataset.telv = test_label_vector
    dataset.valv = valid_label_vector
    
    if toy_size > 0:
      sub_ds = train_random_subsample(dataset, toy_size, subsample_rseed)
      dataset.trdlr = sub_ds.trdlr
      dataset.trfm = sub_ds.trfm
      dataset.trlv = sub_ds.trlv
    dataset.trfm, dataset.trlv, dataset.trdlr = remove_duplicates(dataset)
    

    return dataset
    
def load_data_in_libsvm_format(data_path=None, file_prefix=None, feature_size=-1, topk=100):
    features = []
    dids = []
    initial_list = []
    qids = []
    labels = []
    initial_scores = []
    initial_list_lengths = []

    feature_fin = open(f'{data_path}/{file_prefix}/{file_prefix}.txt')
    qid_to_idx = {}
    line_num = -1
    for line in feature_fin:
        line_num += 1
        arr = line.strip().split(' ')
        qid = arr[1].split(':')[1]
        if qid not in qid_to_idx:
            qid_to_idx[qid] = len(qid_to_idx)
            qids.append(qid)
            initial_list.append([])
            labels.append([])

        # create query-document information
        qidx = qid_to_idx[qid]
        if len(initial_list[qidx]) == topk:
            continue
        initial_list[qidx].append(line_num)
        label = int(arr[0])
        labels[qidx].append(label)
        did = qid + '_' + str(line_num)
        dids.append(did)

        # read query-document feature vectors
        auto_feature_size = feature_size == -1
        
        if auto_feature_size:
            feature_size = 5

        features.append([0.0 for _ in range(feature_size)])
        for x in arr[2:]:
            arr2 = x.split(':')
            feature_idx = int(arr2[0]) - 1
            if feature_idx >= feature_size and auto_feature_size:
                features[-1] += [0.0 for _ in range(feature_idx - feature_size + 1)]
                feature_size = feature_idx + 1
            if feature_idx < feature_size:
                features[-1][int(feature_idx)] = float(arr2[1])

    feature_fin.close()

    initial_list_lengths = [
        len(initial_list[i]) for i in range(len(initial_list))]

    ds = {}
    ds['fm'] = np.array(features)
    ds['lv'] = np.concatenate([np.array(x) for x in labels], axis=0)
    ds['dlr'] = np.cumsum([0]+initial_list_lengths)
    return type('ltr', (object,), ds)

def load_splits_ultra(data_path):
    ds = {}
    feature_size = -1
    for sp in ['train', 'test', 'valid']:
        if os.path.exists(os.path.join(data_path, f'{sp}/{sp}.txt')):
            split_ds = load_data_in_libsvm_format(data_path, sp, feature_size)
            if sp == 'train':
                feature_size = split_ds.fm.shape[1]
            for vec in ['fm', 'lv', 'dlr']:
                ds[sp[:2] + vec] = eval(f'split_ds.{vec}')
    return type('ltr', (object,), ds)
    
    
def lv2clicks(lv, dlr):
    clicks = []
    for qid in range(dlr.shape[0] - 1):
        s_i, e_i = dlr[qid:qid+2]
        clicks.append(lv[None,s_i:e_i])
    return clicks

def lv2sessions(lv, dlr):
    sessions = []
    for qid in range(dlr.shape[0] - 1):
        s_i, e_i = dlr[qid:qid+2]
        sessions.append(np.arange(e_i-s_i)[None,:])
    return sessions

def outlier2group(outlierness, sessions, dlr):
    groups = []
    biglist_index = []
    last_position = 0
    for qid in range(dlr.shape[0] - 1):
        s_i, e_i = dlr[qid:qid+2]
        position = np.where(outlierness[s_i: e_i] > 0)[0]
        sess = sessions[qid]
        group = np.zeros(sess.shape[0])
        if len(position) > 0:
            group = np.where(sess==position[0])[1] + 1
            
        groups.append(group)
        biglist_index.append(np.arange(last_position, last_position+sess.shape[0]))
        last_position += sess.shape[0]
    return np.concatenate(groups, 0), biglist_index

def load_dataset(dataset_name, datasets_info, session_cnt):
    with open(datasets_info) as f:
        dataset_info = json.load(f)[dataset_name]
    if dataset_info['type'] == 'libsvm':
        dataset = load_splits_ultra(dataset_info['path'])
        clicks = lv2clicks(dataset.trlv, dataset.trdlr)
        sessions = lv2sessions(dataset.trlv, dataset.trdlr)
    elif dataset_info['type'] == 'pickle':
        with open(dataset_info['path'], 'rb') as f:
            dataset = type('ltr', (object,), pickle.load(f))
        if 'clicks_path' in dataset_info and os.path.exists(dataset_info['clicks_path']):
            with open(dataset_info['clicks_path'], 'rb') as f:
                clicks = pickle.load(f)
            if 'sessions_path' in dataset_info:
                with open(dataset_info['sessions_path'], 'rb') as f:
                    sessions = pickle.load(f)
            else:
                sessions = lv2sessions(dataset.trlv, dataset.trdlr)
                
        else:
            if 'lv2prob' in dataset_info:
                lv2prob = eval(dataset_info['lv2prob'])
                dataset.trlv = lv2prob(dataset.trlv)
            clicks = lv2clicks(dataset.trlv, dataset.trdlr)
            sessions = lv2sessions(dataset.trlv, dataset.trdlr)
    
    if 'outlierness' in dataset_info:
        outlierness = dataset.trfm[:, dataset_info['outlierness']]
    elif 'clicks_path' in dataset_info and os.path.exists(dataset_info['clicks_path'].replace('clicks', 'outlierness')):
        with open(dataset_info['clicks_path'].replace('clicks', 'outlierness'), 'rb') as f:
            outlierness = pickle.load(f)
    else:
        outlierness = np.zeros_like(dataset.trfm[:,0])
        
    
    dataset.clicks = clicks
    dataset.sessions = sessions
    
    session_cnt = min(session_cnt, clicks[0].shape[0])

    dataset.sessions = [np.repeat(x,session_cnt//x.shape[0],0) for x in dataset.sessions]
    dataset.sessions = [x[:session_cnt,:] for x in dataset.sessions]
    dataset.clicks = [x[:session_cnt,:] for x in dataset.clicks]
    dataset.group_ids, dataset.biglist_index = outlier2group(outlierness, dataset.sessions, dataset.trdlr)
    
    print('num features : {}'.format(dataset.trfm.shape[1]))
    print('num docs (train, valid, test) : ({},{},{})'.format(dataset.trfm.shape[0], dataset.vafm.shape[0], dataset.tefm.shape[0]))
    print('num queries (train, valid, test) : ({},{},{})'.format(dataset.trdlr.shape[0], dataset.vadlr.shape[0], dataset.tedlr.shape[0]))
    
    return dataset

def subsample_dataset(query_cnt, session_cnt, dataset):
    dataset.trdlr = dataset.dlr[:query_cnt+1]
    dataset.trfm = dataset.fm[:dataset.trdlr[-1],:]
    dataset.trlv = dataset.lv[:dataset.trfm.shape[0]]
    dataset.clicks = dataset.clicks[:dataset.trdlr.shape[0] - 1]
    dataset.sessions = dataset.sessions[:dataset.trdlr.shape[0] - 1]

    dataset.clicks = [x[:session_cnt,:] for x in dataset.clicks]
    dataset.sessions = [x[:session_cnt,:] for x in dataset.sessions]


    dataset.group_ids = dataset.group_ids[:query_cnt]
    
    return dataset


def lambdarank(dataset, model_path=None, learning_rate=0.05, n_estimators=300, eval_at=[10], early_stopping_rounds=10000):
    start = time.time()
    if model_path is not None and os.path.exists(model_path):
        booster = lgb.Booster(model_file=model_path)
        print('loading lgb took {} secs.'.format(time.time() - start))
        return booster.predict

    gbm = lgb.LGBMRanker(learning_rate=learning_rate, n_estimators=n_estimators)

    gbm.fit(dataset.trfm, dataset.trlv, 
          group=np.diff(dataset.trdlr), 
          eval_set=[(dataset.vafm, dataset.valv)],
          eval_group=[np.diff(dataset.vadlr)], 
          eval_at=eval_at, 
          early_stopping_rounds=early_stopping_rounds, 
          verbose=False)

    if model_path is not None:
        gbm.booster_.save_model(model_path)

    print('training lgb took {} secs.'.format(time.time() - start))
    return gbm.booster_.predict

