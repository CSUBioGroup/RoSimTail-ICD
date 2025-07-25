"""
    This file contains evaluation methods that take in a set of predicted labels 
        and a set of ground truth labels and calculate precision, recall, accuracy, f1, and metrics @k
"""
from collections import defaultdict
import csv
import json
import numpy as np
import os
import sys

from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

import datasets

MIMIC_3_DIR = "../data/mimic3/new/"
MIMIC_2_DIR = "../data/mimic2/"
def pre_micro_f1(yhat, y):
    ymic = y.ravel()
    yhatmic = yhat.ravel()
    intersect=np.logical_and(yhatmic, ymic).sum(axis=0)
    prec = intersect/ (yhatmic.sum(axis=0) + 1e-10)
    rec = intersect / (ymic.sum(axis=0) + 1e-10)

    f1 = 2*(prec*rec)/(prec+rec+ 1e-10)
    return f1

import torch

def pre_micro_f1_gpu(yhat, y):
    """
    计算 Micro-F1，支持 GPU 加速。
    yhat, y: (samples, labels)
    """
    yhat = yhat.float()
    y = y.float()

    ymic = y.view(-1)  # 扁平化处理 (ravel)
    yhatmic = yhat.view(-1)

    intersect = (yhatmic * ymic).sum()  # 计算 True Positive
    prec = intersect / (yhatmic.sum() + 1e-10)  # 避免除零
    rec = intersect / (ymic.sum() + 1e-10)

    f1 = 2 * (prec * rec) / (prec + rec + 1e-10)
    return f1


def pre_macro_f1(yhat, y):
    prec =intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    rec = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)

    f1 = 2*(prec*rec)/(prec+rec+ 1e-10)
    return f1


def pre_macro_f1_gpu(yhat, y):
    """
    计算 Macro-F1，支持 GPU 加速。
    yhat, y: (samples, labels)
    """
    yhat = yhat.float()
    y = y.float()

    intersect = (yhat * y).sum(dim=0)  # 计算每个标签的 True Positive
    prec = intersect / (yhat.sum(dim=0) + 1e-10)
    rec = intersect / (y.sum(dim=0) + 1e-10)

    f1 = 2 * (prec * rec) / (prec + rec + 1e-10)
    return f1.mean()  # 计算所有标签的平均 F1


#大到小 1->00.5
def find_theta(y, yhat_raw):
    sample,label=y.shape
    f1_map = {
        'max_macro_f1': np.zeros(label),
        "max_theta": np.ones(label) / 2
    }


    yhat = (yhat_raw > 0.39).astype(int)

    macro_f1_all = pre_macro_f1(yhat, y)
    micro_f1_all=pre_micro_f1(yhat, y)
    print("macro_f1_all, micro_f1_all(shape)",macro_f1_all.shape)
    print("macro_f1_all",macro_f1_all)
    print("micro_f1_all",micro_f1_all)

    # sorted_array, indices = np.sort(macro_f1_all)
    indices = np.argsort(y.sum(axis=0)) #根据值得大小进行排列
    # sort_order_desc = np.argsort(-macro_f1_all)
    yhat_temp =yhat.copy()
    max_mif=micro_f1_all
    for i in range(label):
        temp_i=indices[i]
        for t in np.arange(0.05,1.0,0.05):
        # for t in np.interp(np.linspace(0, 1, 51), (0, 1), indices[i]):
            yhat_temp[temp_i] = (yhat_raw[temp_i] > t).astype(int)
            micro_f1_temp = pre_micro_f1(yhat_temp, y)
            if micro_f1_temp>max_mif:
                max_mif=micro_f1_temp
                f1_map['max_macro_f1'] = pre_macro_f1(yhat_temp, y)
                f1_map['max_theta'][temp_i]=t
                f1_map['max_micro_f1']=max_mif
    the_first_f1 = np.mean(f1_map['max_macro_f1'])
    print("the_macro_f1", the_first_f1)
    print("max_mif", max_mif)
    return f1_map

#小到大0.05->1
def find_theta_2(y, yhat_raw):
    sample,label=y.shape
    print(y.shape)
    f1_map = {
        'max_macro_f1': np.zeros(label),
        "max_theta": np.ones(label) / 2
    }
    yhat = (yhat_raw > 0.5).astype(int)

    macro_f1_all = pre_macro_f1(yhat, y)
    micro_f1_all=pre_micro_f1(yhat, y)

    print("macro_f1_all, micro_f1_all(shape)",macro_f1_all.shape)
    print("macro_f1_all",macro_f1_all)
    print("micro_f1_all",micro_f1_all)

    # sorted_array, indices = np.sort(macro_f1_all)
    indices = np.argsort(macro_f1_all)
    # sort_order_desc = np.argsort(-macro_f1_all)
    yhat_temp =yhat
    max_mif=micro_f1_all
    for i in range(label):
        temp_i=indices[i]
        for t in np.linspace(0.1, 0.55,55):
            yhat_temp[temp_i] = (yhat_raw[temp_i] > t).astype(int)
            micro_f1_temp = pre_micro_f1(yhat_temp, y)
            if micro_f1_temp>max_mif:
                max_mif=micro_f1_temp
                f1_map['max_macro_f1'] = pre_macro_f1(yhat_temp, y)
                f1_map['max_theta'][temp_i]=t
                f1_map['max_micro_f1']=max_mif
    the_first_f1 = np.mean(f1_map['max_macro_f1'])
    print("the_first_f1", the_first_f1)
    print("max_mif", max_mif)
    return f1_map



import numpy as np
import os
from tqdm import tqdm

def find_theta_binary_search(y, yhat_raw, save_path="best_theta_clean.npy"):

    if os.path.exists(save_path):
        f1_map = np.load(save_path, allow_pickle=True).item()
        return f1_map

    sample, label = y.shape
    f1_map = {
        'max_macro_f1': np.zeros(label),  
        "max_theta": np.ones(label) / 2,  
        "max_micro_f1": 0  
    }

    yhat = (yhat_raw > 0.5).astype(int)
    macro_f1_all = pre_macro_f1(yhat, y)
    micro_f1_all = pre_micro_f1(yhat, y)

    indices = np.argsort(macro_f1_all)
    max_mif = micro_f1_all  

    for i in tqdm(range(label), desc="Optimizing θ", ncols=100, unit="label"):
        temp_i = indices[i]
        low, high = 0.1, 0.5  
        best_t = 0.5  
        best_f1 = max_mif  

        for _ in tqdm(range(6), desc=f"Label {temp_i}", leave=False, ncols=60, unit="iter"):
            mid = (low + high) / 2
            yhat_temp = yhat.copy()
            yhat_temp[:, temp_i] = (yhat_raw[:, temp_i] > mid).astype(int)

            micro_f1_temp = pre_micro_f1(yhat_temp, y)

            if micro_f1_temp > best_f1:
                best_f1 = micro_f1_temp
                best_t = mid
                low = mid  
            else:
                high = mid  

        f1_map['max_theta'][temp_i] = best_t
        f1_map['max_micro_f1'] = best_f1

    np.save(save_path, f1_map)

    return f1_map


import torch
def find_theta_binary_search_gpu(y, yhat_raw, device="cuda"):

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    y = torch.tensor(y, dtype=torch.float32, device=device)  # 转换到 GPU
    yhat_raw = torch.tensor(yhat_raw, dtype=torch.float32, device=device)  # 转换到 GPU

    sample, label = y.shape
    f1_map = {
        'max_macro_f1': torch.zeros(label, device=device),  
        "max_theta": torch.ones(label, device=device) / 2,  
        "max_micro_f1": torch.tensor(0.0, device=device)  
    }

    yhat = (yhat_raw > 0.5).float()  
    macro_f1_all = pre_macro_f1_gpu(yhat, y)  
    micro_f1_all = pre_micro_f1_gpu(yhat, y)  

    indices = torch.argsort(macro_f1_all)

    max_mif = micro_f1_all

    for i in tqdm(range(label), desc="Optimizing θ", ncols=100, unit="label"):
        temp_i = indices[i].item()

        low, high = 0.1, 0.5  
        best_t = 0.5  
        best_f1 = max_mif  

        for _ in tqdm(range(6), desc=f"Label {temp_i}", leave=False, ncols=60, unit="iter"):
            mid = (low + high) / 2
            yhat_temp = yhat.clone()  
            yhat_temp[:, temp_i] = (yhat_raw[:, temp_i] > mid).float()  

            micro_f1_temp = pre_micro_f1_gpu(yhat_temp, y)  

            if micro_f1_temp > best_f1:
                best_f1 = micro_f1_temp
                best_t = mid
                low = mid  
            else:
                high = mid  

        f1_map['max_theta'][temp_i] = best_t
        f1_map['max_micro_f1'] = best_f1

    return f1_map




def find_theta_macro(y, yhat_raw):
    sample,label=y.shape
    print(y.shape)
    macro_map=dict()
    macro_map['max_macro_f1']=np.zeros(label)
    macro_map['max_macro_precision']=np.zeros(label)
    macro_map['max_macro_recall']=np.zeros(label)
    macro_map["max_theta"]=np.ones(label)
    macro_map["max_yhat"]=np.zeros(label)
    for t in np.linspace(0.2, 0.9,100):
        yhat = (yhat_raw > t).astype(int)

        macro_precision = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
        macro_recall = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)

        macro_f1 = 2*(macro_precision*macro_recall)/(macro_precision+macro_recall+1e-10)

        for i in range(label):
            if macro_f1[i] > macro_map['max_macro_f1'][i]:
                macro_map['max_macro_f1'][i] = macro_f1[i]
                macro_map['max_macro_precision'][i]=macro_precision[i]
                macro_map['max_macro_recall'][i]=macro_recall[i]
                macro_map['max_theta'][i]=t
                macro_map["max_yhat"][i]=yhat.sum(axis=0)[i]

    the_first_f1=np.mean(macro_map['max_macro_f1'])
    for i in range(label):
        print("第i个",i)
        print("macro_map[max_macro_f1]",macro_map['max_macro_f1'][i])
    
    prec=np.mean(macro_map['max_macro_precision'])
    rec=np.mean(macro_map['max_macro_recall'])
    
    if prec + rec == 0:
        the_second_f1 = 0.
    else:
        the_second_f1 = 2*(prec*rec)/(prec+rec)
    print("the_first_f1",the_first_f1)
    print("the_second_f1",the_second_f1)

    print(" ")
    print("yhat",macro_map["max_yhat"].sum(axis=0))
    
    yhat_raw_temp=np.zeros((sample,label))
    for i in range(label):
        theta=macro_map['max_theta'][i]
        print("第",i,"个: ",theta)
        yhat_raw_temp[:,i] = (yhat_raw[:,i] > theta).astype(int)
    ymic = y.ravel()
    yhatmic = yhat_raw_temp.ravel()
    print("ymic",ymic.shape)
    print("yhatmic",yhatmic.shape)
    
    micro_precision=intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)
    print("intersect_size(yhatmic, ymic, 0)",intersect_size(yhatmic, ymic, 0))
    
    print(" yhatmic.su", yhatmic.sum(axis=0))
    micro_recall=intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)
    print(" ymic.su", ymic.sum(axis=0))
    print("micro_precision",micro_precision)
    print("micro_recall",micro_recall)
    if micro_precision + micro_recall == 0:
        micro_f1 = 0.
    else:
        micro_f1 = 2*(micro_precision*micro_recall)/(micro_precision+micro_recall)
    print("micro_f1",micro_f1)
    return macro_map,the_first_f1,the_second_f1,micro_f1


def all_metrics(y, yhat_raw):

    names = ["acc", "prec", "rec", "f1"]

    max_macro_f1=np.zeros(50)
    max_theta=np.zeros(50)
    for t in np.linspace(0.1, 0.9, 10):
        yhat = (yhat_raw > t).astype(int)

        macro_precision = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
        macro_recall = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
        print("safa ",macro_precision.shape)

        macro_f1 = 2*(macro_precision*macro_recall)/(macro_precision+macro_recall+1e-10)

        for i in range(50):
            if macro_f1[i] > max_macro_f1[i]:
                max_macro_f1[i] = macro_f1[i]
                max_theta[i]=t



def all_metrics(yhat, y, k=8, yhat_raw=None, calc_auc=True):
    """
        Inputs:
            yhat: binary predictions matrix 
            y: binary ground truth matrix
            k: for @k metrics
            yhat_raw: prediction scores matrix (floats)
        Outputs:
            dict holding relevant metrics 
    """
    names = ["acc", "prec", "rec", "f1"]


    # print("yhat",yhat.shape)
    # print("y",y.shape)
    # print("yhat_raw",yhat_raw.shape)
    #macro
    macro = all_macro(yhat, y)


    #micro
    ymic = y.ravel()
    yhatmic = yhat.ravel()
    # print("ymic",ymic.shape)
    # print("yhatmic",yhatmic.shape)
    
    micro = all_micro(yhatmic, ymic)

    metrics = {names[i] + "_macro": macro[i] for i in range(len(macro))}
    metrics.update({names[i] + "_micro": micro[i] for i in range(len(micro))})

    #AUC and @k
    if yhat_raw is not None and calc_auc:
        #allow k to be passed as int or list
        if type(k) != list:
            k = [k]
        for k_i in k:
            rec_at_k = recall_at_k(yhat_raw, y, k_i)
            metrics['rec_at_%d' % k_i] = rec_at_k
            prec_at_k = precision_at_k(yhat_raw, y, k_i)
            metrics['prec_at_%d' % k_i] = prec_at_k
            metrics['f1_at_%d' % k_i] = 2*(prec_at_k*rec_at_k)/(prec_at_k+rec_at_k)

        roc_auc = auc_metrics(yhat_raw, y, ymic)
        metrics.update(roc_auc)

    return metrics

def all_macro(yhat, y):
    return macro_accuracy(yhat, y), macro_precision(yhat, y), macro_recall(yhat, y), macro_f1(yhat, y)

def all_micro(yhatmic, ymic):
    return micro_accuracy(yhatmic, ymic), micro_precision(yhatmic, ymic), micro_recall(yhatmic, ymic), micro_f1(yhatmic, ymic)


def macro_accuracy(yhat, y):
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    return np.mean(num)

def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1


def inst_precision(yhat, y):
    num = intersect_size(yhat, y, 1) / yhat.sum(axis=1)
    #correct for divide-by-zeros
    num[np.isnan(num)] = 0.
    return np.mean(num)

def inst_recall(yhat, y):
    num = intersect_size(yhat, y, 1) / y.sum(axis=1)
    #correct for divide-by-zeros
    num[np.isnan(num)] = 0.
    return np.mean(num)

def inst_f1(yhat, y):
    prec = inst_precision(yhat, y)
    rec = inst_recall(yhat, y)
    f1 = 2*(prec*rec)/(prec+rec)
    return f1


def recall_at_k(yhat_raw, y, k):
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i,tk].sum()
        denom = y[i,:].sum()
        vals.append(num_true_in_top_k / float(denom))

    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.

    return np.mean(vals)

def precision_at_k(yhat_raw, y, k):
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y[i,tk].sum()
            denom = len(tk)
            vals.append(num_true_in_top_k / float(denom))

    return np.mean(vals)


def micro_accuracy(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / union_size(yhatmic, ymic, 0)

def micro_precision(yhatmic, ymic):
     return intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)

def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)

def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

def auc_metrics(yhat_raw, y, ymic):
    if yhat_raw.shape[0] <= 1:
        return
    fpr = {}
    tpr = {}
    roc_auc = {}
    #get AUC for each label individually
    relevant_labels = []
    auc_labels = {}
    for i in range(y.shape[1]):
        #only if there are true positives for this label
        if y[:,i].sum() > 0:
            fpr[i], tpr[i], _ = roc_curve(y[:,i], yhat_raw[:,i])
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                auc_score = auc(fpr[i], tpr[i])
                if not np.isnan(auc_score): 
                    auc_labels["auc_%d" % i] = auc_score
                    relevant_labels.append(i)

    #macro-AUC: just average the auc scores
    aucs = []
    for i in relevant_labels:
        aucs.append(auc_labels['auc_%d' % i])
    roc_auc['auc_macro'] = np.mean(aucs)

    #micro-AUC: just look at each individual prediction
    yhatmic = yhat_raw.ravel()
    fpr["micro"], tpr["micro"], _ = roc_curve(ymic, yhatmic) 
    roc_auc["auc_micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc



def results_by_type(Y, mdir, version='mimic3'):
    d2ind = {}
    p2ind = {}

    #get predictions for diagnoses and procedures
    diag_preds = defaultdict(lambda: set([]))
    proc_preds = defaultdict(lambda: set([]))
    preds = defaultdict(lambda: set())
    with open('%s/preds_test.psv' % mdir, 'r') as f:
        r = csv.reader(f, delimiter='|')
        for row in r:
            if len(row) > 1:
                for code in row[1:]:
                    preds[row[0]].add(code)
                    if code != '':
                        try:
                            pos = code.index('.')
                            if pos == 3 or (code[0] == 'E' and pos == 4):
                                if code not in d2ind:
                                    d2ind[code] = len(d2ind)
                                diag_preds[row[0]].add(code)
                            elif pos == 2:
                                if code not in p2ind:
                                    p2ind[code] = len(p2ind)
                                proc_preds[row[0]].add(code)
                        except:
                            if len(code) == 3 or (code[0] == 'E' and len(code) == 4):
                                if code not in d2ind:
                                    d2ind[code] = len(d2ind)
                                diag_preds[row[0]].add(code)
    #get ground truth for diagnoses and procedures
    diag_golds = defaultdict(lambda: set([]))
    proc_golds = defaultdict(lambda: set([]))
    golds = defaultdict(lambda: set())
    test_file = '%s/test_%s.csv' % (MIMIC_3_DIR, str(Y)) if version == 'mimic3' else '%s/test.csv' % MIMIC_2_DIR
    with open(test_file, 'r') as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            codes = set([c for c in row[3].split(';')])
            for code in codes:
                golds[row[1]].add(code)
                try:
                    pos = code.index('.')
                    if pos == 3:
                        if code not in d2ind:
                            d2ind[code] = len(d2ind)
                        diag_golds[row[1]].add(code)
                    elif pos == 2:
                        if code not in p2ind:
                            p2ind[code] = len(p2ind)
                        proc_golds[row[1]].add(code)
                except:
                    if len(code) == 3 or (code[0] == 'E' and len(code) == 4):
                        if code not in d2ind:
                            d2ind[code] = len(d2ind)
                        diag_golds[row[1]].add(code)

    hadm_ids = sorted(set(diag_golds.keys()).intersection(set(diag_preds.keys())))

    ind2d = {i:d for d,i in d2ind.items()}
    ind2p = {i:p for p,i in p2ind.items()}
    type_dicts = (ind2d, ind2p)
    return diag_preds, diag_golds, proc_preds, proc_golds, golds, preds, hadm_ids, type_dicts


def diag_f1(diag_preds, diag_golds, ind2d, hadm_ids):
    num_labels = len(ind2d)
    yhat_diag = np.zeros((len(hadm_ids), num_labels))
    y_diag = np.zeros((len(hadm_ids), num_labels))
    for i,hadm_id in tqdm(enumerate(hadm_ids)):
        yhat_diag_inds = [1 if ind2d[j] in diag_preds[hadm_id] else 0 for j in range(num_labels)]
        gold_diag_inds = [1 if ind2d[j] in diag_golds[hadm_id] else 0 for j in range(num_labels)]
        yhat_diag[i] = yhat_diag_inds
        y_diag[i] = gold_diag_inds
    return micro_f1(yhat_diag.ravel(), y_diag.ravel())

def proc_f1(proc_preds, proc_golds, ind2p, hadm_ids):
    num_labels = len(ind2p)
    yhat_proc = np.zeros((len(hadm_ids), num_labels))
    y_proc = np.zeros((len(hadm_ids), num_labels))
    for i,hadm_id in tqdm(enumerate(hadm_ids)):
        yhat_proc_inds = [1 if ind2p[j] in proc_preds[hadm_id] else 0 for j in range(num_labels)]
        gold_proc_inds = [1 if ind2p[j] in proc_golds[hadm_id] else 0 for j in range(num_labels)]
        yhat_proc[i] = yhat_proc_inds
        y_proc[i] = gold_proc_inds
    return micro_f1(yhat_proc.ravel(), y_proc.ravel())

def metrics_from_dicts(preds, golds, mdir, ind2c):
    with open('%s/pred_100_scores_test.json' % mdir, 'r') as f:
        scors = json.load(f)

    hadm_ids = sorted(set(golds.keys()).intersection(set(preds.keys())))
    num_labels = len(ind2c)
    yhat = np.zeros((len(hadm_ids), num_labels))
    yhat_raw = np.zeros((len(hadm_ids), num_labels))
    y = np.zeros((len(hadm_ids), num_labels))
    for i,hadm_id in tqdm(enumerate(hadm_ids)):
        yhat_inds = [1 if ind2c[j] in preds[hadm_id] else 0 for j in range(num_labels)]
        yhat_raw_inds = [scors[hadm_id][ind2c[j]] if ind2c[j] in scors[hadm_id] else 0 for j in range(num_labels)]
        gold_inds = [1 if ind2c[j] in golds[hadm_id] else 0 for j in range(num_labels)]
        yhat[i] = yhat_inds
        yhat_raw[i] = yhat_raw_inds
        y[i] = gold_inds
    return yhat, yhat_raw, y, all_metrics(yhat, y, yhat_raw=yhat_raw, calc_auc=False)


def union_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)

def intersect_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)

def print_metrics(metrics):
    print()
    if "auc_macro" in metrics.keys():
        print("[MACRO] accuracy, precision, recall, f-measure, AUC")
        print("%.4f, %.4f, %.4f, %.4f, %.4f" % (metrics["acc_macro"], metrics["prec_macro"], metrics["rec_macro"], metrics["f1_macro"], metrics["auc_macro"]))
    else:
        print("[MACRO] accuracy, precision, recall, f-measure")
        print("%.4f, %.4f, %.4f, %.4f" % (metrics["acc_macro"], metrics["prec_macro"], metrics["rec_macro"], metrics["f1_macro"]))

    if "auc_micro" in metrics.keys():
        print("[MICRO] accuracy, precision, recall, f-measure, AUC")
        print("%.4f, %.4f, %.4f, %.4f, %.4f" % (metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"], metrics["auc_micro"]))
    else:
        print("[MICRO] accuracy, precision, recall, f-measure")
        print("%.4f, %.4f, %.4f, %.4f" % (metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"]))
    for metric, val in metrics.items():
        if metric.find("rec_at") != -1:
            print("%s: %.4f" % (metric, val))
    print()

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("usage: python " + str(os.path.basename(__file__) + " [train_dataset] [|Y| (as string)] [version (mimic2 or mimic3)] [model_dir]"))
        sys.exit(0)
    train_path, Y, version, mdir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    ind2c, _ = datasets.load_full_codes(train_path, version=version)

    diag_preds, diag_golds, proc_preds, proc_golds, golds, preds, hadm_ids, type_dicts = results_by_type(Y, mdir, version)
    yhat, yhat_raw, y, metrics = metrics_from_dicts(preds, golds, mdir, ind2c)
    print_metrics(metrics)

    k = [5] if Y == '50' else [8,15]
    prec_at_8 = precision_at_k(yhat_raw, y, k=8)
    print("PRECISION@8: %.4f" % prec_at_8)
    prec_at_15 = precision_at_k(yhat_raw, y, k=15)
    print("PRECISION@15: %.4f" % prec_at_15)

    f1_diag = diag_f1(diag_preds, diag_golds, type_dicts[0], hadm_ids)
    f1_proc = proc_f1(proc_preds, proc_golds, type_dicts[1], hadm_ids)
    print("[BY CODE TYPE] f1-diag f1-proc")
    print("%.4f %.4f" % (f1_diag, f1_proc))
