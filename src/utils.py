import torch
import tqdm
import torch.nn.functional as F
import pickle
import numpy as np
from collections import Counter


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss



import os
def buildAdjacencyCOOC(vocab, train_data, output_file="co_occurrence_matrix.txt"):

    if os.path.exists(output_file):
        print(f"{output_file} 已经存在，直接加载...")
        co_occurrence_matrix = np.loadtxt(output_file, delimiter="\t")
        return co_occurrence_matrix

    print(f"{output_file} 不存在，开始构建共现矩阵...")
    num_labels = len(vocab.label_to_id)  # 标签总数
    co_occurrence_matrix = np.zeros((num_labels, num_labels), dtype=np.int32)

    for example in train_data:
        label_indices = example["label_ids"]
        
        for i in range(len(label_indices)):
            for j in range(i + 1, len(label_indices)):
                co_occurrence_matrix[label_indices[i], label_indices[j]] += 1
                co_occurrence_matrix[label_indices[j], label_indices[i]] += 1

    total_sum = np.sum(co_occurrence_matrix)
    if total_sum > 0:
        normalized_matrix = co_occurrence_matrix / total_sum
    else:
        normalized_matrix = co_occurrence_matrix

    np.savetxt(output_file, normalized_matrix, fmt="%.6f", delimiter="\t")
    print(f"归一化后的共现矩阵已保存到文件：{output_file}")

    return normalized_matrix
    


def compute_class_freq_and_train_num(label_to_id, train_dataset, freq_cutoff=0):
    all_labels = []
    r_all = []


    term2count = Counter(all_labels)

    for label_id in label_to_id.values():
        if label_id not in term2count:
            term2count[label_id] = 0  # 如果标签不存在，则频次为0

    filtered_labels = {term: count for term, count in term2count.items() if count >= freq_cutoff}

    labels_ref = [label_id for label, label_id in label_to_id.items() if label_id in filtered_labels]

    class_freq = [filtered_labels.get(label, 0) for label in labels_ref]

    train_num = len(train_dataset)

    print(f"======np.mean(r_all):{np.mean(r_all)}======")
    print(f"======np.median(r_all):{np.median(r_all)}======")
    s = (np.mean(r_all) + np.median(r_all)) / 2
    print(f"======(np.mean(r_all) + np.median(r_all))/2:{s}======")
    print(f"train_num：{train_num}")
    
    return class_freq

def compute_cosine_similarity_loss(att,labDescVec):
    labDescVec = labDescVec.to(att.device)
    cosine_sim = F.cosine_similarity(att, labDescVec, dim=-1)
    similarity_loss = 1 - cosine_sim.mean() + 0.1
    return similarity_loss


def compute_cosine_similarity_loss(att,labDescVec):
    labDescVec = labDescVec.to(att.device)
    cosine_sim = F.cosine_similarity(att, labDescVec, dim=-1)
    similarity_loss = 1 - cosine_sim.mean() + 0.1
    return similarity_loss



def compute_cosine_similarity_loss_tail(att, labDescVec, head_tail_mask):

    labDescVec = labDescVec.to(att.device)
    

    cosine_sim = F.cosine_similarity(att, labDescVec, dim=-1)

    cosine_loss_per_label = 1 - cosine_sim

    cosine_loss_per_label_tail = cosine_loss_per_label[:, head_tail_mask == 0]

    tail_similarity_loss = cosine_loss_per_label_tail.sum()

    return tail_similarity_loss



def head_tail_mask_bycoverage(class_freq, label_to_id, coverage_ratio=0.6):
    print(f"coverage_ratio: {coverage_ratio}")
    class_freq = np.array(class_freq)
    sorted_labels = np.argsort(class_freq)[::-1]
    sorted_freqs = class_freq[sorted_labels]
    
    total_samples = np.sum(sorted_freqs)
    cumulative_coverage = np.cumsum(sorted_freqs) / total_samples
    
    split_idx = np.searchsorted(cumulative_coverage, coverage_ratio)  

    head_labels = set(sorted_labels[:split_idx])
    tail_labels = set(sorted_labels[split_idx:])

    head_tail_mask = [1 if i in head_labels else 0 for i in range(len(label_to_id))]

    return head_tail_mask, tail_labels

def update_head_tail_mask(head_tail_mask, class_freq, tail_labels, top_percentage=0.1):
    class_freq = np.array(class_freq)
    tail_labels = np.array(list(tail_labels))
    tail_freqs = class_freq[tail_labels]
    
    sorted_tail_labels = np.argsort(tail_freqs)[::-1]
    sorted_freqs = tail_freqs[sorted_tail_labels]
    
    total_tail = np.sum(sorted_freqs)
    cumulative_coverage = np.cumsum(sorted_freqs) / total_tail
    
    split_idx = np.searchsorted(cumulative_coverage, top_percentage)

    head_labels = set(sorted_tail_labels[:split_idx])  # 头部标签集合
    tail_labels = set(sorted_tail_labels[split_idx:])  # 尾部标签集合
    
    for idx in head_labels:
        if head_tail_mask[idx] == 0:
            head_tail_mask[idx] = 1
    
    return head_tail_mask