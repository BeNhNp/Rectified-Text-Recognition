#!/usr/bin/python
# -*- coding:utf-8 -*-

def get_str_list(output, target, dataset=None):
    # label_seq
    assert output.dim() == 2 and target.dim() == 2
    
    end_label = dataset.char2id[dataset.EOS]
    unknown_label = dataset.char2id[dataset.UNKNOWN]
    num_samples, max_len_labels = output.size()
    num_classes = dataset.num_classes
#     print(target.shape, output.shape)
    assert num_samples == target.size(0) and max_len_labels <= target.size(1)
    output = output.numpy()
    target = target.numpy()
    
    # list of char list
    pred_list, targ_list = [], []
    for i in range(num_samples):
        pred_list_i = []
        for j in range(max_len_labels):
            if output[i, j] != end_label:
                if output[i, j] != unknown_label and output[i, j] < len(dataset.id2char):
                    pred_list_i.append(dataset.id2char[output[i, j]])
            else:
                break
        pred_list.append(pred_list_i)
    
    for i in range(num_samples):
        targ_list_i = []
        for j in range(max_len_labels):
            if target[i, j] != end_label:
                if target[i, j] != unknown_label:
                    targ_list_i.append(dataset.id2char[target[i, j]])
            else:
                break
        targ_list.append(targ_list_i)
    
    # char list to string
    pred_list = [''.join(pred).lower() for pred in pred_list]
    targ_list = [''.join(targ).lower() for targ in targ_list]

    return pred_list, targ_list

def Accuracy(output, target, dataset=None):
    pred_list, targ_list = get_str_list(output, target, dataset)
    
    acc_list = [(pred == targ) for pred, targ in zip(pred_list, targ_list)]
    accuracy = 1.0 * sum(acc_list) / len(acc_list)
    return accuracy
