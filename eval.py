import json
import pickle
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from model.cross_encoder import DoubleBert
from train_v2 import MyTrainer
from config import Configs
from dataloader import MyDataset
from config import Configs2
from tqdm import tqdm    
import numpy as np

config = Configs2

model = DoubleBert(config=config)

state_dict = torch.load('./model/Full-Train_epoch_1_step_10801.pth')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

model.load_state_dict(state_dict, strict=False)


def sort_list_of_dicts(list_of_dicts):
    sorted_list = sorted(list_of_dicts, key=lambda x: x['score'])
    return sorted_list


def search_qid(list_of_dicts, target_qid):
    left, right = 0, len(list_of_dicts) - 1

    while left <= right:
        mid = (left + right) // 2
        mid_qid = list_of_dicts[mid]["QID"]

        if int(mid_qid) == int(target_qid):
            return list_of_dicts[mid]["PID"]
        elif int(mid_qid) < int(target_qid):
            left = mid + 1
        else:
            right = mid - 1

    return None

def numpy_encoder(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError("Type not serializable")


with open(config.load_biencoder, 'rb') as file:
    base_encoder = pickle.load(file)
    
collection = json.load(open(config.load_collection, 'r'))


list_eval = []


for sample in tqdm(base_encoder):
    

# for sample in tqdm(query):
#     with torch.no_grad():
#         query = sample['query']
#         positive = sample['positive']['text']
#         id_pos = sample['positive']['id']
        
#         sim, soft = model(query, positive)
        
#         list_eval.append({
#             'id': id_pos,
#             'sim': sim.item()
#         })
        
#         for item in sample['negative']:
#             text = item['text']
#             id_neg = item['id']

#             sim, soft = model(query, text)
            
#             list_eval.append({
#                 'id': id_neg,
#                 'sim': sim.item()
#             })


# for item in tqdm(pair_train):
#     temp = []
#     query = ''
#     for itm in query_list:
#         if itm['QID'] == str(item['qid']):
#             query = itm['PID']
#             break
#     id_pos = item['pid']
#     pos_text = collection[item['pid']]['Description']
    
#     neg = []
#     for itm in item['bm25_top_1000_pids']:
#         id_neg = itm
#         neg_text = collection[id_neg]['Description']
#         neg.append({
#             'id': id_neg,
#             'text': neg_text,
#         })
        
#     list_eval.append({
#         'query': query,
#         'positive': {
#             'id': id_pos,
#             'text': pos_text,
#         },
#         'negative': neg
#     })
    
    
    
            
# json.dump(list_eval, open('/home/luungoc/BTL-2023.1/Deep learning/src/result.json', 'w'), ensure_ascii=False, default=numpy_encoder)



for sample in tqdm(query_list[:1000]):
    with torch.no_grad():
        query = sample['query']
        positive = sample['positive']['text']
        id_pos = sample['positive']['id']
        
        sim, soft = model(query, positive)
        
        list_eval.append({
            'id': id_pos,
            'sim': sim.item()
        })
        
        for item in tqdm(sample['negative'][:100]):
            text = item['text']
            id_neg = item['id']

            sim, soft = model(query, text)
            
            list_eval.append({
                'id': id_neg,
                'sim': sim.item()
            })
            
json.dump(list_eval, open('./hihi.json', 'w'), ensure_ascii=False, default=numpy_encoder)