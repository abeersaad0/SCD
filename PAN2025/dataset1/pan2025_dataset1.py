import os
import sys
import re
import argparse
import torch ##
import torch.nn as nn##
import random
import numpy as np
from os.path import join
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModel
import glob
import json
import codecs#to read files
import unicodedata
from collections import Counter
import shutil
from collections import defaultdict
import html
from itertools import combinations
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Parameter, Linear
from torch.nn import LayerNorm

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str)
parser.add_argument('-o', type=str)

args = parser.parse_args()

input_path =args.i
output_path = args.o
os.makedirs(output_path, exist_ok=True)



def seed_everything(seed):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def validation_files(document_path):
    document =codecs.open(document_path, "r", "utf-8").read()
    if "\r" in document:
      document = document.replace ("\r", "\\r")
    if "\n\n" in document:
      document = re.sub(r'\n+', '\n', document).strip('\n')
    para_list = document.split("\n")
    para_list = list(filter(None, para_list))

    return para_list




def load_data(filename):
    print('loading test set')
    tokenized_para = {}
        
    for idx, document_path in enumerate(tqdm(glob.glob(filename + '/*.txt'))):           
        share_id = os.path.basename(document_path)[8:-4]
        para_list = validation_files(document_path)

        tokenized_para[share_id] =[tokenizer(paragraph, return_tensors='pt', max_length=max_seq_length, truncation=True,padding='max_length') for paragraph in para_list]

    return tokenized_para



seed=42
max_seq_length = 256 #512 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ("device:", device)
seed_everything(seed)
# model initial
model_id = 'roberta-large'
model_id2 = 'AIDA-UPM/star'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model_features = AutoModel.from_pretrained(model_id2).to(device)
print('model set was initialized')


print ("the input path is",input_path)
input_data = load_data (input_path)
print('Test set was loaded(', len(input_data), "files)")

preds_file = {}
counter_for_print = 0
for data in input_data.keys(): #one document "data"[] in documents "input_data"{}
  if counter_for_print % 1000==0:
      print ( counter_for_print,flush=True)

  preds_file[data] = []
  for dic in input_data[data]: #one paragraph "dic"{} in one document "input_data[data]"[]
      cur_input_ids = dic['input_ids'].squeeze(1).to(device)
      cur_attention_mask = dic['attention_mask'].squeeze(1).to(device)
      #cur_token_type_ids = dic['token_type_ids'].squeeze(1).to(device)
      with torch.no_grad():
        outputs = model_features(input_ids=cur_input_ids, attention_mask=cur_attention_mask)#, token_type_ids=cur_token_type_ids)

      preds_file[data].append(outputs.last_hidden_state[:,0,:].tolist()[0])
  counter_for_print+=1

print('features were extracted')


#one layer of GCN
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        self.reset_parameters()
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col = edge_index
        if x.size(0) == 1:
            out_degree = torch.tensor([1], dtype=torch.long) .to(device)
        else:
            out_degree = degree(col, num_nodes=x.size(0), dtype=x.dtype)
        deg =  out_degree #+ in_degree
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,norm=norm)
        out += self.bias
        return out
        
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()



#graph net
class Net(torch.nn.Module):
    def __init__(self, num_node_features, dropout =0.5):
        super(Net, self).__init__()
        
        self.conv1 = GCNConv(num_node_features, num_node_features//2)
        self.conv2 = GCNConv(num_node_features//2, num_node_features//4)
        self.conv3 = GCNConv(num_node_features//4, num_node_features//16)
        self.conv4 = GCNConv(num_node_features//16, num_node_features//32)

        self.dropout = nn.Dropout(dropout)
        self.lin1 =Linear (num_node_features//2,num_node_features//4)
        self.lin2 =Linear (num_node_features//4,num_node_features//16)
        self.lin3 =Linear (num_node_features//16,num_node_features//32)

        self.lin1_main =Linear (num_node_features,num_node_features//2)
        self.lin2_main =Linear (num_node_features,num_node_features//4)
        self.lin3_main =Linear (num_node_features,num_node_features//16)
        self.lin4_main =Linear (num_node_features,num_node_features//32)

        self.classifier =Linear (num_node_features//32,1)

    def forward(self, x, edge_index):
        row_net, col_net = edge_index
        predicted = torch.empty(len(row_net),x.shape[1]//2).to(device)
        predicted2 = torch.empty(len(row_net),x.shape[1]//4).to(device)
        predicted3 = torch.empty(len(row_net),x.shape[1]//16).to(device)
        predicted4 = torch.empty(len(row_net),x.shape[1]//32).to(device)

        x_main = x.detach().clone()
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x_1 = self.lin1_main (x_main)
        for i in range(len(row_net)):
            predicted[i] =torch.sum(torch.stack([x[row_net[i]], x[col_net[i]],x_1[i]]), dim=0)      
        predicted = self.lin1 (predicted)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x_2 = self.lin2_main (x_main)
        for i in range(len(row_net)):
            predicted2[i] =torch.sum(torch.stack([x[row_net[i]], x[col_net[i]],predicted[i],x_2[i]]), dim=0)
        predicted = self.lin2 (predicted2)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x_3 = self.lin3_main (x_main)
        for i in range(len(row_net)):
            predicted3[i] =torch.sum(torch.stack([x[row_net[i]], x[col_net[i]],predicted[i],x_3[i]]), dim=0)
        predicted = self.lin3 (predicted3)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x_4 = self.lin4_main (x_main)
        for i in range(len(row_net)):
            predicted4[i] =torch.sum(torch.stack([x[row_net[i]], x[col_net[i]],predicted[i],x_4[i]]), dim=0)
         
        predicted = self.classifier (predicted4)
        predicted = predicted.squeeze(dim=1)
        

        return predicted
        
        
      
features_dimensions = len(preds_file[list(preds_file.keys())[0]][0])
model = Net(features_dimensions).to(device)
model_state = torch.load('model.pt')
model.load_state_dict(model_state['state_dict'])
best_thr = 0.5
model.eval()
document_f1_scores = []
counter_for_print =0

# Prepare data for classification
for id_doc_s in preds_file.keys():
      if counter_for_print % 1000==0:
          print ( counter_for_print,flush=True)
      out =[]
      feature_list = []
      row_col_edges =[]
      filename = f"solution-problem-{id_doc_s}.json"
      filepath = os.path.join(output_path, filename)

      for i in range(len(preds_file[id_doc_s])):
          feature_list.append (preds_file[id_doc_s][i])

      only_features_t = torch.tensor(feature_list).to(device)
      only_features_t = torch.tensor(feature_list).to(device)
      all_edges = [(i, i+1) for i in range(len(preds_file[id_doc_s])-1)]
      row_only = [x for (x,y) in all_edges ]
      col_only = [y for (x,y) in all_edges ]
      row_col_edges.append(row_only)
      row_col_edges.append(col_only)
      row_col_edges_t = torch.tensor(row_col_edges).to(device)

      out = model(only_features_t,row_col_edges_t)
      out =  torch.sigmoid(out)
      out_train = (out>= best_thr).float()

      x_np= out_train.detach().cpu().numpy()
      data = {"changes": [int(item) for item in x_np]}
      with open(filepath, 'w') as f:
            json.dump(data, f)
      counter_for_print +=1