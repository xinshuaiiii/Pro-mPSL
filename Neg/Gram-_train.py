import json
import math
import os
import random
import pandas as pd
from gensim.models import KeyedVectors
import numpy as np
import torch
from torch import optim
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import hamming_loss, f1_score, multilabel_confusion_matrix
import pickle
import time
import copy
from Gram_train_model import Only_LSTM
from torchvision import models
from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report
import torch.nn.utils.rnn as rnn_utils
from tqdm import trange
from sampler import ClassAwareSampler
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter



def seed_torch(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True



def check_accuracy(model, X, y):
  
  model.eval()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #print(X[0].shape[0])
  test_seq_len = [x.shape[0] for x in X]

  X = rnn_utils.pad_sequence(X, batch_first=True, padding_value=0.0)
  X = rnn_utils.pack_padded_sequence(X, test_seq_len, batch_first=True, enforce_sorted=False)
  #X = torch.stack(preprocessed_text_test)   #X=list
  X = X.to(device)
  y_pred = []

  with torch.no_grad():
    out = model(X)
    #print('out {}\n'.format(out))
    pred_label = torch.sigmoid(out.detach().cpu())
    y_pred.append(pred_label)
    y_pred = torch.vstack(y_pred)
    y = torch.stack(y)
    # print('y_pred为{}\n'.format(y_pred))
    # print('y {}\n'.format(y))
    #best_th, best_mcc = find_best_threshold(pred_label, y)   
    best_threshold = [0.7,0.8,0.5,0.4,0.4]        
    y_pred = (pred_label >= torch.Tensor(best_threshold)).float().cpu()
 
    macro_f1score = f1_score(y, y_pred, average='macro')
    micro_f1score = f1_score(y, y_pred, average='micro')
    hammingloss = hamming_loss(y, y_pred)
    class_names = ['Cytoplasmic','CytoplasmicMembrane','Extracellular','OuterMembrane','Periplasmic']
    
    classification_result = classification_report(y, y_pred, target_names=class_names)
    print(classification_result)  
    confusion_matrix = multilabel_confusion_matrix(y, y_pred)
    print(confusion_matrix)  
    
  return hammingloss, macro_f1score, micro_f1score, out, classification_result, confusion_matrix 

def collate_fn(train_data):  
    train_data.sort(key=lambda data: len(data[0]), reverse=True)  
    data = [sq[0] for sq in train_data]
    label = [sq[1] for sq in train_data]
    data_length = [len(l) for l in data]
    train_data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    label = rnn_utils.pad_sequence(label, batch_first=True, padding_value=0.0)   
    return train_data, label, data_length




class dataset(Dataset):
  def __init__(self, x, y):
    self.x  = x
    self.y = y

  def __len__(self):
    return len(self.x)
  
  def __getitem__(self, idx):
    tuple_ = (self.x[idx], self.y[idx])
    return tuple_



def train(model,
          X_train,
          X_test,
          y_train,
          y_test,
          total_epoch,
          batch_size,
          learning_rate,  
          state=None,
          ):
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(len(X_train))
  print(len(X_test))


  cls_data_list = []

  num_classes = 5 
  num_samples_cls=1    
  for i in range(num_classes):
      data_list = []
      for index, data in enumerate(y_train):
          if data[i].item() == 1:
              data_list.append(index)
      cls_data_list.append(data_list)

  #print(cls_data_list)  
  class_names = ['Cytoplasmic','CytoplasmicMembrane','Extracellular','OuterMembrane','Periplasmic']
  for index , n in enumerate(cls_data_list):
      print("numbesr of samples for category {} is{}".format(class_names[index],len(n)))  
#####################################################################
  sampler = ClassAwareSampler(data_list=cls_data_list, num_classes=num_classes, num_samples_cls=num_samples_cls, reduce=2)

  train_data = DataLoader(dataset(X_train, y_train), batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
  #optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.0001)   
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  criterion = nn.BCEWithLogitsLoss()  


  model = model.to(device)
  state = dict()  
  state['microf1'] = []
  state['macrof1'] = []
  state['hammingloss'] = []
  state['val_hammingloss'] = []
  state['val_microf1'] = []
  state['val_macrof1'] = []

  best_val = -0.00001    
  epochs_without_improvement = 0
  best_threshold = [0.7,0.8,0.5,0.4,0.4]
  for epoch in trange(0, total_epoch):

    running_loss = 0
    y_pred_list = []
    y_train_list = []
    epoch_time = 0
    model.train()


    for index, (X, y, seq_len) in enumerate(train_data):
      
      t = time.time()

      X = rnn_utils.pack_padded_sequence(X, seq_len, batch_first=True)
      out = model(X.to(device))
      #print('train out {}\n'.format(out))
      loss = criterion(out, y.to(device))
      #backward
      optimizer.zero_grad()
      loss.backward()
      clip_grad_norm_(model.parameters(), max_norm=10)
      #update
      optimizer.step()

      epoch_time += time.time() - t
      pred_train_label = torch.sigmoid(out.detach().cpu())
      y_pred = (pred_train_label >= torch.Tensor(best_threshold)).float().cpu()
      y_pred_list.append(y_pred)
      y_train_list.append(y)
      running_loss += loss.item()

    #gpu_tracker.track()

    y_pred = torch.vstack(y_pred_list)   
    y_train = torch.vstack(y_train_list)
    print('After resampling, there are {} samples in each category'.format(y_train.shape[0]/10))

    macro_f1score = f1_score(y_train, y_pred, average='macro')
    micro_f1score = f1_score(y_train, y_pred, average='micro')
    hammingloss = hamming_loss(y_train, y_pred)
    val_hamming, val_macro_f1score, val_micro_f1score, y_test_out, classification_result, confusion_matrix = check_accuracy(model, X_test, y_test)


    state['microf1'].append(micro_f1score)
    state['macrof1'].append(macro_f1score)
    state['hammingloss'].append(hammingloss)
    state['val_macrof1'].append(val_macro_f1score)
    state['val_microf1'].append(val_micro_f1score)
    state['val_hammingloss'].append(val_hamming)
    state['optimizer'] = optimizer.state_dict()
    state['last_model'] = model.state_dict()
    
    if(best_val * 0.995 < val_macro_f1score):
      state['model_best_val'] = copy.deepcopy(model.state_dict())
      best_val = val_macro_f1score
      best_confusion_matrix = confusion_matrix
      best_classification_result = classification_result
      state['best_train'] = macro_f1score
      state['best_val'] = best_val
      state['best_epoch'] = epoch
      epochs_without_improvement = 0
      model_dir = os.path.join(source_dir, 'val_models_fold_2_reduce_2')
      for fname in os.listdir(model_dir): 
            if fname.startswith('_'.join(['fold',str(val_fold_number-1)])):  
               
                os.remove(os.path.join(model_dir, fname))  
      torch.save(model.state_dict(), os.path.join(model_dir, '_'.join(['fold',str(val_fold_number-1)])+'_'.join([prefix,'epoch'])+str(epoch)+'para')) 
    else:
        epochs_without_improvement += 1
    print("{}K-Fold,vai F1{},epoch {}".format(val_fold_number-1,state['best_val'],state['best_epoch']))
    print('epoch:{} loss:{:.5f} hamming_loss:{:.5f} macro_f1score:{:.5f} micro_f1score:{:.5f} val_hamming_loss:{:.5f} val_macro_f1score:{:.5f} val_micro_f1score:{:.5f}'.
          format(epoch, running_loss, hammingloss, macro_f1score, micro_f1score, val_hamming, val_macro_f1score, val_micro_f1score))
    writer.add_scalar("loss"+str(val_fold_number-1),running_loss,epoch)
    writer.add_scalars("hamming_loss train/test",{"hammingloss_train"+str(val_fold_number-1):hammingloss,
                                                  "hammingloss_test"+str(val_fold_number-1):val_hamming},epoch)
    writer.add_scalars("macro_f1score train/test",{"macro_f1score_train"+str(val_fold_number-1):macro_f1score,
                                                  "macro_f1score_test"+str(val_fold_number-1):val_macro_f1score},epoch)
    writer.add_scalars("micro_f1score train/test",{"micro_f1score_train"+str(val_fold_number-1):micro_f1score,
                                                  "micro_f1score_test"+str(val_fold_number-1):val_micro_f1score},epoch)

    log_dir = os.path.join(source_dir, 'val_logs')
    # Log all results in validation set with different thresholds
    #with open(os.path.join(log_dir, '_'.join([prefix,'epoch'])+str(epoch+1)+'_'.join(['fold',str(val_fold_number-1)])+'.json'),'w') as f:
    d = {}
    d["f1_accuracy_default"] =  state['best_val']
    d["confusion_matrix"] = best_confusion_matrix
    d["classification_result"] = best_classification_result
    np.save(os.path.join(log_dir, '_'.join([prefix,'epoch'])+str(epoch)+'_'.join(['fold',str(val_fold_number-1)])+'.npy'), d)
    #json.dump(d, f)  

    if epochs_without_improvement > 10:
        break
  
  writer.close()
  
  return running_loss, state['best_train'], state['best_val']


seed_torch()
writer = SummaryWriter("/root/tf-logs")   



train_data = pd.read_pickle('train.pkl')
val_data = pd.read_pickle('val.pkl')
val_fold_number = 8  

train_embeddings = train_data.embeddings.values
val_embeddings = val_data.embeddings.values


train_labels = [torch.Tensor(np.array(label)) for label in train_data.onehot_label.values]
val_labels = [torch.Tensor(np.array(label)) for label in val_data.onehot_label.values]

preprocessed_text_train = [torch.Tensor(np.array(embeddings)) for embeddings in train_embeddings]  
preprocessed_text_val = [torch.Tensor(np.array(embeddings)) for embeddings in val_embeddings]

train_X = preprocessed_text_train
val_X = preprocessed_text_val

source_dir = 'your dir'
prefix = 'train_new' 


print('Train label shape {}'.format(len(train_labels)))
print('Test label shape {}'.format(len(val_labels)))
#print(PSSM_test.shape)


model = Only_LSTM(input_size_LSTM = 1024, hidden_size = 397 , slope=0.01, dropout=0.225669)
val_macro_f1score = 0
train(model, train_X, val_X, train_labels, val_labels,total_epoch=80,batch_size=64,learning_rate=0.000355, state=None)



