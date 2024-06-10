'''
这里是使用一组超参数和通过该组超参数得到的最优阈值来进行8折交叉验证获得8个子模型(有early-stoping)
分别对8折交叉验证的8个训练集应用8组贝叶斯超参数并获得8个子模型

2024.1.17该文件用来测试革兰氏阴性数据集在八折交叉验证，仅包含双向LSTM网络，DBLoss,重采样
但是没有选择阈值和超参数的情况下的效果
'''

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
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True



def check_accuracy(model, X, y): # 检查模型在验证集上的F1值
  
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
    #print('out为{}\n'.format(out))
    pred_label = torch.sigmoid(out.detach().cpu())
    y_pred.append(pred_label)
    y_pred = torch.vstack(y_pred)
    y = torch.stack(y)
    # print('y_pred为{}\n'.format(y_pred))
    # print('y为{}\n'.format(y))
    #best_th, best_mcc = find_best_threshold(pred_label, y)   #获取每个类的最小最佳阈值
    best_threshold = [0.7,0.8,0.5,0.4,0.4]        #默认的阈值
    y_pred = (pred_label >= torch.Tensor(best_threshold)).float().cpu()
 
    macro_f1score = f1_score(y, y_pred, average='macro')
    micro_f1score = f1_score(y, y_pred, average='micro')
    hammingloss = hamming_loss(y, y_pred)
    class_names = ['Cytoplasmic','CytoplasmicMembrane','Extracellular','OuterMembrane','Periplasmic']
    
    classification_result = classification_report(y, y_pred, target_names=class_names)
    print(classification_result)  #输出多标签中每个类的精确率和召回率
    confusion_matrix = multilabel_confusion_matrix(y, y_pred)
    print(confusion_matrix)   #为每个类生成混淆矩阵：列true[0,1],行predict[0,1]
    
  return hammingloss, macro_f1score, micro_f1score, out, classification_result, confusion_matrix  #获得汉明损失和F1分数

def collate_fn(train_data):   #重新设计dataloader数据分法
    train_data.sort(key=lambda data: len(data[0]), reverse=True)  #根据data的长度进行排序，reverse从大到小排序
    data = [sq[0] for sq in train_data]
    label = [sq[1] for sq in train_data]
    data_length = [len(l) for l in data]
    train_data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    label = rnn_utils.pad_sequence(label, batch_first=True, padding_value=0.0)   # 这行代码只是为了把列表变为tensor
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
          learning_rate,  #lstm hidden_size=160时可以有效收敛
          state=None,
          ):
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(len(X_train))
  print(len(X_test))

  #####统计每个类别的样本数量，用于计算classaware重采样的epoch内样本数####
  cls_data_list = []

  num_classes = 5 #类别数量
  num_samples_cls=1    #整个epoch中每次从任意一类中取几个数据
  for i in range(num_classes):
      data_list = []
      for index, data in enumerate(y_train):
          if data[i].item() == 1:
              data_list.append(index)
      cls_data_list.append(data_list)

  #print(cls_data_list)  #统计出每个类别有哪些样本
  class_names = ['Cytoplasmic','CytoplasmicMembrane','Extracellular','OuterMembrane','Periplasmic']
  for index , n in enumerate(cls_data_list):
      print("{}类别的样本数量为{}".format(class_names[index],len(n)))  #输出每个类别的样本数量
#####################################################################
  sampler = ClassAwareSampler(data_list=cls_data_list, num_classes=num_classes, num_samples_cls=num_samples_cls, reduce=2)
  #reduce用于削减epoch的样本数
  train_data = DataLoader(dataset(X_train, y_train), batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
  #optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.0001)   #weight_decay L2正则化
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  criterion = nn.BCEWithLogitsLoss()  


  model = model.to(device)
  state = dict()  #空字典
  state['microf1'] = []
  state['macrof1'] = []
  state['hammingloss'] = []
  state['val_hammingloss'] = []
  state['val_microf1'] = []
  state['val_macrof1'] = []

  best_val = -0.00001    #2023.5.6修改内容，为了防止val_macro_f1score等于0的情况时，不更新state字典
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
      #print('训练集的out为{}\n'.format(out))
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

    y_pred = torch.vstack(y_pred_list)   #将每个batch的预测结果拼接起来
    y_train = torch.vstack(y_train_list)
    print('经过重采样后每类样本共{}个'.format(y_train.shape[0]/10))

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
      for fname in os.listdir(model_dir):  #返回指定的文件夹包含的文件或文件夹的名字的列表
            if fname.startswith('_'.join(['fold',str(val_fold_number-1)])):  #返回一个布尔值，判断字符串fname是否以指定的子字符串开头
                #.join表示将字符串中的元素以指定的字符'_'连接生成一个新的字符串
                os.remove(os.path.join(model_dir, fname))  #删除指定路径的文件
      torch.save(model.state_dict(), os.path.join(model_dir, '_'.join(['fold',str(val_fold_number-1)])+'_'.join([prefix,'epoch'])+str(epoch)+'para')) #保存模型参数
    else:
        epochs_without_improvement += 1
    print("第{}折交叉验证,验证集最好F1为{},epoch为{}".format(val_fold_number-1,state['best_val'],state['best_epoch']))
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
    #json.dump(d, f)  #将dict类型的数据转换成json格式的数据

    if epochs_without_improvement > 10:
        break
  
  writer.close()
  
  return running_loss, state['best_train'], state['best_val']


seed_torch()
writer = SummaryWriter("/root/tf-logs")   

###########直接将验证集和训练集分割好了############

train_data = pd.read_pickle('/root/autodl-tmp/原核数据集/革兰氏阴性数据集/G-训练集/fold7/train_fold7.pkl')
val_data = pd.read_pickle('/root/autodl-tmp/原核数据集/革兰氏阴性数据集/G-训练集/fold7/val_fold7.pkl')
val_fold_number = 8  #范围1-8
################################################
train_embeddings = train_data.embeddings.values
val_embeddings = val_data.embeddings.values


train_labels = [torch.Tensor(np.array(label)) for label in train_data.onehot_label.values]
val_labels = [torch.Tensor(np.array(label)) for label in val_data.onehot_label.values]

preprocessed_text_train = [torch.Tensor(np.array(embeddings)) for embeddings in train_embeddings]  #列表中每个元素torch.Size([500, 25])
preprocessed_text_val = [torch.Tensor(np.array(embeddings)) for embeddings in val_embeddings]

train_X = preprocessed_text_train
val_X = preprocessed_text_val

source_dir = '/root/autodl-tmp/原核模型/革兰氏阴性菌/LSTM模型'
prefix = 'train_new' 


print('Train label shape {}'.format(len(train_labels)))
print('Test label shape {}'.format(len(val_labels)))
#print(PSSM_test.shape)


model = Only_LSTM(input_size_LSTM = 1024, hidden_size = 397 , slope=0.01, dropout=0.225669)
val_macro_f1score = 0
train(model, train_X, val_X, train_labels, val_labels,total_epoch=80,batch_size=64,learning_rate=0.000355, state=None)



