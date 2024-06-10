from lib2to3.pgen2 import token
import torch
from torch import nn
from torchsummary import summary
import torchvision
from torchstat import stat
import torchvision.models as models

# Using static embedding
class Only_LSTM(nn.Module):
  def __init__(self, input_size_LSTM, hidden_size,slope=0.01, dropout=0.2):
    super(Only_LSTM, self).__init__()

    self.rnn = nn.LSTM(input_size_LSTM,
                        hidden_size,
                        num_layers=2,     #LSTM层数2022.9.9加入
                        batch_first=True,
                        bidirectional=True)
    
    self.fc = nn.Linear(hidden_size * 2, 5)  #全连接层,输出五个类别
  
    self.dropout = nn.Dropout(dropout)

  def forward(self, features):
    out, (hidden, cell) = self.rnn(features)
    #hidden_size维度(num_layers * num_directions, batch, hidden_size)，这里是[2,250,160]
    out = torch.cat([hidden[-2, :, :], hidden[-1, :, :]], dim=1)
    out = self.dropout(out)           
    #print(out,out.shape)    #[250,320]
    out = self.fc(out)   #全连接层
    #print('最终输出形状为:{}'.format(out.shape))   # [250,10]

    return out


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ =='__main__':
  adjacency = torch.randn(10,10)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
  model = Only_LSTM(input_size_LSTM = 1024, hidden_size = 336 ).to(device)
  for name,parameters in model.named_parameters():
        print(name,':',parameters.size())
  a = get_parameter_number(model)
  print(a)
