"""
RNN/LSTM
ZihanGan 2021.12.11
"""

import sys
sys.path.append("C:\\anaconda2\\pkgs\\tensorflow-base-2.3.0-eigen_py37h17acbac_0\\Lib\\site-packages\\tensorflow")
sys.path.append("C:\\anaconda2\\pkgs\\tensorflow-base-2.3.0-eigen_py37h17acbac_0\\Lib\\site-packages")
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np
#matplotlib环境问题，运行前注释
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from matplotlib import pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import *
import torch.optim as optim
#import time
#time_start=time.time()
#time_end=time.time()
#print('pockets time',time_end-time_start)

class LSTM(nn.Module):
    def __init__(self, input_dim,output_dim,embed_dim=128,hidden_dim=128):
        super(LSTM, self).__init__()
        self.lr=0.003
        self.epoches=100
        self.embedding=nn.Embedding(input_dim,embed_dim)
        self.drop=nn.Dropout(0.2)
        self.encoder=nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        self.decoder1=nn.Linear(hidden_dim*2,64)
        self.decoder2=nn.Linear(64,16)
        self.decoder3=nn.Linear(16,output_dim)
        self.optimizer=optim.Adam(self.parameters(),self.lr)
        self.loss=nn.CrossEntropyLoss()
        self.losses=[]
        self.recall=[]
        self.precision=[]
        #self.optimizer.param_groups
    def forward(self, x):
        x1=self.embedding(x)
        x2=self.drop(x1)
        x3,_=self.encoder(x2)
        #x3=self.drop(x3)
        x4=self.decoder1(x3)
        x4=F.avg_pool2d(x4,(x.shape[1], 1)).squeeze()
        x5=self.decoder2(x4)
        x6=self.decoder3(x5)
        #x3=torch.argmax(x2,2)
        return x6
    def testify(self,test):
        #self.eval()
        p=0
        r=0
        l=len(test.dataset)
        for i, (x, y) in enumerate(test):
            with torch.no_grad():
                x, y = x.to(device), y.to(device)
                y_hat=self.forward(x)
            y_hat=y_hat.max(-1, keepdim=True)[1]
            y_hat=((y_hat.view_as(y))==y)
            p+=y_hat.sum().item()
            r+=sum([y_hat[i].item()&y[i].item() for i in range(len(y))])
        return 2*r/l,p/l
        # y_hat=[]
        # l=len(test.dataset)
        # for i,(x,y) in enumerate(test):
        #     y_hat=np.hstack([y_hat,np.array(torch.argmax(self.forward(x),0)).T])
        # r,p=y_hat==y,y==1
        # precision=sum(r)/l
        # recall=sum([r[i]&p[i] for i in range(l)])/sum(p)
        # return recall,precision
    def render(self):
        plt.plot(self.losses)
        plt.xlabel('epoch',fontsize=14)
        plt.ylabel('loss',fontsize=14)
        plt.title('LCE',fontsize=24)
        plt.show()
        return
    def train(self,train_data,test_data):
        for _ in range(self.epoches):
            L=0
            for i,(x,y) in enumerate(train_data):
                #x,y=x.to(DEVICE),y.to(DEVICE)
                self.optimizer.zero_grad()
                #l=self.loss(torch.argmax(self.forward(x[j]),1),y[j])
                l=self.loss(self.forward(x),y)
                L+=l
                l.backward()
                self.optimizer.step()
                print(i,':',L.item())
            self.losses.append(float(L))
                #r,p=self.testify(test_data)
                #self.recall.append(r)
                #self.precision.append(p)
                #print("epoch=",i+1,":recall=",r,"/Precision=",p,"/loss=",float(l))
        return

class RNN(nn.Module):
    def __init__(self, input_dim,output_dim,hidden_dim=128,hidden_layers=2):
        super(RNN, self).__init__()
        self.lr=0.003
        self.epoches=10
        self.encoder=nn.RNN(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=hidden_layers)
        self.decoder1=nn.Linear(hidden_dim,64)
        self.decoder2=nn.Linear(64,32)
        self.decoder3=nn.Linear(32,output_dim)
        self.optimizer=optim.Adam(self.parameters(),self.lr)
        self.loss=nn.CrossEntropyLoss()
        self.losses=[]
        self.recall=[]
        self.precision=[]
        #self.optimizer.param_groups
    def forward(self, x):
        x1,_=self.encoder(x)
        x2=self.decoder1(x1)
        x3=self.decoder2(x2)
        x4=self.decoder3(x3)
        x4=torch.squeeze(x4,1)
        #x3=torch.argmax(x2,2)
        return x4
    def testify(self,x,y):
        l=x.shape[0]*x.shape[1]
        y_hat=[]
        for i in range(len(x)):
            #y_hat=np.hstack([y_hat,np.array(self.forward(x[i])>0.5).T[0]])
            #print(y_hat.shape)
            y_hat=np.hstack([y_hat,np.array(torch.argmax(self.forward(x[i]),1)).T])
        r,p=y_hat==y,y==1
        precision=sum(r)/l
        recall=sum([r[i]&p[i] for i in range(l)])/sum(p)
        return recall,precision
    def render(self,flag):
        if flag=='loss':
            plt.plot(self.losses)
            plt.xlabel('epoch',fontsize=14)
            plt.ylabel('loss',fontsize=14)
            plt.title(flag,fontsize=24)
        elif flag=='recall':
            plt.plot(self.recall)
            plt.xlabel('epoch',fontsize=14)
            plt.ylabel('precents',fontsize=14)
            plt.title('recall',fontsize=24)
        elif flag=='precision':
            plt.plot(self.precision)
            plt.xlabel('epoch',fontsize=14)
            plt.ylabel('precents',fontsize=14)
            plt.title('precision',fontsize=24)
        plt.show()
        return
    def train(self,x,y,x_t,y_t):
        y=torch.LongTensor(y)
        b=len(x)
        y=y.reshape(100,250)
        for i in range(self.epoches):
            L=0
            for j in range(b):
                self.optimizer.zero_grad()
                l=self.loss(self.forward(x[j]),y[j])
                #l=self.loss(torch.argmax(self.forward(x[j]),1),y[j])
                L+=l
                l.backward(retain_graph=True)
                self.optimizer.step()
            self.losses.append(float(L))
            r,p=self.testify(x_t,y_t)
            self.recall.append(r)
            self.precision.append(p)
            print("epoch=",i+1,":recall=",r,"/Precision=",p,"/loss=",float(l))
        return

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#词典按词频大小排列，取字典前10000个单词即选择前10000最常见单词
(x,y),(x_t,y_t)=imdb.load_data(num_words=10000)
#转换为张量
#for i in range(len(x)):
#    x[i],x_t[i]=torch.Tensor(x[i]),torch.Tensor(x_t[i])
X=np.hstack([x,x_t])
#补0
X=sequence.pad_sequences(X,value=0,padding='post',maxlen=50)
#X=nn.utils.rnn.pad_sequence(X,batch_first=True,padding_value=0)
x,x_t=X[:25000],X[25000:]
x=torch.tensor(x.reshape([100,len(x)//100,1,x[0].shape[0]])).float()
x_t=torch.tensor(x_t.reshape([100,len(x_t)//100,1,x_t[0].shape[0]])).float()
#二分类
m=RNN(x[0][0][0].shape[0],2)
m.train(x,y,x_t,y_t)
m.render('loss')
m.render('recall')
m.render('precision')
#LSTM
cut=256
#如果加入embedding，需要进一步清洗
(x,y),(x_t,y_t)=imdb.load_data(num_words=cut)
x=sequence.pad_sequences(x,value=0,padding='post',maxlen=cut)
x_t=sequence.pad_sequences(x_t,value=0,padding='post',maxlen=cut)
train_data=DataLoader(TensorDataset(torch.LongTensor(x),torch.LongTensor(y)),batch_size=256)
test_data=DataLoader(TensorDataset(torch.LongTensor(x_t),torch.LongTensor(y_t)),batch_size=500)
m=LSTM(cut,2)
m.train(train_data,test_data)
r,p=m.testify(test_data)
m.render()