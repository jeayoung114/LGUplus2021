"""
Collaborative Denoising Auto-Encoders for Top-N Recommender Systems, 
Yao Wu et al.,
WSDM 2016.
"""
import os
import math
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CDAE_implicit(torch.nn.Module):
    def __init__(self, train, valid, num_epochs, hidden_dim, learning_rate, reg_lambda, dropout, device, activation="tanh"):
        super().__init__()
        self.train_mat = train
        self.valid_mat = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]

        self.num_epochs = num_epochs
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.activation = activation
        self.dropout = dropout

        self.device = device

        self.build_graph()


    def build_graph(self):
        # W, W'와 b, b', V 만들고 초기화
        self.enc_w = nn.Parameter(torch.ones(self.num_items, self.hidden_dim))
        self.enc_b = nn.Parameter(torch.ones(self.hidden_dim))
        nn.init.normal_(self.enc_w, 0, 0.01)
        nn.init.normal_(self.enc_b, 0, 0.01)
        self.dec_w = nn.Parameter(torch.ones(self.hidden_dim, self.num_items))
        self.dec_b = nn.Parameter(torch.ones(self.num_items))
        nn.init.normal_(self.dec_w, 0, 0.01)
        nn.init.normal_(self.dec_b, 0, 0.01)
        self.user_embedding = nn.Embedding(self.num_users, self.hidden_dim)

        # 최적화 방법 설정
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.reg_lambda)

        # 모델을 device로 보냄
        self.to(self.device)


    def forward(self, u, x):     
        # 입력의 일부를 제거
        denoised_x = F.dropout(x, self.dropout, training=self.training) 

        # encoder 과정
        if self.activation == 'None':
            h = denoised_x @ self.enc_w + self.enc_b + self.user_embedding(u)
        elif self.activation == 'tanh':
            h = torch.tanh(denoised_x @ self.enc_w + self.enc_b + self.user_embedding(u))
        else:
            h = torch.sigmoid(denoised_x @ self.enc_w + self.enc_b + self.user_embedding(u))

        # decoder 과정
        output = torch.sigmoid(h @ self.dec_w + self.dec_b)
        return output


    def fit(self):
        train_matrix = torch.FloatTensor(self.train_mat).to(self.device)
        batch_idx = np.arange(self.num_users)
        batch_idx = torch.LongTensor(batch_idx).to(self.device)

        for epoch in range(0, self.num_epochs):
            self.train()
            loss = self.train_model_per_batch(batch_idx, train_matrix)
            print('epoch %d  loss = %.4f' % (epoch + 1, loss))

            if torch.isnan(loss):
                print('Loss NAN. Train finish.')
                break

        self.eval()
        with torch.no_grad():
            self.reconstructed = self.forward(batch_idx, train_matrix).detach().cpu().numpy()


    def train_model_per_batch(self, batch_idx, train_matrix):
        # grad 초기화
        self.optimizer.zero_grad()

        # 모델 forwrad
        output = self.forward(batch_idx, train_matrix)

        # loss 구함
        if self.loss_function == 'MSE':
            loss = F.mse_loss(output, train_matrix, reduction='none').sum(1).mean()
        else:
            loss = F.binary_cross_entropy(output, train_matrix, reduction='none').sum(1).mean()

        # 미분
        loss.backward()

        # 최적화
        self.optimizer.step()

        return loss


    def predict(self, user_id, item_ids):
        return self.reconstructed[user_id, item_ids]

        