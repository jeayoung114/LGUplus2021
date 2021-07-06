"""
Variational Autoencoders for Collaborative Filtering, 
Dawen Liang et al.,
WWW 2018.
"""
import os
import math
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultVAE_implicit(torch.nn.Module):
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

        self.total_anneal_steps = 200000
        self.anneal_cap = 0.2

        self.dropout = dropout

        self.update_count = 0
        self.device = device

        self.build_graph()


    def build_graph(self):
        # W, W'와 b, b'만들기
        self.enc_w = nn.Parameter(torch.ones(self.num_items, self.hidden_dim * 2))
        self.enc_b = nn.Parameter(torch.ones(self.hidden_dim * 2))
        nn.init.xavier_uniform_(self.enc_w)
        nn.init.normal_(self.enc_b, 0, 0.001)

        self.dec_w = nn.Parameter(torch.ones(self.hidden_dim, self.num_items))
        self.dec_b = nn.Parameter(torch.ones(self.num_items))
        nn.init.xavier_uniform_(self.dec_w)
        nn.init.normal_(self.dec_b, 0, 0.001)

        # 최적화 방법 설정
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.reg_lambda)

        # 모델을 device로 보냄
        self.to(self.device)


    def forward(self, x):
        # 입력의 일부를 제거
        denoised_x = F.dropout(F.normalize(x), self.dropout, training=self.training)

        # encoder 과정
        h = denoised_x @ self.enc_w + self.enc_b

        # 잠재인수 z 만들기
        mu_q = h[:, :self.hidden_dim]
        logvar_q = h[:, self.hidden_dim:]
        std_q = torch.exp(0.5 * logvar_q)

        epsilon = torch.zeros_like(std_q).normal_(mean=0, std=0.01)
        sampled_z = mu_q + self.training * epsilon * std_q

        # decoder 과정
        output = sampled_z @ self.dec_w + self.dec_b

        # KL loss
        if self.training:
            kl_loss = ((0.5 * (-logvar_q + torch.exp(logvar_q) + torch.pow(mu_q, 2) - 1)).sum(1)).mean()
            return output, kl_loss
        else:
            return output


    def fit(self):
        train_matrix = torch.FloatTensor(self.train_mat).to(self.device)

        for epoch in range(0, self.num_epochs):
            self.train()

            if self.total_anneal_steps > 0:
                self.anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
            else:
                self.anneal = self.anneal_cap

            loss = self.train_model_per_batch(train_matrix)
            print('epoch %d  loss = %.4f' % (epoch + 1, loss))

            if torch.isnan(loss):
                print('Loss NAN. Train finish.')
                break

        self.eval()
        with torch.no_grad():
            self.reconstructed = self.forward(train_matrix).detach().cpu().numpy()


    def train_model_per_batch(self, train_matrix):
        # grad 초기화
        self.optimizer.zero_grad()

        # 모델 forwrad
        output, kl_loss = self.forward(train_matrix)

        # loss 구함
        ce_loss = -(F.log_softmax(output, 1) * train_matrix).sum(1).mean()
        loss = ce_loss + kl_loss * self.anneal

        # 미분
        loss.backward()

        # 최적화
        self.optimizer.step()

        self.update_count += 1

        return loss


    def predict(self, user_id, item_ids):
        return self.reconstructed[user_id, item_ids]
