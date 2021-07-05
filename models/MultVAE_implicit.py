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

        self.enc_dims = [self.num_items] + [self.hidden_dim]
        self.dec_dims = self.enc_dims[::-1]
        self.dims = self.enc_dims + self.dec_dims[1:]

        self.total_anneal_steps = 200000
        self.anneal_cap = 0.2

        self.dropout = dropout
        self.reg = self.reg_lambda
        self.lr = self.learning_rate

        self.eps = 1e-6
        self.anneal = 0.
        self.update_count = 0
        self.device = device

        self.build_graph()


    def build_graph(self):
        self.encoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:])):
            if i == len(self.enc_dims[:-1]) - 1:
                d_out *= 2
            self.encoder.append(nn.Linear(d_in, d_out))
            if i != len(self.enc_dims[:-1]) - 1:
                self.encoder.append(nn.Tanh())

        self.decoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.dec_dims[:-1], self.dec_dims[1:])):
            self.decoder.append(nn.Linear(d_in, d_out))
            if i != len(self.dec_dims[:-1]) - 1:
                self.decoder.append(nn.Tanh())

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)

        # Send model to device (cpu or gpu)
        self.to(self.device)


    def forward(self, x):
        # encoder
        h = F.dropout(F.normalize(x), p=self.dropout, training=self.training)
        for layer in self.encoder:
            h = layer(h)

        # sample
        mu_q = h[:, :self.enc_dims[-1]]
        logvar_q = h[:, self.enc_dims[-1]:]  # log sigmod^2  batch x 200
        std_q = torch.exp(0.5 * logvar_q)  # sigmod batch x 200

        epsilon = torch.zeros_like(std_q).normal_(mean=0, std=0.01)
        sampled_z = mu_q + self.training * epsilon * std_q

        output = sampled_z
        for layer in self.decoder:
            output = layer(output)

        if self.training:
            kl_loss = ((0.5 * (-logvar_q + torch.exp(logvar_q) + torch.pow(mu_q, 2) - 1)).sum(1)).mean()
            return output, kl_loss
        else:
            return output


    def fit(self):
        train_matrix = torch.FloatTensor(self.train_mat).to(self.device)
        batch_idx = np.arange(self.num_users)
        batch_idx = torch.LongTensor(batch_idx).to(self.device)

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


    def train_model_per_batch(self, batch_matrix):
        # zero grad
        self.optimizer.zero_grad()

        # model forwrad
        output, kl_loss = self.forward(batch_matrix)

        # loss        
        ce_loss = -(F.log_softmax(output, 1) * batch_matrix).sum(1).mean()

        loss = ce_loss + kl_loss * self.anneal

        # backward
        loss.backward()

        # step
        self.optimizer.step()

        self.update_count += 1

        return loss


    def predict(self, user_id, item_ids):
        return self.reconstructed[user_id, item_ids]
