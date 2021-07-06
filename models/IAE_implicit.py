import os
import math
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class IAE_implicit(torch.nn.Module):
    def __init__(self, train, valid, num_epochs, hidden_dim, learning_rate, reg_lambda, device, activation="tanh", loss="CE"):
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
        self.loss_function = loss

        self.device = device

        self.build_graph()


    def build_graph(self):
        # NN layers
        self.encoder = nn.Linear(self.num_users, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.num_users)
        nn.init.normal_(self.encoder.weight, 0, 0.01)
        nn.init.normal_(self.decoder.weight, 0, 0.01)

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.reg_lambda)

        # Send model to device (cpu or gpu)
        self.to(self.device)


    def forward(self, x):
        if self.activation == 'None':
            h = self.encoder(x)
        elif self.activation == 'tanh':
            h = torch.tanh(self.encoder(x))
        else:
            h = torch.sigmoid(self.encoder(x))

        output = torch.sigmoid(self.decoder(h))
        return output


    def fit(self):
        train_matrix = torch.FloatTensor(self.train_mat.T).to(self.device)

        for epoch in range(0, self.num_epochs):
            self.train()
            
            loss = self.train_model_per_batch(train_matrix)
            print('epoch %d  loss = %.4f' % (epoch + 1, loss))

            if torch.isnan(loss):
                print('Loss NAN. Train finish.')
                break

        self.eval()
        with torch.no_grad():
            self.reconstructed = self.forward(train_matrix).detach().cpu().numpy()
            self.reconstructed = self.reconstructed.T
            

    def train_model_per_batch(self, batch_matrix):
        # zero grad
        self.optimizer.zero_grad()

        # model forwrad
        output = self.forward(batch_matrix)

        # loss
        if self.loss_function == 'MSE':
            loss = F.mse_loss(output, batch_matrix, reduction='none').sum(1).mean()
        else:
            loss = F.binary_cross_entropy(output, batch_matrix, reduction='none').sum(1).mean()

        # backward
        loss.backward()
        
        # step
        self.optimizer.step()
        return loss



    def predict(self, user_id, item_ids):
        return self.reconstructed[user_id, item_ids]


    # def predict(self, user_ids, eval_pos_matrix, eval_items=None):
    #     batch_eval_pos = eval_pos_matrix[user_ids]
    #     # eval_output = np.zeros(batch_eval_pos.shape, dtype=np.float32)
    #     with torch.no_grad():
    #         eval_matrix = torch.FloatTensor(batch_eval_pos.toarray()).to(self.device)
    #         eval_output = self.forward(eval_matrix).detach().cpu().numpy()

    #         if eval_items is not None:
    #             eval_output[np.logical_not(eval_items)]=float('-inf')
    #         else:
    #             eval_output[batch_eval_pos.nonzero()] = float('-inf')

    #         return eval_output

