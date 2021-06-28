# import os
# import math
# from time import time
# from numpy import dot
# from numpy.linalg import norm

import numpy as np
# import scipy.sparse as sp

import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data as data
# import torch.backends.cudnn as cudnn


class WMF_GD_implicit_model(torch.nn.Module):
    def __init__(self, num_users, num_items, n_features):
        super().__init__()
        self.user_factors = torch.nn.Embedding(num_users, n_features, sparse=False)
        self.item_factors = torch.nn.Embedding(num_items, n_features, sparse=False)
        
        torch.nn.init.normal_(self.user_factors.weight, std=0.01)
        torch.nn.init.normal_(self.item_factors.weight, std=0.01)

    def forward(self):
        return torch.matmul(self.user_factors.weight, self.item_factors.weight.T)

class WMF_GD_implicit():
    def __init__(self, train, valid, n_features=20, learning_rate = 1e-2, reg_lambda =0.1, num_epochs = 550):
        self.train = train
        self.valid = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]
        self.num_epcohs = num_epochs
        self.n_features = n_features

        self.y = np.zeros_like(self.train)
        for i, row in enumerate(self.train):
            self.y[i, np.where(row > 0.5)[0]] = 1.0
            self.y[i, np.where(row < 0.5)[0]] = 0.5

        self.model = WMF_GD_implicit_model(self.num_users, self.num_items, self.n_features)#.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=reg_lambda)


    def weighted_mse_loss(self, y, target, predict):
        return (y * (target - predict) ** 2).sum()


    def fit(self):
        ratings = torch.FloatTensor(self.train)#.cuda()
        weights = torch.FloatTensor(self.y)#.cuda()

        # U와 V를 업데이트 함.
        for epoch in range(self.num_epcohs):
            self.optimizer.zero_grad()

            # 예측
            prediction = self.model.forward()
            loss = self.weighted_mse_loss(weights, ratings, prediction)

            # Backpropagate
            loss.backward()

            # Update the parameters
            self.optimizer.step()

            print("epoch %d, loss: %f"%(epoch, loss))

        with torch.no_grad():
            self.reconstructed = self.model.forward().cpu().numpy()


    def predict(self, user_id, item_ids):
        return self.reconstructed[user_id, item_ids]
