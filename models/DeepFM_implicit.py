import os
import math
from time import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader

# https://github.com/rixwew/pytorch-fm/blob/master/torchfm/model/dfm.py을 참고했습니다.
class DeepFM_implicit(torch.nn.Module):
    def __init__(self, train_data, train_label, valid_data, valid_label, field_dims, embed_dim, mlp_dims, dropout,
                 num_epochs, early_stop_trial, learning_rate, reg_lambda, batch_size, device):
        super().__init__()

        self.train_data = train_data
        self.train_label = train_label
        self.valid_data = valid_data
        self.valid_label = valid_label
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout

        self.num_epochs = num_epochs
        self.early_stop_trial = early_stop_trial
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.batch_size = batch_size

        self.device = device

        self.build_graph()

    def build_graph(self):
        # 편향 및 단일 필드 계산을 위한 모듈 선언
        self.linear = FeaturesLinear(self.field_dims)
        # 두 필드 간의 계산을 위한 모듈 선언
        self.embedding = FeaturesEmbedding(self.field_dims, self.embed_dim)
        self.fm = FactorizationMachine()
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, self.dropout)

        # Loss 설정
        self.criterion = nn.BCELoss()
        # 최적화 방법 설정
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.reg_lambda)

        # 모델을 device로 보냄
        self.to(self.device)

    def forward(self, x):
        # x: (batch_size, num_fields) -> embed_x: ((batch_size, num_fields, embed_dim)
        embed_x = self.embedding(x)

        # FM과 동일(first + second) 
        FM = self.linear(x) + self.fm(embed_x)

        # Wide & Deep에서 Deep 부분만 (Wide는 FM과 중복됨)
        Deep = self.mlp(embed_x.view(-1, self.embed_output_dim))
        x = FM + Deep
        output = torch.sigmoid(x.squeeze(1))
        return output

    def fit(self):
        train_loader = DataLoader(range(self.train_data.shape[0]), batch_size=self.batch_size, shuffle=True)

        best_AUC = 0
        num_trials = 0
        for epoch in range(1, self.num_epochs+1):
            # Train
            self.train()
            for b, batch_idxes in enumerate(train_loader):
                batch_data = torch.tensor(self.train_data[batch_idxes], dtype=torch.long, device=self.device)
                batch_labels = torch.tensor(self.train_label[batch_idxes], dtype=torch.float, device=self.device)

                loss = self.train_model_per_batch(batch_data, batch_labels)

            # Valid
            self.eval()
            pred_array = self.predict(self.valid_data)
            AUC = roc_auc_score(self.valid_label, pred_array)
            logloss = log_loss(self.valid_label, pred_array)

            if AUC > best_AUC:
                best_AUC = AUC
                torch.save(self.state_dict(), f"saves/{self.__class__.__name__}_best_model.pt")
                num_trials = 0
            else:
                num_trials += 1
            
            if num_trials >= self.early_stop_trial and self.early_stop_trial>0:
                print(f'Early stop at epoch:{epoch}')
                self.restore()
                break
            
            print(f'epoch {epoch} train_loss = {loss:.4f} valid_AUC = {AUC:.4f} valid_log_loss = {logloss:.4f}')
        return

    def train_model_per_batch(self, batch_data, batch_labels):
        # zero grad
        self.optimizer.zero_grad()

        # model forwrad
        logits = self.forward(batch_data)

        # backward
        loss = self.criterion(logits, batch_labels)
        loss.backward()

        # step
        self.optimizer.step()

        return loss

    def predict(self, pred_data):
        self.eval()

        pred_data_loader = DataLoader(range(pred_data.shape[0]), batch_size=self.batch_size, shuffle=False)

        pred_array = np.zeros(pred_data.shape[0])
        for b, batch_idxes in enumerate(pred_data_loader):
            batch_data = torch.tensor(pred_data[batch_idxes], dtype=torch.long, device=self.device)
            with torch.no_grad():
                pred_array[batch_idxes] = self.forward(batch_data).cpu().numpy()

        return pred_array

    def restore(self):
        with open(f"saves/{self.__class__.__name__}_best_model.pt", 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, * np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias
