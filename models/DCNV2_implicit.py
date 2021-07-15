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

# https://github.com/rixwew/pytorch-fm/blob/master/torchfm/model/dcn.py을 참고했습니다.
class DCNV2_implicit(torch.nn.Module):
    def __init__(self, train_data, train_label, valid_data, valid_label, field_dims, embed_dim, num_layers, mlp_dims, dropout,
                 num_epochs, early_stop_trial, learning_rate, reg_lambda, batch_size, device):
        super().__init__()

        self.train_data = train_data
        self.train_label = train_label
        self.valid_data = valid_data
        self.valid_label = valid_label
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.num_layers = num_layers
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
        self.embedding = FeaturesEmbedding(self.field_dims, self.embed_dim)
        self.embed_output_dim = len(self.field_dims) * self.embed_dim
        self.cn = CrossNetworkMMoE(self.embed_output_dim, self.num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, self.dropout, output_layer=False)
        self.linear = torch.nn.Linear(self.mlp_dims[-1] + self.embed_output_dim, 1)

        # 최적화 방법 설정
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.reg_lambda)

        # 모델을 device로 보냄
        self.to(self.device)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        x_l1 = self.cn(embed_x)
        h_l2 = self.mlp(embed_x)
        x_stack = torch.cat([x_l1, h_l2], dim=1)
        p = self.linear(x_stack)
        output = torch.sigmoid(p.squeeze(1))
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

class CrossNetworkMMoE(nn.Module):
    def __init__(self, in_features, low_rank=32, num_experts=4, layer_num=2, device='cpu'):
        super(CrossNetworkMMoE, self).__init__()
        self.layer_num = layer_num
        self.num_experts = num_experts

        # U: (in_features, low_rank)
        self.U_list = nn.Parameter(torch.Tensor(self.layer_num, num_experts, in_features, low_rank))
        # V: (in_features, low_rank)
        self.V_list = nn.Parameter(torch.Tensor(self.layer_num, num_experts, in_features, low_rank))
        # C: (low_rank, low_rank)
        self.C_list = nn.Parameter(torch.Tensor(self.layer_num, num_experts, low_rank, low_rank))
        self.gating = nn.ModuleList([nn.Linear(in_features, 1, bias=False) for i in range(self.num_experts)])

        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        init_para_list = [self.U_list, self.V_list, self.C_list]
        for para in init_para_list:
            for i in range(self.layer_num):
                nn.init.xavier_normal_(para[i])

        for i in range(len(self.bias)):
            nn.init.zeros_(self.bias[i])

        self.to(device)

    def forward(self, x):
        x_0 = x.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_of_experts.append(self.gating[expert_id](x_l.squeeze(2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = torch.matmul(self.V_list[i][expert_id].t(), x_l)  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = torch.tanh(v_x)
                v_x = torch.matmul(self.C_list[i][expert_id], v_x)
                v_x = torch.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = torch.matmul(self.U_list[i][expert_id], v_x)  # (bs, in_features, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(dot_.squeeze(2))

            # (3) mixture of low-rank experts
            output_of_experts = torch.stack(output_of_experts, 2)  # (bs, in_features, num_experts)
            gating_score_of_experts = torch.stack(gating_score_of_experts, 1)  # (bs, num_experts, 1)
            moe_out = torch.matmul(output_of_experts, gating_score_of_experts.softmax(1))
            x_l = moe_out + x_l  # (bs, in_features, 1)

        x_l = x_l.squeeze()  # (bs, in_features)
        return x_l



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