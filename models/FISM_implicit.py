"""
FISM: Factored Item Similarity Models for Top-N Recommender Systems,
Kabbur et al.,
KDD 2013.
"""
import math
from time import time

import numpy as np
import scipy.sparse as sp


class FISMrmse_implicit():
    def __init__(self, train, valid, lr=0.001, num_epochs=10,
                 rho=3, # negative 샘플링 비율
                 alpha=0.5, # 사용자 초모수
                 beta=1e-4, # 가중치 l2 정규화 
                 item_bias_reg=1e-2, # 편향 l2 정규화 
                 user_bias_reg=0.1, # 편향 l2 정규화 
                 num_factors=16): # P, Q 행렬 은닉층 차원
        self.train = train
        self.valid = valid
        self.n_users = train.shape[0]
        self.num_items = train.shape[1]

        self.lr = lr
        self.num_epochs = num_epochs
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        self.P = self.normal(size=(self.num_items, num_factors))
        self.Q = self.normal(size=(self.num_items, num_factors))
        self.user_biases = self.normal(size=self.n_users)
        self.item_biases = self.normal(size=self.num_items)
        self.num_factors = num_factors

    def normal(self, size=None):
        return np.random.normal(loc=0.0, scale=0.01, size=size)

    def fit(self):
        self.train_matrix = sp.dok_matrix(self.train)
        losses = []
        for epoch in range(self.num_epochs):
            start = time()

            # 샘플링
            R = sp.dok_matrix((self.n_users, self.num_items), dtype=np.float32)
            for (u, i) in self.train_matrix.keys():
                R[u, i] = 1
                # negative 샘플링
                for t in range(self.rho):
                    j = np.random.randint(self.num_items)
                    while (u, j) in self.train_matrix.keys() or (u, j) in R.keys():
                        j = np.random.randint(self.num_items)
                    R[u, j] = 0
            print('Sample finished.')

            loss = 0
            for (u, i) in R.keys():
                # 사용자 u가 평가한 항목 수
                n_u = len(self.train_matrix[u])
                # P_j의 합 계산
                x = np.zeros(shape=(self.num_factors,))
                for j in self.train_matrix[u].keys():
                    j = j[1]
                    if j == i: continue
                    x += self.P[j]
                x = x / math.pow(n_u - 1, self.alpha)

                # 사용자, 항목 편향 
                b_u = self.user_biases[u]
                b_i = self.item_biases[i]
                # 예측 값 계산
                predict_r_ui = b_u + b_i + np.dot(self.Q[i], x)
                # 실제 값과 예측 값의 차이
                error_ui = R[u, i] - predict_r_ui

                # 사용자, 항목 편향 업데이트
                self.user_biases[u] = b_u + self.lr * (error_ui - self.user_bias_reg * b_u)
                self.item_biases[i] = b_i + self.lr * (error_ui - self.item_bias_reg * b_i)

                loss += error_ui * error_ui + self.user_bias_reg * b_u * b_u + self.item_bias_reg * b_i * b_i

                # Q 행렬 업데이트
                self.Q[i] = self.Q[i] + self.lr * (error_ui * x - self.beta * self.Q[i])

                loss += self.beta * np.dot(self.Q[i], self.Q[i])

                for j in self.train_matrix[u].keys():
                    j = j[1]
                    if j == i: continue
                    # P 행렬 업데이트
                    self.P[j] = self.P[j] + self.lr * \
                                (error_ui / math.pow(n_u - 1, self.alpha) * self.Q[i] - self.beta * self.P[j])
                    loss += self.beta * np.dot(self.P[j], self.P[j])

            loss /= 2
            losses.append(loss)
            print('Epoch %d: loss = %.4f [%.1fs]' % (epoch, loss, time() - start))


    def predict(self, user_id, item_ids):
        predictions = []
        u = user_id
        for i in item_ids:
            bias = self.user_biases[u] + self.item_biases[i]

            dot_sum = 0
            for j in self.train_matrix[u].keys():
                j = j[1]
                dot_sum += np.dot(self.P[i], self.Q[j])

            n_u = len(self.train_matrix[u])
            if n_u <= 0:
                w_u = 0
            else:
                w_u = math.pow(n_u, -self.alpha)
            predictions.append(bias + w_u * dot_sum)

        return np.array(predictions)
