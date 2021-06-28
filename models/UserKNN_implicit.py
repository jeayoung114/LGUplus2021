import os
import math
from time import time
from numpy import dot
from numpy.linalg import norm
# from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import jaccard_score
from numpy import inf


class UserKNN_implicit():
    def __init__(self, train, valid, top_k, sim = 'cos', inverse_freq=False):
        self.train = train
        self.valid = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]
        self.top_k = top_k

        self.sim = sim

        if inverse_freq:
            self.inverse_item_freq = np.sum(self.train, axis=0)
            self.inverse_item_freq = 1/np.log(self.inverse_item_freq+1)
            self.inverse_item_freq[self.inverse_item_freq == inf] = 0
        else:
            self.inverse_item_freq = np.ones(self.num_items)

        self.inverse_item_freq_train = self.train * self.inverse_item_freq[None, :]


    def fit(self):
        # 사용자-사용자 유사도 저장 행렬 만듦
        user_user_sim_matrix = np.zeros((self.num_users, self.num_users))
        if self.sim == 'cos':
            for user_i in range(0, self.num_users):
                for user_j in range(user_i+1, self.num_users):
                    a = self.train[user_i]
                    b = self.train[user_j]

                    invers_freq_a = self.inverse_item_freq_train[user_i]
                    invers_freq_b = self.inverse_item_freq_train[user_j]

                    # 코사인 유사도 구하기
                    user_user_sim_matrix[user_i, user_j] = np.dot(invers_freq_a, invers_freq_b)/(np.linalg.norm(a)*np.linalg.norm(b))

        elif self.sim == 'jaccard':
            for user_i in range(0, self.num_users):
                for user_j in range(user_i+1, self.num_users):
                    a = self.train[user_i]
                    b = self.train[user_j]

                    # jaccard 유사도 구하기
                    user_user_sim_matrix[user_i, user_j] = jaccard_score(a, b)
        else:
            print("Similarity 선택해야 합니다!")

        self.user_user_sim_matrix = (user_user_sim_matrix + user_user_sim_matrix.T)


    def predict(self, user_id, item_ids):
        predicted_values = []
        for one_missing_item in item_ids:
            
            # item i를 시청한 사용자들
            rated_users = np.where(self.train[:, one_missing_item] > 0.5)[0]
            unsorted_sim = self.user_user_sim_matrix[user_id, rated_users]

            # 유사도 정렬
            sorted_sim = np.sort(unsorted_sim)
            sorted_sim = sorted_sim[::-1]

            if(self.top_k > len(sorted_sim)):
                top_k = len(sorted_sim)
            else:
                top_k = self.top_k 

            # Top K 이웃의 유사도 가져오기
            sorted_sim = sorted_sim[0:top_k]

            # 예측 값 구하기
            if(top_k == 0):
                predicted_values.append(0.0)
            else:
                predicted_values.append(np.sum(sorted_sim) / top_k)

        return predicted_values
