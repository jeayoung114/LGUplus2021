import numpy as np
from sklearn.utils.extmath import randomized_svd
import scipy.sparse as sp

class SVD_explicit():
    def __init__(self, train, valid, rank):
        self.train = train
        self.valid = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]
        self.rank = rank

        self.user_mean = self.train.sum(axis=1)/(self.train!=0.0).sum(axis=1).astype(float)

        for i, row in enumerate(self.train):
            self.train[i, np.where(row > 0.5)[0]] -= self.user_mean[i] 

    def fit(self):
        u, s, v = randomized_svd(self.train, n_components=self.rank, random_state=None)
        s_V = sp.diags(s) * v
        self.reconstructed = np.dot(u, s_V)

    def predict(self, user_id, item_ids):
        return self.user_mean[user_id] + self.reconstructed[user_id, item_ids]