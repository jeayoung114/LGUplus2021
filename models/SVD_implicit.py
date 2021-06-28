import numpy as np
from sklearn.utils.extmath import randomized_svd
import scipy.sparse as sp

class SVD_implicit():
    def __init__(self, train, valid, rank):
        self.train = train
        self.valid = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]
        self.rank = rank

    def fit(self):
        u, s, v = randomized_svd(self.train, n_components=self.rank, random_state=None)
        s_V = sp.diags(s) * v
        self.reconstructed = np.dot(u, s_V)

    def predict(self, user_id, item_ids):
        return self.reconstructed[user_id, item_ids]
        