"""
SLIM: Sparse Linear Methods for Top-N Recommender Systems,
Xia Ning et al.,
ICDM 2011.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import ElasticNet
from tqdm import tqdm
from multiprocessing import Pool

class SLIM_implicit():
    def __init__(self, train, valid, l1_reg=1e-3, l2_reg=1e-3, num_epochs=10, topk=100):
        self.train = train
        self.valid = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]
        self.num_epochs = num_epochs
        self.topk = topk

        alpha = l1_reg + l2_reg
        l1_ratio = l1_reg / alpha
        self.slim = ElasticNet(alpha=alpha,
                                l1_ratio=l1_ratio,
                                positive=True, # weight matrix has non-negative values
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=self.num_epochs,
                                tol=1e-3)


    def fit(self):
        train_matrix = sp.csc_matrix(self.train)
        num_blocks = 10000000
        numCells = 0

        rows = np.zeros(num_blocks, dtype=np.int32)
        cols = np.zeros(num_blocks, dtype=np.int32)
        values = np.zeros(num_blocks, dtype=np.float32)

        tqdm_iterator = tqdm(range(self.num_items), desc='# items covered', total=self.num_items)
        for item in tqdm_iterator:
            # j-th column of R (ground-truth)
            y = train_matrix[:, item].toarray()

            # set the j-th column of R to zero
            start_pos = train_matrix.indptr[item]
            end_pos = train_matrix.indptr[item + 1]

            current_item_data_backup = train_matrix.data[start_pos: end_pos].copy()
            train_matrix.data[start_pos: end_pos] = 0.0

            # train SLIM
            self.slim.fit(train_matrix, y)

            nonzero_model_coef_index = self.slim.sparse_coef_.indices
            nonzero_model_coef_value = self.slim.sparse_coef_.data

            for row_index, value in zip(nonzero_model_coef_index, nonzero_model_coef_value):
                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(num_blocks, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(num_blocks, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(num_blocks, dtype=np.float32)))

                rows[numCells] = row_index
                cols[numCells] = item
                values[numCells] = value

                numCells += 1
            
            train_matrix.data[start_pos:end_pos] = current_item_data_backup
        
        # make sparse W matrix
        self.W_sparse = sp.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), \
                                    shape=(self.num_items, self.num_items), dtype=np.float32)

        self.reconstructed = (train_matrix.tocsr() @ self.W_sparse).toarray()


    def predict(self, user_id, item_ids):
        return self.reconstructed[user_id, item_ids]
