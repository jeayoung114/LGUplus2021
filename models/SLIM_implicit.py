"""
SLIM: Sparse Linear Methods for Top-N Recommender Systems,
Xia Ning et al.,
ICDM 2011.
"""
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import ElasticNet

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

            self.slim.fit(train_matrix, y)

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index

            nonzero_model_coef_index = self.slim.sparse_coef_.indices
            nonzero_model_coef_value = self.slim.sparse_coef_.data

            local_topK = min(len(nonzero_model_coef_value)-1, self.topk)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            for index in ranking:
                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(num_blocks, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(num_blocks, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(num_blocks, dtype=np.float32)))

                rows[numCells] = nonzero_model_coef_index[index]
                cols[numCells] = item
                values[numCells] = nonzero_model_coef_value[index]

                numCells += 1
            
            train_matrix.data[start_pos:end_pos] = current_item_data_backup
        
        self.W_sparse = sp.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), \
                                    shape=(self.num_items, self.num_items), dtype=np.float32)

        self.reconstructed = (train_matrix.tocsr() @ self.W_sparse).toarray()


    def predict(self, user_id, item_ids):
        return self.reconstructed[user_id, item_ids]
