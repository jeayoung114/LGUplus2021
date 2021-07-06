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

        alpha = l1_reg + l2_reg # l1, l2 규제 계수의 합
        l1_ratio = l1_reg / alpha # alpha를 이용해 l1 비율 계산
        self.slim = ElasticNet(alpha=alpha,
                                l1_ratio=l1_ratio,
                                positive=True, # 가중치 행렬이 양수만 갖음
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
            # 특정 영화의 암시적 피드백 (ground-truth)
            y = train_matrix[:, item].toarray()
            # 사용자-항목 행렬에서 특정 영화의 시작, 끝 포인터
            start_pos = train_matrix.indptr[item]
            end_pos = train_matrix.indptr[item + 1]
            # 특정 영화의 피드백 값을 포인터로 인덱싱하여 따로 백업
            current_item_data_backup = train_matrix.data[start_pos: end_pos].copy()
            # 사용자-항목 행렬에서 특정 영화에 대해 0으로 채워주면 가중치 행렬에 0이 들어간 효과와 같음
            train_matrix.data[start_pos: end_pos] = 0.0
            # .fit 함수를 이용해 SLIM 학습
            self.slim.fit(train_matrix, y)

            nonzero_model_coef_index = self.slim.sparse_coef_.indices # 가중치 행렬의 인덱스
            nonzero_model_coef_value = self.slim.sparse_coef_.data # 가중치 행렬의 값

            for row_index, value in zip(nonzero_model_coef_index, nonzero_model_coef_value):
                # rows array에 평점이 가득 차면, 빈 행렬을 새로 추가 (기본값: 1000만)
                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(num_blocks, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(num_blocks, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(num_blocks, dtype=np.float32)))
                # 해당하는 행, 열 인덱스와 값을 각 array에 저장
                rows[numCells] = row_index
                cols[numCells] = item
                values[numCells] = value
                # 평점이 추가될 때마다 numCells +1
                numCells += 1
            # 따로 저장했던 특정 영화의 피드백 값을 다시 복원
            train_matrix.data[start_pos:end_pos] = current_item_data_backup
        
        # 희소 W 행렬 생성
        self.W_sparse = sp.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), \
                                    shape=(self.num_items, self.num_items), dtype=np.float32)
        # 사용자-항목 행렬과 희소 W 행렬의 행렬 곱을 통해 예측 값 행렬 생성
        self.reconstructed = (train_matrix.tocsr() @ self.W_sparse).toarray()


    def predict(self, user_id, item_ids):
        return self.reconstructed[user_id, item_ids]
