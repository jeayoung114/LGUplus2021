# 기본 패키지 import
import numpy as np
from utils import load_data
from utils import eval_implicit
from utils import eval_explicit
import warnings
import random
from os.path import join
import warnings

import torch
import numpy as np
import random

# colab에서 나오는 warning들을 무시합니다.
warnings.filterwarnings('ignore')

# 결과 재현을 위해 해당 코드에서 사용되는 라이브러리들의 Seed를 고정합니다.
def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.set_deterministic(True)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

seed = 1
seed_everything(seed)

#################################################################
#   Explicit feedback 실습
#################################################################
from models.UserKNN_explicit import UserKNN_explicit
from models.ItemKNN_explicit import ItemKNN_explicit
from models.SVD_explicit import SVD_explicit
from models.MF_explicit import MF_explicit

"""
dataset loading
"""
dataset = "naver_movie_dataset_small.csv"
train_data, valid_data, test_data, idx2title = load_data(dataset, implicit=False)

"""
model 학습
"""
userknn = UserKNN_explicit(train=np.copy(train_data), valid=valid_data, top_k=10)
itemknn = ItemKNN_explicit(train=np.copy(train_data), valid=valid_data, top_k=10)
svd = SVD_explicit(train=np.copy(train_data), valid=valid_data, rank=30)
mf = MF_explicit(train=np.copy(train_data), valid=valid_data, n_features=20, learning_rate=1e-2, num_epochs = 100)

userknn.fit()
itemknn.fit()
svd.fit()
mf.fit()

"""
model 평가
"""
userknn_rmse = eval_explicit(userknn, train_data+valid_data, test_data)
itemknn_rmse = eval_explicit(itemknn, train_data+valid_data, test_data)
svd_rmse = eval_explicit(svd, train_data+valid_data, test_data)
mf_rmse = eval_explicit(mf, train_data+valid_data, test_data)

print("RMSE 결과")
print("UserKNN: %f"%(userknn_rmse))
print("ItemKNN: %f"%(itemknn_rmse))
print("SVD: %f"%(svd_rmse))
print("MF: %f"%(mf_rmse))


#################################################################
#   Implicit feedback 실습
#################################################################
from models.UserKNN_implicit import UserKNN_implicit
from models.ItemKNN_implicit import ItemKNN_implicit
from models.SVD_implicit import SVD_implicit
from models.WMF_GD_implicit import WMF_GD_implicit
from models.WMF_ALS_implicit import WMF_ALS_implicit
from models.LogisticMF_implicit import LogisticMF_implicit

"""
dataset loading
"""
dataset = "naver_movie_dataset_small.csv"
train_data, valid_data, test_data, idx2title = load_data(dataset, implicit=True)

"""
model 학습
"""
userknn = UserKNN_implicit(train=train_data, valid=valid_data, top_k=10, sim = 'cos', inverse_freq=False)
itemknn = ItemKNN_implicit(train=train_data, valid=valid_data, top_k=10, sim = 'cos', inverse_freq=False)
svd = SVD_implicit(train_data, valid_data, rank=30)
wmf_gd = WMF_GD_implicit(train=train_data, valid=valid_data, n_features=20, learning_rate=1e-2, num_epochs = 100)
wmf_als = WMF_ALS_implicit(train=train_data, valid=valid_data, n_features=20, num_epochs = 10)
logistic_mf = LogisticMF_implicit(train=train_data, valid=valid_data, alpha=2, n_features=20, learning_rate=1e-2, num_epochs = 100)

userknn.fit()
itemknn.fit()
svd.fit()
wmf_gd.fit()
wmf_als.fit()
logistic_mf.fit()

"""
model 평가
"""
top_k = 50
userknn_prec, userknn_recall, userknn_ndcg = eval_implicit(userknn, train_data+valid_data, test_data, top_k)
itemknn_prec, itemknn_recall, itemknn_ndcg = eval_implicit(itemknn, train_data+valid_data, test_data, top_k)
svd_prec, svd_recall, svd_ndcg = eval_implicit(svd, train_data+valid_data, test_data, top_k)
wmf_gd_prec, wmf_gd_recall, wmf_gd_ndcg = eval_implicit(wmf_gd, train_data+valid_data, test_data, top_k)
wmf_als_prec, wmf_als_recall, wmf_als_ndcg = eval_implicit(wmf_als, train_data+valid_data, test_data, top_k)
logistic_mf_prec, logistic_mf_recall, logistic_mf_ndcg = eval_implicit(logistic_mf, train_data+valid_data, test_data, top_k)

print("Precision, Recall, NDCG@%s 결과"%(top_k))
print("UserKNN: %f, %f, %f"%(userknn_prec, userknn_recall, userknn_ndcg))
print("ItemKNN: %f, %f, %f"%(itemknn_prec, itemknn_recall, itemknn_ndcg))
print("SVD: %f, %f, %f"%(svd_prec, svd_recall, svd_ndcg))
print("WMF_GD: %f, %f, %f"%(wmf_gd_prec, wmf_gd_recall, wmf_gd_ndcg))
print("WMG_ALS: %f, %f, %f"%(wmf_als_prec, wmf_als_recall, wmf_als_ndcg))
print("LogisticMF: %f, %f, %f"%(logistic_mf_prec, logistic_mf_recall, logistic_mf_ndcg))
