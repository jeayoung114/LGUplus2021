# 기본 패키지 import
from utils import load_data
from utils import eval_implicit
from utils import eval_explicit
import warnings
import random
from os.path import join
import warnings
import torch
import numpy as np

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
#   Implicit feedback 실습
#################################################################
# from models.SLIM_implicit import SLIM_implicit
# from models.FISM_implicit import FISMrmse_implicit
# from models.EASE_implicit import EASE_implicit

# from models.UAE_implicit import UAE_implicit
# from models.IAE_implicit import IAE_implicit
# from models.DAE_implicit import DAE_implicit
# from models.CDAE_implicit import CDAE_implicit
from models.MultVAE_implicit import MultVAE_implicit

"""
dataset loading
"""
dataset = "naver_movie_dataset_small.csv"
train_data, valid_data, test_data, idx2title = load_data(dataset, implicit=True)
train_data = train_data+valid_data
top_k = 50

"""
model 학습
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# slim = SLIM_implicit(train=train_data, valid=valid_data, l1_reg=1e-3, l2_reg=1e-3, num_epochs=10)
# slim.fit()
# slim_prec, slim_recall, slim_ndcg = eval_implicit(slim, train_data+valid_data, test_data, top_k)
# print("SLIM: %f, %f, %f"%(slim_prec, slim_recall, slim_ndcg))

# fism = FISMrmse_implicit(train=train_data, valid=valid_data, num_epochs=10)
# fism.fit()
# fism_prec, fism_recall, fism_ndcg = eval_implicit(fism, train_data, test_data, top_k)
# print("FISM: %f, %f, %f"%(fism_prec, fism_recall, fism_ndcg))

# ease = EASE_implicit(train=train_data, valid=valid_data, reg_lambda=100)
# ease.fit()
# ease_prec, ease_recall, ease_ndcg = eval_implicit(ease, train_data, test_data, top_k)
# print("EASE: %f, %f, %f"%(ease_prec, ease_recall, ease_ndcg))

# uae = UAE_implicit(train=train_data, valid=valid_data, hidden_dim=10, num_epochs=10, learning_rate=0.1, reg_lambda=0.001, device = device, activation= 'tanh')
# uae.fit()
# uae_prec, uae_recall, uae_ndcg = eval_implicit(uae, train_data, test_data, top_k)
# print("U-AE: %f, %f, %f"%(uae_prec, uae_recall, uae_ndcg))

# iae = IAE_implicit(train=train_data, valid=valid_data, hidden_dim=10, num_epochs=10, learning_rate=0.1, reg_lambda=0.001, device = device, activation= 'tanh')
# iae.fit()
# iae_prec, iae_recall, iae_ndcg = eval_implicit(iae, train_data, test_data, top_k)
# print("I-AE: %f, %f, %f"%(iae_prec, iae_recall, iae_ndcg))

# dae = DAE_implicit(train=train_data, valid=valid_data, hidden_dim=10, dropout=0.1, num_epochs=10, learning_rate=0.1, reg_lambda=0.001, device = device, activation= 'tanh')
# dae.fit()
# dae_prec, dae_recall, dae_ndcg = eval_implicit(dae, train_data, test_data, top_k)
# print("DAE: %f, %f, %f"%(dae_prec, dae_recall, dae_ndcg))

# cdae = CDAE_implicit(train=train_data, valid=valid_data, hidden_dim=10, dropout=0.1, num_epochs=10, learning_rate=0.1, reg_lambda=0.001, device = device, activation= 'tanh')
# cdae.fit()
# cdae_prec, cdae_recall, cdae_ndcg = eval_implicit(cdae, train_data, test_data, top_k)
# print("CDAE: %f, %f, %f"%(cdae_prec, cdae_recall, cdae_ndcg))

multvae = MultVAE_implicit(train=train_data, valid=valid_data, hidden_dim=100, dropout=0.1, num_epochs=100, learning_rate=0.005, reg_lambda=0.001, device = device, activation= 'tanh')
multvae.fit()
multvae_prec, multvae_recall, multvae_ndcg = eval_implicit(multvae, train_data, test_data, top_k)
print("MultVAE: %f, %f, %f"%(multvae_prec, multvae_recall, multvae_ndcg))


# print("Precision, Recall, NDCG@%s 결과"%(top_k))