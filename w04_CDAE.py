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

import time

# colab에서 나오는 warning들을 무시합니다.
warnings.filterwarnings('ignore')

# 결과 재현을 위해 해당 코드에서 사용되는 라이브러리들의 Seed를 고정합니다.
def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    # torch.set_deterministic(True)
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
from models.DAE_implicit import DAE_implicit
from models.CDAE_implicit import CDAE_implicit

from models.MultVAE_implicit import MultVAE_implicit

"""
dataset loading
"""
dataset = "small"
train_data, valid_data, test_data, idx2title = load_data(dataset, implicit=True)
train_data = train_data+valid_data
top_k = 50

"""
model 학습
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# slim = SLIM_implicit(train=train_data, valid=valid_data, l1_reg=1e-3, l2_reg=1e-3, num_epochs=10)
# slim.fit()
# slim_prec, slim_recall, slim_ndcg = eval_implicit(slim, train_data+valid_data, test_data, top_k)
# print("SLIM: %f, %f, %f"%(slim_prec, slim_recall, slim_ndcg))

# ease = EASE_implicit(train=train_data, valid=valid_data, reg_lambda=100)
# ease.fit()
# ease_prec, ease_recall, ease_ndcg = eval_implicit(ease, train_data, test_data, top_k)
# print("EASE: %f, %f, %f"%(ease_prec, ease_recall, ease_ndcg))


# print("====================  hidden  ====================")
# cdae = CDAE_implicit(train=train_data, valid=valid_data, hidden_dim=500, dropout=0.5, num_epochs=150, learning_rate=0.01, reg_lambda=0.001, device = device, activation= 'sigmoid')
# cdae.fit()
# cdae_prec, cdae_recall, cdae_ndcg = eval_implicit(cdae, train_data, test_data, top_k)
# print("CDAE: %f, %f, %f"%(cdae_prec, cdae_recall, cdae_ndcg))

# uae = DAE_implicit(train=train_data, valid=valid_data, hidden_dim=1000, num_epochs=200, learning_rate=0.01, reg_lambda=0.001, device = device, activation= 'sigmoid')
# uae.fit()
# uae_prec, uae_recall, uae_ndcg = eval_implicit(uae, train_data, test_data, top_k)
# print("U-AE: %f, %f, %f"%(uae_prec, uae_recall, uae_ndcg))
# print("====================")


# multvae = MultVAE_implicit(train=train_data, valid=valid_data, hidden_dim=100, dropout=0.5, num_epochs=150, learning_rate=0.005, reg_lambda=0.001, device = device, activation= 'tanh')
# multvae.fit()
# multvae_prec, multvae_recall, multvae_ndcg = eval_implicit(multvae, train_data, test_data, top_k)
# print("MultVAE: %f, %f, %f"%(multvae_prec, multvae_recall, multvae_ndcg))



multvae = MultVAE_implicit(train=train_data, valid=valid_data, hidden_dim=400, dropout=0.5, num_epochs=150, learning_rate=0.001, reg_lambda=0.001, device = device)
multvae.fit()
multvae_prec, multvae_recall, multvae_ndcg = eval_implicit(multvae, train_data, test_data, top_k)
print("MultVAE: %f, %f, %f"%(multvae_prec, multvae_recall, multvae_ndcg))


# multvae = MultVAE_implicit(train=train_data, valid=valid_data, hidden_dim=100, dropout=0.5, num_epochs=150, learning_rate=0.001, reg_lambda=0.001, device = device, activation= 'tanh')
# multvae.fit()
# multvae_prec, multvae_recall, multvae_ndcg = eval_implicit(multvae, train_data, test_data, top_k)
# print("MultVAE: %f, %f, %f"%(multvae_prec, multvae_recall, multvae_ndcg))

# multvae = MultVAE_implicit(train=train_data, valid=valid_data, hidden_dim=50, dropout=0.5, num_epochs=150, learning_rate=0.001, reg_lambda=0.001, device = device, activation= 'tanh')
# multvae.fit()
# multvae_prec, multvae_recall, multvae_ndcg = eval_implicit(multvae, train_data, test_data, top_k)
# print("MultVAE: %f, %f, %f"%(multvae_prec, multvae_recall, multvae_ndcg))
