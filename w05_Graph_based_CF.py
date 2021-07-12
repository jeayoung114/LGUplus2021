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
from models.NGCF_implicit import NGCF_implicit
from models.LightGCN_implicit import LightGCN_implicit

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
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# start = time.time()  # 시작 시간 저장
# ngcf = NGCF_implicit(train_data, valid_data, learning_rate=0.01, regs=0.001, batch_size=1024, num_epochs=100, emb_size=400, layers_size=3, node_dropout=0.5, mess_dropout=0.5, device=device)
# ngcf.fit()
# ngcf_prec, ngcf_recall, ngcf_ndcg = eval_implicit(ngcf, train_data, test_data, top_k)
# print("NGCF: %f, %f, %f"%(ngcf_prec, ngcf_recall, ngcf_ndcg))
# print("time 분 :", (time.time() - start)/60.0)
# print("======================================")


start = time.time()  # 시작 시간 저장
ngcf = LightGCN_implicit(train_data, valid_data, learning_rate=0.001, regs=0.0001, batch_size=20480, num_epochs=100, emb_size=200, num_layers=2, node_dropout=0.5, device=device)
ngcf.fit()
ngcf_prec, ngcf_recall, ngcf_ndcg = eval_implicit(ngcf, train_data, test_data, top_k)
print("NGCF: %f, %f, %f"%(ngcf_prec, ngcf_recall, ngcf_ndcg))
print("time 분 :", (time.time() - start)/60.0)
print("======================================")

