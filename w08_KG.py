# 기본 패키지 import
from utils import load_data_kg

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


"""
dataset loading
"""
dataset = "music"
train_data, train_label, valid_data, valid_label, test_data, test_label, num_user, num_entity, num_relation, kg = load_data_kg(dataset)


"""
model 학습
"""
from models.KGCN_implicit_KG import KGCN_implicit_KG


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

start = time.time()  # 시작 시간 저장

seed_everything(seed)
kgcn = KGCN_implicit_KG(train_data=train_data, train_label=train_label, valid_data=valid_data, valid_label=valid_label, num_user=num_user, num_ent=num_entity, num_rel=num_relation, kg=kg, n_iter=1, dim=16, n_neighbor=8, \
    aggregator='sum', num_epochs=20, early_stop_trial=5, learning_rate=5e-4, reg_lambda=1e-4, batch_size=32, device=device)

kgcn.fit()
# ngcf_prec, ngcf_recall, ngcf_ndcg = eval_implicit(ngcf, train_data, test_data, top_k)
# print("NGCF: %f, %f, %f"%(ngcf_prec, ngcf_recall, ngcf_ndcg))
print("time 분 :", (time.time() - start)/60.0)
print("======================================")

