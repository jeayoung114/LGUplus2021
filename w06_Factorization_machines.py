# 기본 패키지 import
from models.LR_implicit import LR_implicit
from models.FM_implicit import FM_implicit
from models.FFM_implicit import FFM_implicit
from models.WideAndDeep_implicit import WideAndDeep_implicit
from models.DeepFM_implicit import DeepFM_implicit
from models.xDeepFM_implicit import xDeepFM_implicit
from models.NFM_implicit import NFM_implicit
from models.DCN_implicit import DCN_implicit
from models.DCNV2_implicit import DCNV2_implicit
from utils import load_data_CTR
from utils import eval_implicit_CTR

import warnings
import random
import warnings
import torch
import numpy as np
import time

from os.path import join

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

"""
dataset loading
"""
# small, 2m, 5m
dataset = "small"
use_features = ['user_id', 'item_id', 'people', 'country', 'genre']
# use_features = ['user_id', 'item_id', 'genre', 'people']
train_arr, train_rating, valid_arr, valid_rating, test_arr, test_rating, field_dims = load_data_CTR(dataset, use_features, pos_threshold=6)
print("Train Sample: ", train_arr[0])
"""
model 학습
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

seed_everything(seed)
start = time.time()  # 시작 시간 저장
LR = LR_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=100, 
                 learning_rate=0.01, reg_lambda=0.001, batch_size=1024, early_stop_trial=20, device=device)
LR.fit()
LR_AUC, LR_logloss = eval_implicit_CTR(LR, test_arr, test_rating)
print(f"[LR]\t Test_AUC = {LR_AUC:.4f} Test_logloss = {LR_logloss:.4f}")
print("parmas :", sum(p.numel() for p in LR.parameters() if p.requires_grad))
print("time 초 :", (time.time() - start))
print("time 분 :", (time.time() - start)/60.0)
print("======================================")

seed_everything(seed)
start = time.time()  # 시작 시간 저장
FM = FM_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=100, embed_dim=20,
                 learning_rate=0.01, reg_lambda=0.001, batch_size=1024, early_stop_trial=20, device=device)
FM.fit()
FM_AUC, FM_logloss = eval_implicit_CTR(FM, test_arr, test_rating)
print(f"[FM]\t Test_AUC = {FM_AUC:.4f} Test_logloss = {FM_logloss:.4f}")
print("parmas :", sum(p.numel() for p in FM.parameters() if p.requires_grad))
print("time 초 :", (time.time() - start))
print("time 분 :", (time.time() - start)/60.0)
print("======================================")


seed_everything(seed)
start = time.time()  # 시작 시간 저장
FFM = FFM_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=100, embed_dim=20,
                   learning_rate=0.01, reg_lambda=0.001, batch_size=1024, early_stop_trial=20, device=device)
FFM.fit()
FFM_AUC, FFM_logloss = eval_implicit_CTR(FFM, test_arr, test_rating)
print(f"[FFM]\t Test_AUC = {FFM_AUC:.4f} Test_logloss = {FFM_logloss:.4f}")
print("parmas :", sum(p.numel() for p in FFM.parameters() if p.requires_grad))
print("time 초 :", (time.time() - start))
print("time 분 :", (time.time() - start)/60.0)
print("======================================")


seed_everything(seed)
start = time.time()  # 시작 시간 저장
WideAndDeep = WideAndDeep_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=100, embed_dim=20,
                                   mlp_dims=[20, 20], dropout=0.2, learning_rate=0.01, reg_lambda=0.001, batch_size=1024, early_stop_trial=20, device=device)
WideAndDeep.fit()
WideAndDeep_AUC, WideAndDeep_logloss = eval_implicit_CTR(WideAndDeep, test_arr, test_rating)
print(f"[WideAndDeep]\t Test_AUC = {WideAndDeep_AUC:.4f} Test_logloss = {WideAndDeep_logloss:.4f}")
print("parmas :", sum(p.numel() for p in WideAndDeep.parameters() if p.requires_grad))
print("time 초 :", (time.time() - start))
print("time 분 :", (time.time() - start)/60.0)
print("======================================")


seed_everything(seed)
start = time.time()  # 시작 시간 저장
DeepFM = DeepFM_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=100, embed_dim=20,
                         mlp_dims=[20, 20], dropout=0.2, learning_rate=0.01, reg_lambda=0.001, batch_size=1024, early_stop_trial=20, device=device)
DeepFM.fit()
DeepFM_AUC, DeepFM_logloss = eval_implicit_CTR(DeepFM, test_arr, test_rating)
print(f"[DeepFM]\t Test_AUC = {DeepFM_AUC:.4f} Test_logloss = {DeepFM_logloss:.4f}")
print("parmas :", sum(p.numel() for p in DeepFM.parameters() if p.requires_grad))
print("time 초 :", (time.time() - start))
print("time 분 :", (time.time() - start)/60.0)
print("======================================")


seed_everything(seed)
start = time.time()  # 시작 시간 저장
xDeepFM = xDeepFM_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=100, embed_dim=20, cross_layer_sizes=(20, 20), split_half=False,
                           mlp_dims=[20, 20], dropout=0.2, learning_rate=0.01, reg_lambda=0.001, batch_size=1024, early_stop_trial=20, device=device)
xDeepFM.fit()
xDeepFM_AUC, xDeepFM_logloss = eval_implicit_CTR(xDeepFM, test_arr, test_rating)
print(f"[xDeepFM]\t Test_AUC = {xDeepFM_AUC:.4f} Test_logloss = {xDeepFM_logloss:.4f}")
print("parmas :", sum(p.numel() for p in xDeepFM.parameters() if p.requires_grad))
print("time 초 :", (time.time() - start))
print("time 분 :", (time.time() - start)/60.0)
print("======================================")


seed_everything(seed)
start = time.time()  # 시작 시간 저장
NFM = NFM_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=100, embed_dim=20,
                   mlp_dims=[20, 20], dropout=0.2, learning_rate=0.01, reg_lambda=0.001, batch_size=1024, early_stop_trial=20, device=device)
NFM.fit()
NFM_AUC, NFM_logloss = eval_implicit_CTR(NFM, test_arr, test_rating)
print(f"[NFM]\t Test_AUC = {NFM_AUC:.4f} Test_logloss = {NFM_logloss:.4f}")
print("parmas :", sum(p.numel() for p in NFM.parameters() if p.requires_grad))
print("time 초 :", (time.time() - start))
print("time 분 :", (time.time() - start)/60.0)
print("======================================")

seed_everything(seed)
start = time.time()  # 시작 시간 저장
DCN = DCN_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=100, embed_dim=20, num_layers=3,
                   mlp_dims=[20, 20], dropout=0.2, learning_rate=0.01, reg_lambda=0.001, batch_size=1024, early_stop_trial=20, device=device)
DCN.fit()
DCN_AUC, DCN_logloss = eval_implicit_CTR(DCN, test_arr, test_rating)
print(f"[DCN]\t Test_AUC = {DCN_AUC:.4f} Test_logloss = {DCN_logloss:.4f}")
print("parmas :", sum(p.numel() for p in DCN.parameters() if p.requires_grad))
print("time 초 :", (time.time() - start))
print("time 분 :", (time.time() - start)/60.0)
print("======================================")


seed_everything(seed)
start = time.time()  # 시작 시간 저장
DCNV2 = DCNV2_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=100, embed_dim=20, num_layers=3,
                   mlp_dims=[20, 20], dropout=0.2, learning_rate=0.01, reg_lambda=0.001, batch_size=1024, early_stop_trial=20, device=device)
DCNV2.fit()
DCNV2_AUC, DCNV2_logloss = eval_implicit_CTR(DCNV2, test_arr, test_rating)
print(f"[DCNV2]\t Test_AUC = {DCNV2_AUC:.4f} Test_logloss = {DCNV2_logloss:.4f}")
print("parmas :", sum(p.numel() for p in DCNV2.parameters() if p.requires_grad))
print("time 초 :", (time.time() - start))
print("time 분 :", (time.time() - start)/60.0)
print("======================================")

print(f"[LR]\t Test_AUC = {LR_AUC:.4f} Test_logloss = {LR_logloss:.4f}")
print(f"[FM]\t Test_AUC = {FM_AUC:.4f} Test_logloss = {FM_logloss:.4f}")
print(f"[FFM]\t Test_AUC = {FFM_AUC:.4f} Test_logloss = {FFM_logloss:.4f}")
print(f"[WideAndDeep]\t Test_AUC = {WideAndDeep_AUC:.4f} Test_logloss = {WideAndDeep_logloss:.4f}")
print(f"[DeepFM]\t Test_AUC = {DeepFM_AUC:.4f} Test_logloss = {DeepFM_logloss:.4f}")
print(f"[xDeepFM]\t Test_AUC = {xDeepFM_AUC:.4f} Test_logloss = {xDeepFM_logloss:.4f}")
print(f"[NFM]\t Test_AUC = {NFM_AUC:.4f} Test_logloss = {NFM_logloss:.4f}")
print(f"[DCN]\t Test_AUC = {DCN_AUC:.4f} Test_logloss = {DCN_logloss:.4f}")
print(f"[DCNV2]\t Test_AUC = {DCNV2_AUC:.4f} Test_logloss = {DCNV2_logloss:.4f}")
