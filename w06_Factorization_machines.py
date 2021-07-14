# 기본 패키지 import
from models.FM_implicit import FM_implicit
from models.FFM_implicit import FFM_implicit
from models.WideAndDeep_implicit import WideAndDeep_implicit
from models.DeepFM_implicit import DeepFM_implicit
from models.xDeepFM_implicit import xDeepFM_implicit
from models.NFM_implicit import NFM_implicit
from models.DCN_implicit import DCN_implicit
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
dataset = "small"
train_arr, train_rating, valid_arr, valid_rating, test_arr, test_rating, field_dims = load_data_CTR(dataset)
print("Train Sample: ", train_arr[0])
"""
model 학습
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

start = time.time()  # 시작 시간 저장
FM = FM_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=10, embed_dim=20,
                 learning_rate=0.01, reg_lambda=0.001, batch_size=256,
                 device=device)
FM.fit()
FM_AUC, FM_logloss = eval_implicit_CTR(FM, test_arr, test_rating)
print(f"[FM]\t Test_AUC = {FM_AUC} Test_logloss = {FM_logloss}")
print("time 분 :", (time.time() - start)/60.0)
print("======================================")


start = time.time()  # 시작 시간 저장
FFM = FFM_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=10, embed_dim=20,
                   learning_rate=0.01, reg_lambda=0.001, batch_size=256,
                   device=device)
FFM.fit()
FFM_AUC, FFM_logloss = eval_implicit_CTR(FFM, test_arr, test_rating)
print(f"[FFM]\t Test_AUC = {FFM_AUC} Test_logloss = {FFM_logloss}")
print("time 분 :", (time.time() - start)/60.0)
print("======================================")


start = time.time()  # 시작 시간 저장
WideAndDeep = WideAndDeep_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=10, embed_dim=20,
                                   mlp_dims=[20, 20], dropout=0.2, learning_rate=0.01, reg_lambda=0.001, batch_size=256, device=device)
WideAndDeep.fit()
WideAndDeep_AUC, WideAndDeep_logloss = eval_implicit_CTR(WideAndDeep, test_arr, test_rating)
print(f"[WideAndDeep]\t Test_AUC = {WideAndDeep_AUC} Test_logloss = {WideAndDeep_logloss}")
print("time 분 :", (time.time() - start)/60.0)
print("======================================")


start = time.time()  # 시작 시간 저장
DeepFM = DeepFM_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=10, embed_dim=20,
                                   mlp_dims=[20, 20], dropout=0.2, learning_rate=0.01, reg_lambda=0.001, batch_size=256, device=device)
DeepFM.fit()
DeepFM_AUC, DeepFM_logloss = eval_implicit_CTR(DeepFM, test_arr, test_rating)
print(f"[DeepFM]\t Test_AUC = {DeepFM_AUC} Test_logloss = {DeepFM_logloss}")
print("time 분 :", (time.time() - start)/60.0)
print("======================================")


start = time.time()  # 시작 시간 저장
xDeepFM = xDeepFM_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=10, embed_dim=20, cross_layer_sizes=(20, 20), split_half=False,
                           mlp_dims=[20, 20], dropout=0.2, learning_rate=0.01, reg_lambda=0.001, batch_size=256, device=device)
xDeepFM.fit()
xDeepFM_AUC, xDeepFM_logloss = eval_implicit_CTR(xDeepFM, test_arr, test_rating)
print(f"[xDeepFM]\t Test_AUC = {xDeepFM_AUC} Test_logloss = {xDeepFM_logloss}")
print("time 분 :", (time.time() - start)/60.0)
print("======================================")


start = time.time()  # 시작 시간 저장
NFM = NFM_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=10, embed_dim=20,
                   mlp_dims=[20, 20], dropout=0.2, learning_rate=0.01, reg_lambda=0.001, batch_size=256, device=device)
NFM.fit()
NFM_AUC, NFM_logloss = eval_implicit_CTR(NFM, test_arr, test_rating)
print(f"[NFM]\t Test_AUC = {NFM_AUC} Test_logloss = {NFM_logloss}")
print("time 분 :", (time.time() - start)/60.0)
print("======================================")

start = time.time()  # 시작 시간 저장
DCN = DCN_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=10, embed_dim=20, num_layers=3,
                   mlp_dims=[20, 20], dropout=0.2, learning_rate=0.01, reg_lambda=0.001, batch_size=256, device=device)
DCN.fit()
DCN_AUC, DCN_logloss = eval_implicit_CTR(DCN, test_arr, test_rating)
print(f"[DCN]\t Test_AUC = {DCN_AUC} Test_logloss = {DCN_logloss}")
print("time 분 :", (time.time() - start)/60.0)
print("======================================")
