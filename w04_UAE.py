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
from models.IAE_implicit import IAE_implicit
from models.UAE_implicit import UAE_implicit
from models.MultVAE_implicit import MultVAE_implicit

"""
dataset loading
"""
dataset = "2m"
train_data, valid_data, test_data, idx2title = load_data(dataset, implicit=True)
train_data = train_data+valid_data
top_k = 50

"""
model 학습
"""
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

start = time.time()  # 시작 시간 저장
multvae = MultVAE_implicit(train=train_data, valid=valid_data, hidden_dim=400, dropout=0.5, num_epochs=150, learning_rate=0.001, reg_lambda=0.0001, device = device)
multvae.fit()
multvae_prec, multvae_recall, multvae_ndcg = eval_implicit(multvae, train_data, test_data, top_k)
print("MultVAE 400: %f, %f, %f"%(multvae_prec, multvae_recall, multvae_ndcg))
print("time 분 :", (time.time() - start)/60.0)
print("======================================")


start = time.time()  # 시작 시간 저장
multvae = MultVAE_implicit(train=train_data, valid=valid_data, hidden_dim=500, dropout=0.5, num_epochs=150, learning_rate=0.001, reg_lambda=0.0001, device = device)
multvae.fit()
multvae_prec, multvae_recall, multvae_ndcg = eval_implicit(multvae, train_data, test_data, top_k)
print("MultVAE 500: %f, %f, %f"%(multvae_prec, multvae_recall, multvae_ndcg))
print("time 분 :", (time.time() - start)/60.0)
print("======================================")



start = time.time()  # 시작 시간 저장
multvae = MultVAE_implicit(train=train_data, valid=valid_data, hidden_dim=1000, dropout=0.5, num_epochs=150, learning_rate=0.001, reg_lambda=0.0001, device = device)
multvae.fit()
multvae_prec, multvae_recall, multvae_ndcg = eval_implicit(multvae, train_data, test_data, top_k)
print("MultVAE 1000: %f, %f, %f"%(multvae_prec, multvae_recall, multvae_ndcg))
print("time 분 :", (time.time() - start)/60.0)
print("======================================")


start = time.time()  # 시작 시간 저장
multvae = MultVAE_implicit(train=train_data, valid=valid_data, hidden_dim=1500, dropout=0.5, num_epochs=150, learning_rate=0.001, reg_lambda=0.0001, device = device)
multvae.fit()
multvae_prec, multvae_recall, multvae_ndcg = eval_implicit(multvae, train_data, test_data, top_k)
print("MultVAE 1500: %f, %f, %f"%(multvae_prec, multvae_recall, multvae_ndcg))
print("time 분 :", (time.time() - start)/60.0)
print("======================================")



start = time.time()  # 시작 시간 저장
multvae = MultVAE_implicit(train=train_data, valid=valid_data, hidden_dim=2000, dropout=0.5, num_epochs=150, learning_rate=0.001, reg_lambda=0.0001, device = device)
multvae.fit()
multvae_prec, multvae_recall, multvae_ndcg = eval_implicit(multvae, train_data, test_data, top_k)
print("MultVAE 2000: %f, %f, %f"%(multvae_prec, multvae_recall, multvae_ndcg))
print("time 분 :", (time.time() - start)/60.0)
print("======================================")




# start = time.time()  # 시작 시간 저장
# uae = UAE_implicit(train=train_data, valid=valid_data, hidden_dim=500, num_epochs=50, learning_rate=0.01, reg_lambda=0.0001, device = device, activation= 'sigmoid')
# uae.fit()
# uae_prec, uae_recall, uae_ndcg = eval_implicit(uae, train_data, test_data, top_k)
# print("U-AE 500: %f, %f, %f"%(uae_prec, uae_recall, uae_ndcg))
# print("time 분 :", (time.time() - start)/60.0)
# print("======================================")


# start = time.time()  # 시작 시간 저장
# uae = UAE_implicit(train=train_data, valid=valid_data, hidden_dim=500, num_epochs=100, learning_rate=0.01, reg_lambda=0.0001, device = device, activation= 'sigmoid')
# uae.fit()
# uae_prec, uae_recall, uae_ndcg = eval_implicit(uae, train_data, test_data, top_k)
# print("U-AE 500: %f, %f, %f"%(uae_prec, uae_recall, uae_ndcg))
# print("time 분 :", (time.time() - start)/60.0)
# print("======================================")


# start = time.time()  # 시작 시간 저장
# uae = UAE_implicit(train=train_data, valid=valid_data, hidden_dim=1000, num_epochs=100, learning_rate=0.01, reg_lambda=0.0001, device = device, activation= 'sigmoid')
# uae.fit()
# uae_prec, uae_recall, uae_ndcg = eval_implicit(uae, train_data, test_data, top_k)
# print("U-AE 1000: %f, %f, %f"%(uae_prec, uae_recall, uae_ndcg))
# print("time 분 :", (time.time() - start)/60.0)
# print("======================================")


# start = time.time()  # 시작 시간 저장
# uae = UAE_implicit(train=train_data, valid=valid_data, hidden_dim=1500, num_epochs=100, learning_rate=0.01, reg_lambda=0.0001, device = device, activation= 'sigmoid')
# uae.fit()
# uae_prec, uae_recall, uae_ndcg = eval_implicit(uae, train_data, test_data, top_k)
# print("U-AE 1500: %f, %f, %f"%(uae_prec, uae_recall, uae_ndcg))
# print("time 분 :", (time.time() - start)/60.0)
# print("======================================")


# start = time.time()  # 시작 시간 저장
# uae = UAE_implicit(train=train_data, valid=valid_data, hidden_dim=2000, num_epochs=100, learning_rate=0.01, reg_lambda=0.0001, device = device, activation= 'sigmoid')
# uae.fit()
# uae_prec, uae_recall, uae_ndcg = eval_implicit(uae, train_data, test_data, top_k)
# print("U-AE 2000: %f, %f, %f"%(uae_prec, uae_recall, uae_ndcg))
# print("time 분 :", (time.time() - start)/60.0)
# print("======================================")





# ######################################################################################################################################################
# ######################################################################################################################################################
# start = time.time()  # 시작 시간 저장
# uae = IAE_implicit(train=train_data, valid=valid_data, hidden_dim=1000, num_epochs=50, learning_rate=0.01, reg_lambda=0.0001, device = device, activation= 'sigmoid')
# uae.fit()
# uae_prec, uae_recall, uae_ndcg = eval_implicit(uae, train_data, test_data, top_k)
# print("I-AE 1000: %f, %f, %f"%(uae_prec, uae_recall, uae_ndcg))
# print("time 분 :", (time.time() - start)/60.0)
# print("======================================")


# start = time.time()  # 시작 시간 저장
# uae = IAE_implicit(train=train_data, valid=valid_data, hidden_dim=1000, num_epochs=100, learning_rate=0.01, reg_lambda=0.0001, device = device, activation= 'sigmoid')
# uae.fit()
# uae_prec, uae_recall, uae_ndcg = eval_implicit(uae, train_data, test_data, top_k)
# print("I-AE 1000: %f, %f, %f"%(uae_prec, uae_recall, uae_ndcg))
# print("time 분 :", (time.time() - start)/60.0)
# print("======================================")


# start = time.time()  # 시작 시간 저장
# uae = IAE_implicit(train=train_data, valid=valid_data, hidden_dim=2000, num_epochs=100, learning_rate=0.01, reg_lambda=0.0001, device = device, activation= 'sigmoid')
# uae.fit()
# uae_prec, uae_recall, uae_ndcg = eval_implicit(uae, train_data, test_data, top_k)
# print("I-AE 2000: %f, %f, %f"%(uae_prec, uae_recall, uae_ndcg))
# print("time 분 :", (time.time() - start)/60.0)
# print("======================================")


# start = time.time()  # 시작 시간 저장
# uae = IAE_implicit(train=train_data, valid=valid_data, hidden_dim=3000, num_epochs=100, learning_rate=0.01, reg_lambda=0.0001, device = device, activation= 'sigmoid')
# uae.fit()
# uae_prec, uae_recall, uae_ndcg = eval_implicit(uae, train_data, test_data, top_k)
# print("I-AE 3000: %f, %f, %f"%(uae_prec, uae_recall, uae_ndcg))
# print("time 분 :", (time.time() - start)/60.0)
# print("======================================")

# ######################################################################################################################################################
# ######################################################################################################################################################

# start = time.time()  # 시작 시간 저장
# uae = DAE_implicit(train=train_data, valid=valid_data, hidden_dim=500, num_epochs=100, learning_rate=0.01, dropout=0.5, reg_lambda=0.0001, device = device, activation= 'sigmoid')
# uae.fit()
# uae_prec, uae_recall, uae_ndcg = eval_implicit(uae, train_data, test_data, top_k)
# print("DAE 500: %f, %f, %f"%(uae_prec, uae_recall, uae_ndcg))
# print("time 분 :", (time.time() - start)/60.0)
# print("======================================")

# start = time.time()  # 시작 시간 저장
# uae = DAE_implicit(train=train_data, valid=valid_data, hidden_dim=500, num_epochs=150, learning_rate=0.01, dropout=0.5, reg_lambda=0.0001, device = device, activation= 'sigmoid')
# uae.fit()
# uae_prec, uae_recall, uae_ndcg = eval_implicit(uae, train_data, test_data, top_k)
# print("DAE 500: %f, %f, %f"%(uae_prec, uae_recall, uae_ndcg))
# print("time 분 :", (time.time() - start)/60.0)
# print("======================================")


# start = time.time()  # 시작 시간 저장
# uae = DAE_implicit(train=train_data, valid=valid_data, hidden_dim=1000, num_epochs=150, learning_rate=0.01, dropout=0.5, reg_lambda=0.0001, device = device, activation= 'sigmoid')
# uae.fit()
# uae_prec, uae_recall, uae_ndcg = eval_implicit(uae, train_data, test_data, top_k)
# print("DAE 1000: %f, %f, %f"%(uae_prec, uae_recall, uae_ndcg))
# print("time 분 :", (time.time() - start)/60.0)
# print("======================================")


# start = time.time()  # 시작 시간 저장
# uae = DAE_implicit(train=train_data, valid=valid_data, hidden_dim=1500, num_epochs=150, learning_rate=0.01, dropout=0.5, reg_lambda=0.0001, device = device, activation= 'sigmoid')
# uae.fit()
# uae_prec, uae_recall, uae_ndcg = eval_implicit(uae, train_data, test_data, top_k)
# print("DAE 1500: %f, %f, %f"%(uae_prec, uae_recall, uae_ndcg))
# print("time 분 :", (time.time() - start)/60.0)
# print("======================================")


# start = time.time()  # 시작 시간 저장
# uae = DAE_implicit(train=train_data, valid=valid_data, hidden_dim=2000, num_epochs=150, learning_rate=0.01, dropout=0.5, reg_lambda=0.0001, device = device, activation= 'sigmoid')
# uae.fit()
# uae_prec, uae_recall, uae_ndcg = eval_implicit(uae, train_data, test_data, top_k)
# print("DAE 2000: %f, %f, %f"%(uae_prec, uae_recall, uae_ndcg))
# print("time 분 :", (time.time() - start)/60.0)
# print("======================================")

######################################################################################################################################################
######################################################################################################################################################


# start = time.time()  # 시작 시간 저장
# uae = CDAE_implicit(train=train_data, valid=valid_data, hidden_dim=500, num_epochs=100, learning_rate=0.01, dropout=0.5, reg_lambda=0.0001, device = device, activation= 'sigmoid')
# uae.fit()
# uae_prec, uae_recall, uae_ndcg = eval_implicit(uae, train_data, test_data, top_k)
# print("CDAE 500: %f, %f, %f"%(uae_prec, uae_recall, uae_ndcg))
# print("time 분 :", (time.time() - start)/60.0)
# print("======================================")

# start = time.time()  # 시작 시간 저장
# uae = CDAE_implicit(train=train_data, valid=valid_data, hidden_dim=500, num_epochs=150, learning_rate=0.01, dropout=0.5, reg_lambda=0.0001, device = device, activation= 'sigmoid')
# uae.fit()
# uae_prec, uae_recall, uae_ndcg = eval_implicit(uae, train_data, test_data, top_k)
# print("CDAE 500: %f, %f, %f"%(uae_prec, uae_recall, uae_ndcg))
# print("time 분 :", (time.time() - start)/60.0)
# print("======================================")


# start = time.time()  # 시작 시간 저장
# uae = CDAE_implicit(train=train_data, valid=valid_data, hidden_dim=1000, num_epochs=150, learning_rate=0.01, dropout=0.5, reg_lambda=0.0001, device = device, activation= 'sigmoid')
# uae.fit()
# uae_prec, uae_recall, uae_ndcg = eval_implicit(uae, train_data, test_data, top_k)
# print("CDAE 1000: %f, %f, %f"%(uae_prec, uae_recall, uae_ndcg))
# print("time 분 :", (time.time() - start)/60.0)
# print("======================================")


# start = time.time()  # 시작 시간 저장
# uae = CDAE_implicit(train=train_data, valid=valid_data, hidden_dim=1500, num_epochs=150, learning_rate=0.01, dropout=0.5, reg_lambda=0.0001, device = device, activation= 'sigmoid')
# uae.fit()
# uae_prec, uae_recall, uae_ndcg = eval_implicit(uae, train_data, test_data, top_k)
# print("CDAE 1500: %f, %f, %f"%(uae_prec, uae_recall, uae_ndcg))
# print("time 분 :", (time.time() - start)/60.0)
# print("======================================")


# start = time.time()  # 시작 시간 저장
# uae = CDAE_implicit(train=train_data, valid=valid_data, hidden_dim=2000, num_epochs=150, learning_rate=0.01, dropout=0.5, reg_lambda=0.0001, device = device, activation= 'sigmoid')
# uae.fit()
# uae_prec, uae_recall, uae_ndcg = eval_implicit(uae, train_data, test_data, top_k)
# print("CDAE 2000: %f, %f, %f"%(uae_prec, uae_recall, uae_ndcg))
# print("time 분 :", (time.time() - start)/60.0)
# print("======================================")

# ######################################################################################################################################################
# ######################################################################################################################################################

