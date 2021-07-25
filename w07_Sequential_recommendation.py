# 기본 패키지 import
from models.TransRec_sequential import TransRec_sequential
from models.SASRec_sequential import SASRec_sequential
from models.BERTRec_sequential import BERTRec_sequential
from IPython.terminal.embed import embed
from utils import eval_sequential, load_data_sequential
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
"""
dataset loading
"""
dataset = "small"
user_train, user_valid, user_test, usernum, itemnum = load_data_sequential(dataset, implicit=True)
top_k = 100

"""
model 학습
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(device)


seed_everything(seed)
start = time.time()  # 시작 시간 저장
TransRec = TransRec_sequential(user_train, user_valid, user_num=usernum, item_num=itemnum, emb_dim=20, maxlen=1,
                               num_epochs=100, eval_every=10, early_stop_trial=3, learning_rate=0.001, reg_lambda=0.0, batch_size=128, device=device)
TransRec.fit()
TransRec_ndcg, TransRec_recall = eval_sequential(TransRec, user_train, user_valid, user_test, usernum, itemnum, top_k, mode='test')
print(f"[TransRec]\t Test_Recall@{top_k} = {TransRec_recall:.4f} Test_NDCG@{top_k} = {TransRec_ndcg:.4f}")
print("parmas :", sum(p.numel() for p in TransRec.parameters() if p.requires_grad))
print("time 초 :", (time.time() - start))
print("time 분 :", (time.time() - start)/60.0)
print("======================================")


seed_everything(seed)
start = time.time()  # 시작 시간 저장
SASRec = SASRec_sequential(user_train, user_valid, user_num=usernum, item_num=itemnum, hidden_dim=50, maxlen=50, num_blocks=2, num_heads=1,
                           num_epochs=100, eval_every=10, early_stop_trial=3, learning_rate=0.001, reg_lambda=0.0, batch_size=128, device=device)
SASRec.fit()
SASRec_ndcg, SASRec_recall = eval_sequential(SASRec, user_train, user_valid, user_test, usernum, itemnum, top_k, mode='test')
print(f"[SASRec]\t Test_Recall@{top_k} = {SASRec_recall:.4f} Test_NDCG@{top_k} = {SASRec_ndcg:.4f}")
print("parmas :", sum(p.numel() for p in SASRec.parameters() if p.requires_grad))
print("time 초 :", (time.time() - start))
print("time 분 :", (time.time() - start)/60.0)
print("======================================")


seed_everything(seed)
start = time.time()  # 시작 시간 저장
BERTRec = BERTRec_sequential(user_train, user_valid, user_num=usernum, item_num=itemnum, hidden=50, maxlen=50, n_layers=2, heads=1, mask_prob=0.2,
                             num_epochs=100, eval_every=10, early_stop_trial=3, learning_rate=0.0001, reg_lambda=0.0, batch_size=128, device=device)
BERTRec.fit()
BERTRec_ndcg, BERTRec_recall = eval_sequential(BERTRec, user_train, user_valid, user_test, usernum, itemnum, top_k, mode='test')
print(f"[BERTRec]\t Test_Recall@{top_k} = {BERTRec_recall:.4f} Test_NDCG@{top_k} = {BERTRec_ndcg:.4f}")
print("parmas :", sum(p.numel() for p in BERTRec.parameters() if p.requires_grad))
print("time 초 :", (time.time() - start))
print("time 분 :", (time.time() - start)/60.0)
print("======================================")


print(f"[TransRec]\t Test_Recall@{top_k} = {TransRec_recall:.4f} Test_NDCG@{top_k} = {TransRec_ndcg:.4f}")
print(f"[SASRec]\t Test_Recall@{top_k} = {SASRec_recall:.4f} Test_NDCG@{top_k} = {SASRec_ndcg:.4f}")
print(f"[BERTRec]\t Test_Recall@{top_k} = {BERTRec_recall:.4f} Test_NDCG@{top_k} = {BERTRec_ndcg:.4f}")
