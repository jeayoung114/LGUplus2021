from utils import eval_sequential
import os
import math
import random
from time import time
from IPython.terminal.embed import embed
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader
from multiprocessing import Process, Queue

import sys
sys.path.append("..")

# https://github.com/pmixer/SASRec.pytorch 을 참고했습니다.


class BERTRec_sequential(torch.nn.Module):
    def __init__(self, user_train, user_valid, user_num, item_num, hidden, maxlen, n_layers, heads, mask_prob,
                 num_epochs, eval_every, early_stop_trial, learning_rate, reg_lambda, batch_size, device):
        super().__init__()

        self.user_train = user_train
        self.user_valid = user_valid
        self.user_num = user_num
        self.item_num = item_num
        self.hidden = hidden
        self.maxlen = maxlen
        self.n_layers = n_layers
        self.heads = heads
        self.mask_prob = mask_prob
        self.mask_token = item_num + 1

        self.num_epochs = num_epochs
        self.eval_every = eval_every
        self.early_stop_trial = early_stop_trial
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.batch_size = batch_size

        self.device = device

        self.build_graph()

    def build_graph(self):
        # BERT 정의 ()
        self.bert = BERT(self.maxlen, self.item_num, self.n_layers, self.heads, self.hidden)
        self.out = nn.Linear(self.bert.hidden, self.item_num + 1) # padding 값은 0으로 설정

        # Loss 설정
        self.criterion = nn.CrossEntropyLoss(ignore_index=0) # padding은 무시

        # 최적화 방법 설정
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.reg_lambda)

        # 모델을 device로 보냄
        self.to(self.device)

    def forward(self, x):
        # x = (batch, seq_len) -> (batch, seq_len, bert_dim)
        x = self.bert(x)
        # self.out = (batch, seq_len, item_num + 1)
        return self.out(x)

    def fit(self):
        train_loader = DataLoader(list(self.user_train.keys()), batch_size=self.batch_size, shuffle=True)

        top_k = 20
        best_recall = 0
        num_trials = 0
        for epoch in range(1, self.num_epochs+1):
            # Train
            self.train()
            for b, batch_idxes in enumerate(train_loader):
                tokens = np.zeros(shape=(self.batch_size, self.maxlen), dtype=np.int64)
                labels = np.zeros(shape=(self.batch_size, self.maxlen), dtype=np.int64)
                for i, batch_idx in enumerate(batch_idxes):
                    user_seq = self.user_train[int(batch_idx)][:self.maxlen]
                    mask_len = self.maxlen - len(user_seq)
                    for j, s in enumerate(user_seq):
                        prob = random.random()
                        s += 1 # 0부터 시작 -> 1부터 시작
                        if prob < self.mask_prob:
                            tokens[i, mask_len+j] = self.mask_token
                            labels[i, mask_len+j] = s
                        else:
                            tokens[i, mask_len+j] = s
                            labels[i, mask_len+j] = 0

                loss = self.train_model_per_batch(tokens, labels)

            # Valid
            if epoch % self.eval_every == 0:
                self.eval()
                ndcg, recall = eval_sequential(self, self.user_train, self.user_valid, None, self.user_num, self.item_num, top_k=20, mode='valid')

                if recall > best_recall:
                    best_recall = recall
                    torch.save(self.state_dict(), f"saves/{self.__class__.__name__}_best_model.pt")
                    num_trials = 0
                else:
                    num_trials += 1

                if num_trials >= self.early_stop_trial and self.early_stop_trial > 0:
                    print(f'Early stop at epoch:{epoch}')
                    self.restore()
                    break

                print(f'epoch {epoch} train_loss = {loss:.4f} valid_recall@{top_k} = {recall:.4f} valid_ndcg@{top_k} = {ndcg:.4f}')
            else:
                print(f'epoch {epoch} train_loss = {loss:.4f}')
        return

    def train_model_per_batch(self, tokens, labels):
        # 텐서 변환
        tokens = torch.LongTensor(tokens).to(self.device)
        labels = torch.LongTensor(labels).to(self.device)

        # grad 초기화
        self.optimizer.zero_grad()

        # 모델 forwrad
        logits = self.forward(tokens)

        # loss 구함
        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = self.criterion(logits, labels)

        # 역전파
        loss.backward()

        # 가중치 업데이트
        self.optimizer.step()

        return loss

    def predict(self, users, log_seqs, item_indices):
        self.eval()
        with torch.no_grad():
            tokens = np.zeros(shape=(len(log_seqs), self.maxlen), dtype=np.int64)

            tokens[:, :-1] = log_seqs[:, :-1]
            tokens += tokens.astype(np.bool).astype(np.long) # 0 부터 시작 -> 1부터 시작
            tokens[:, -1] = self.mask_token
            tokens = torch.LongTensor(tokens).to(self.device)

            logits = self.forward(tokens)
        return logits[0, -1, 1:]

    def restore(self):
        with open(f"saves/{self.__class__.__name__}_best_model.pt", 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)


class BERT(nn.Module):
    def __init__(self, maxlen, num_items, n_layers, heads, hidden, dropout=0.2):
        super().__init__()

        self.hidden = hidden

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=num_items+2, embed_size=self.hidden, maxlen=maxlen, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def init_weights(self):
        pass


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, maxlen, dropout=0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(maxlen=maxlen, d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):
    def __init__(self, maxlen, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(maxlen, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.attention_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.feed_forward_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # self-attention -> dropout -> residual -> layer normalization
        x = self.attention_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        # position-wise feed_forward -> dropout -> residual -> layer normalization
        x = self.feed_forward_sublayer(x, self.feed_forward)
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(x + self.dropout(sublayer(x)))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
