"""
Neural Collaborative Filtering, 
He, Xiangnan, et al.,
WWW 2017.
"""
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GMF_implicit(torch.nn.Module):
    def __init__(self, train, valid, num_epochs, hidden_dim, learning_rate, reg_lambda, device, batch_size=2, neg_ratio=3, loss="CE"):
        super().__init__()
        self.train_mat = train
        self.valid_mat = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]

        self.num_epochs = num_epochs
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.batch_size = batch_size
        self.loss_function = loss

        self.device = device

        self.make_UIdataset(train, neg_ratio)
        self.build_graph()


    def make_UIdataset(self, train, neg_ratio):
        # {'사용자 ID' = [[positive 샘플, negative 샘플], [1, 1, 1, ..., 0, 0]]}
        UIdataset = {}
        for user_id, items_by_user in enumerate(train):
            UIdataset[user_id] = []
            # positive 샘플 구하기
            pos_item_ids = np.where(items_by_user > 0.5)[0]
            num_pos_samples = len(pos_item_ids)

            # 랜덤 negative 샘플링
            num_neg_samples = neg_ratio * num_pos_samples
            neg_items = np.where(items_by_user < 0.5)[0]
            neg_item_ids = np.random.choice(neg_items, min(num_neg_samples, len(neg_items)), replace=False)
            UIdataset[user_id].append(np.concatenate([pos_item_ids, neg_item_ids]))

            # label 저장
            pos_labels = np.ones(len(pos_item_ids))
            neg_labels = np.zeros(len(neg_item_ids))
            UIdataset[user_id].append(np.concatenate([pos_labels, neg_labels]))

        self.UIdataset = UIdataset
        print("데이터 생성 완료")


    def build_graph(self):
        # 사용자, 항목 임베딩 선언
        self.user_embedding = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.hidden_dim)
        self.item_embedding = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.hidden_dim)

        self.affine_output = nn.Linear(in_features=self.hidden_dim, out_features=1)

        # 최적화 방법 설정
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.reg_lambda)

        # 모델을 device로 보냄
        self.to(self.device)


    def forward(self, user_indices, item_indices):  
        # 해당하는 사용자와 항목 임베딩 갖고오기
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)

        # 원소별 곱
        element_product = torch.mul(user_embedding, item_embedding)

        # one-layer와 활성함수
        logits = self.affine_output(element_product)
        output = torch.sigmoid(logits)

        return output


    def fit(self):
        user_indices = np.arange(self.num_users)
        for epoch in range(0, self.num_epochs):
            start = time()
            epoch_loss = 0
            self.train()
            np.random.RandomState(12345).shuffle(user_indices)

            batch_num = int(len(user_indices) / self.batch_size) + 1
            for batch_idx in range(batch_num):
                batch_user_indices = user_indices[batch_idx*self.batch_size : (batch_idx+1)*self.batch_size]
                batch_user_ids = []
                batch_item_ids = []
                batch_labels = []
                for user_id in batch_user_indices:
                    item_ids = self.UIdataset[user_id][0]
                    labels = self.UIdataset[user_id][1]
                    user_ids = np.full(len(item_ids), user_id)
                    batch_user_ids.extend(user_ids.tolist())
                    batch_item_ids.extend(item_ids.tolist())
                    batch_labels.extend(labels.tolist())

                batch_item_ids = np.array(batch_item_ids)
                batch_labels = np.array(batch_labels)
                batch_loss = self.train_model_per_batch(batch_user_ids, batch_item_ids, batch_labels)
                if torch.isnan(batch_loss):
                    print('Loss NAN. Train finish.')
                    break
                epoch_loss += batch_loss

            if epoch % 10 == 0:
                print('epoch %d  loss: %.4f  training time per epoch:  %.2fs' % (epoch + 1, epoch_loss/batch_num, time() - start))
        print('final epoch %d  loss: %.4f  training time per epoch:  %.2fs' % (epoch + 1, epoch_loss/batch_num, time() - start))


    def train_model_per_batch(self, user_ids, item_ids, labels):  
        # 텐서 변환
        user_ids = torch.LongTensor(user_ids).to(self.device)
        item_ids = torch.LongTensor(item_ids).to(self.device)
        labels = torch.FloatTensor(labels).to(self.device)
        labels = labels.view(-1, 1)

        # grad 초기화
        self.optimizer.zero_grad()

        # 모델 forward
        output = self.forward(user_ids, item_ids)
        output = output.view(-1, 1)

        # loss 구함
        if self.loss_function == 'MSE':
            loss = F.mse_loss(output, labels).sum()
        else:
            loss = F.binary_cross_entropy(output, labels).sum()

        # 역전파
        loss.backward()

        # 최적화
        self.optimizer.step()

        return loss


    def predict(self, user_id, item_ids):
        self.eval()
        with torch.no_grad():
            user_ids = np.full(len(item_ids), user_id)
            user_ids = torch.LongTensor(user_ids).to(self.device)
            item_ids = torch.LongTensor(item_ids).to(self.device)
            eval_output = self.forward(user_ids, item_ids).detach().cpu().numpy()
            
        return eval_output.reshape(-1)
        