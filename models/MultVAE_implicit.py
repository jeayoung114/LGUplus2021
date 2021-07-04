"""
Variational Autoencoders for Collaborative Filtering, 
Dawen Liang et al.,
WWW 2018.
"""
import os
import math
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseRecommender
from dataloader.DataBatcher import DataBatcher

class MultVAE(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(MultVAE, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        if isinstance(model_conf['enc_dims'], str):
            model_conf['enc_dims'] = eval(model_conf['enc_dims'])
        self.enc_dims = [self.num_items] + model_conf['enc_dims']
        self.dec_dims = self.enc_dims[::-1]
        self.dims = self.enc_dims + self.dec_dims[1:]

        self.total_anneal_steps = model_conf['total_anneal_steps']
        self.anneal_cap = model_conf['anneal_cap']

        self.dropout = model_conf['dropout']
        self.reg = model_conf['reg']

        self.batch_size = model_conf['batch_size']
        self.test_batch_size = model_conf['test_batch_size']

        self.lr = model_conf['learning_rate']

        self.eps = 1e-6
        self.anneal = 0.
        self.update_count = 0

        self.device = device

        self.build_graph()

    def build_graph(self):
        self.encoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:])):
            if i == len(self.enc_dims[:-1]) - 1:
                d_out *= 2
            self.encoder.append(nn.Linear(d_in, d_out))
            if i != len(self.enc_dims[:-1]) - 1:
                self.encoder.append(nn.Tanh())

        self.decoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.dec_dims[:-1], self.dec_dims[1:])):
            self.decoder.append(nn.Linear(d_in, d_out))
            if i != len(self.dec_dims[:-1]) - 1:
                self.decoder.append(nn.Tanh())

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)

        # Send model to device (cpu or gpu)
        self.to(self.device)

    def forward(self, x):
        # encoder
        h = F.dropout(F.normalize(x), p=self.dropout, training=self.training)
        for layer in self.encoder:
            h = layer(h)

        # sample
        mu_q = h[:, :self.enc_dims[-1]]
        logvar_q = h[:, self.enc_dims[-1]:]  # log sigmod^2  batch x 200
        std_q = torch.exp(0.5 * logvar_q)  # sigmod batch x 200

        epsilon = torch.zeros_like(std_q).normal_(mean=0, std=0.01)
        sampled_z = mu_q + self.training * epsilon * std_q

        output = sampled_z
        for layer in self.decoder:
            output = layer(output)

        if self.training:
            kl_loss = ((0.5 * (-logvar_q + torch.exp(logvar_q) + torch.pow(mu_q, 2) - 1)).sum(1)).mean()
            return output, kl_loss
        else:
            return output

    def train_model(self, dataset, evaluator, early_stop, neptune, logger, config, neptune_prefix='', seed=2020):
        exp_config = config['Experiment']

        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir

        train_matrix = dataset.train_matrix
        train_users = train_matrix.shape[0]
        train_data_perm = np.arange(train_users)

        # for epoch
        start = time()
        for epoch in range(1, num_epochs + 1):
            self.train()

            epoch_loss = 0.0
            batch_loader = DataBatcher(train_data_perm, batch_size=self.batch_size, drop_remain=False, shuffle=False)
            num_batches = len(batch_loader)

            # ======================== Train
            epoch_train_start = time()
            for b, batch_idx in enumerate(batch_loader):
                batch_matrix = torch.FloatTensor(train_matrix[batch_idx].toarray()).to(self.device)

                if self.total_anneal_steps > 0:
                    self.anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
                else:
                    self.anneal = self.anneal_cap

                batch_loss = self.train_model_per_batch(batch_matrix)
                epoch_loss += batch_loss

                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, batch_loss))
            epoch_train_time = time() - epoch_train_start

            epoch_info = ['epoch=%3d' % epoch, 'loss=%.3f' % epoch_loss, 'train time=%.2f' % epoch_train_time]

            if neptune is not None:
                neptune.log_metric(neptune_prefix+'Loss', epoch_loss, epoch=epoch)

            # ======================== Evaluate
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                # evaluate model
                epoch_eval_start = time()

                valid_score = evaluator.evaluate(self)
                valid_score_str = ['%s=%.4f' % (k, valid_score[k]) for k in valid_score]

                updated, should_stop = early_stop.step(valid_score, epoch)

                if should_stop:
                    logger.info('Early stop triggered.')
                    break
                elif updated:
                    torch.save(self.state_dict(), os.path.join(log_dir, 'best_model.p'))

                epoch_eval_time = time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += valid_score_str

                if neptune is not None:
                    valid_neptune_dict = {neptune_prefix + k: v for k,v in valid_score.items()}
                    neptune.log_metric_from_dict(valid_neptune_dict, epoch=epoch)
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            if epoch % print_step == 0:
                logger.info(', '.join(epoch_info))

            if torch.isnan(epoch_loss):
                logger.info('Loss NAN. Train finish.')
                early_stop.best_score = {k: -1.0 for k in valid_score}
                break

        total_train_time = time() - start

        return early_stop.best_score, total_train_time


    def train_model_per_batch(self, batch_matrix):
        # zero grad
        self.optimizer.zero_grad()

        # model forwrad
        output, kl_loss = self.forward(batch_matrix)

        # loss        
        ce_loss = -(F.log_softmax(output, 1) * batch_matrix).sum(1).mean()

        loss = ce_loss + kl_loss * self.anneal

        # backward
        loss.backward()

        # step
        self.optimizer.step()

        self.update_count += 1

        return loss

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        batch_eval_pos = eval_pos_matrix[user_ids]
        
        with torch.no_grad():
            eval_matrix = torch.FloatTensor(batch_eval_pos.toarray()).to(self.device)
            eval_output = self.forward(eval_matrix).detach().cpu().numpy()
            
            if eval_items is not None:
                eval_output[np.logical_not(eval_items)]=float('-inf')
            else:
                eval_output[batch_eval_pos.nonzero()] = float('-inf')

            return eval_output

    def restore(self, log_dir):
        with open(os.path.join(log_dir, 'best_model.p'), 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)