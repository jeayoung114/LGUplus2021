import numpy as np

class WMF_ALS_implicit():
    def __init__(self, train, valid, n_features=20, reg_lambda =0.1, num_epochs = 550):
        self.train = train
        self.valid = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]
        self.num_epcohs = num_epochs
        self.n_features = n_features
        self.reg_lambda = reg_lambda

        self.y = np.zeros_like(self.train)
        for i, row in enumerate(self.train):
            self.y[i, np.where(row > 0.5)[0]] = 1.0
            self.y[i, np.where(row < 0.5)[0]] = 0.5

        self.user_factors = np.random.rand(self.num_users, self.n_features) * 0.01
        self.item_factors = np.random.rand(self.num_items, self.n_features) * 0.01


    def loss_function(self, y, train, predict, reg_lambda, user_factors, item_factors):
        predict_error = np.square(train - predict)
        loss = np.sum(y * predict_error) + reg_lambda * (np.linalg.norm(user_factors) + np.linalg.norm(item_factors) )
        return loss


    def optimize_user_factors(self, U, V, W, R, num_users, n_features, reg_lambda):
        vT = np.transpose(V)
        for u in range(num_users):
            tilde_Wu = np.diag(W[u])
            Ru_tilde_Wu_V = np.matmul(np.matmul(R[u], tilde_Wu), V)

            VT_tilde_Wu_V = np.matmul(np.matmul(vT, tilde_Wu), V)
            sigma_Wi = np.sum(W[u])
            reg_Wi_I = np.dot(reg_lambda*sigma_Wi, np.identity(n_features))

            U[u] = np.matmul(Ru_tilde_Wu_V, np.linalg.inv(VT_tilde_Wu_V + reg_Wi_I))


    def optimize_item_factors(self, U, V, W, R, num_items, n_features, reg_lambda):
        uT = np.transpose(U)
        for i in range(num_items):
            tilde_Wi = np.diag(W[:,i])
            RTj_tilde_Wi_U = np.matmul(np.matmul(np.transpose(R[:,i]), tilde_Wi), U)

            UT_tilde_Wi_U = np.matmul(np.matmul(uT, tilde_Wi), U)
            sigma_Wu = np.sum(W[:,i])
            reg_Wu_I = np.dot(reg_lambda*sigma_Wu, np.identity(n_features))

            V[i] = np.matmul(RTj_tilde_Wi_U, np.linalg.inv(UT_tilde_Wi_U + reg_Wu_I))


    def fit(self):
        # U와 V를 업데이트 함.
        for epoch in range(self.num_epcohs):
            predict = np.matmul(self.user_factors, np.transpose(self.item_factors))
            loss = self.loss_function(self.y, self.train, predict, self.reg_lambda, self.user_factors, self.item_factors)

            self.optimize_user_factors(U=self.user_factors, V=self.item_factors, W=self.y, R=self.train, num_users=self.num_users, n_features=self.n_features, reg_lambda=self.reg_lambda)
            self.optimize_item_factors(U=self.user_factors, V=self.item_factors, W=self.y, R=self.train, num_items=self.num_items, n_features=self.n_features, reg_lambda=self.reg_lambda)
            
            print("epoch %d, loss: %f"%(epoch, loss))

        self.reconstructed = np.matmul(self.user_factors, np.transpose(self.item_factors))

    def predict(self, user_id, item_ids):
        return self.reconstructed[user_id, item_ids]
