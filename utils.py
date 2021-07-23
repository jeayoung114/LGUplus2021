import os
import math
import ast
import numpy as np
import pandas as pd
import pickle
import copy

from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss
from google_drive_downloader import GoogleDriveDownloader as gdd
from tqdm import tqdm
from IPython import embed


# 2d array를 dictionary로 만듦
# input: [[user_id, item_id, timestamp], ...] 형태의 numpy array
# output: {user_id: [item1, item2, ......], ...} 형태의 dictionary
def make_to_dict(data):
    data_dict = {}
    cur_user = -1
    tmp_user = []
    for row in data:
        user_id, item_id = row[0], row[1]
        if user_id != cur_user:
            if cur_user != -1:
                tmp = np.asarray(tmp_user)
                tmp_items = tmp[:, 1]
                data_dict[cur_user] = list(tmp_items)

            cur_user = user_id
            tmp_user = []
        tmp_user.append(row)

    if cur_user != -1:
        tmp = np.asarray(tmp_user)
        tmp_items = tmp[:, 1]
        data_dict[cur_user] = list(tmp_items)

    return data_dict


"""
dataset 관련 함수
"""


def load_data(data_name, implicit=True):
    data_path = './data/%s' % (data_name)
    if not os.path.exists(data_path):
        if 'small' in data_name:
            # https://drive.google.com/file/d/1_HFBNRk-FUOO1nquQVfWbD1IiqnWNzOW/view?usp=sharing
            gdd.download_file_from_google_drive(
                file_id='1_HFBNRk-FUOO1nquQVfWbD1IiqnWNzOW',
                dest_path=data_path,
            )
        elif '2m' in data_name:
            # https://drive.google.com/file/d/1zqh0MSl3eNeW6NP0O5aotSUM6Ihg47qz/view?usp=sharing
            gdd.download_file_from_google_drive(
                file_id='1zqh0MSl3eNeW6NP0O5aotSUM6Ihg47qz',
                dest_path=data_path,
            )
        else:  # 5m
            # https://drive.google.com/file/d/1WevyoBY_E7mKd2RwL1_GAgeDgtJQOlue/view?usp=sharing
            gdd.download_file_from_google_drive(
                file_id='1WevyoBY_E7mKd2RwL1_GAgeDgtJQOlue',
                dest_path=data_path,
            )
        print("데이터 다운로드 완료!")

    # 데이터셋 불러오기
    column_names = ['user_id', 'item_id', 'rating', 'timestamp', 'title', 'people', 'country', 'genre']
    movie_data = pd.read_csv(data_path, names=column_names)

    if implicit:
        # print("Binarize ratings greater than or equal to %.f" % 0) ###############################################
        # movie_data = movie_data[movie_data['ratings'] >= 0] ###############################################
        movie_data['rating'] = 1

    # 전체 데이터셋의 user, item 수 확인
    user_list = list(movie_data['user_id'].unique())
    item_list = list(movie_data['item_id'].unique())

    num_users = len(user_list)
    num_items = len(item_list)
    num_ratings = len(movie_data)

    user_id_dict = {old_uid: new_uid for new_uid, old_uid in enumerate(user_list)}
    movie_data.user_id = [user_id_dict[x] for x in movie_data.user_id.tolist()]
    print(f"# of users: {num_users},  # of items: {num_items},  # of ratings: {num_ratings}")

    # movie title와 id를 매핑할 dict를 생성
    item_id_title = movie_data[['item_id', 'title']]
    item_id_title = item_id_title.drop_duplicates()
    idx2title = {}
    for i, c in zip(item_id_title['item_id'], item_id_title['title']):
        idx2title[i] = c

    # user 별 train, valid, test 나누기
    movie_data = movie_data[['user_id', 'item_id', 'rating']]
    movie_data = movie_data.sort_values(by="user_id", ascending=True)

    train_valid, test = train_test_split(movie_data, test_size=0.2, stratify=movie_data['user_id'], random_state=1234)
    train, valid = train_test_split(train_valid, test_size=0.1, stratify=train_valid['user_id'], random_state=1234)

    train = train.to_numpy()
    valid = valid.to_numpy()
    test = test.to_numpy()

    matrix = sparse.lil_matrix((num_users, num_items))
    for (u, i, r) in train:
        matrix[u, i] = r
    train = sparse.csr_matrix(matrix)

    matrix = sparse.lil_matrix((num_users, num_items))
    for (u, i, r) in valid:
        matrix[u, i] = r
    valid = sparse.csr_matrix(matrix)

    matrix = sparse.lil_matrix((num_users, num_items))
    for (u, i, r) in test:
        matrix[u, i] = r
    test = sparse.csr_matrix(matrix)

    return train.toarray(), valid.toarray(), test.toarray(), idx2title
    # return train, valid, test, idx2title


def load_data_sequential(data_name, implicit=True):
    data_path = './data/%s' % (data_name)
    if not os.path.exists(data_path):
        if 'small' in data_name:
            # https://drive.google.com/file/d/1_HFBNRk-FUOO1nquQVfWbD1IiqnWNzOW/view?usp=sharing
            gdd.download_file_from_google_drive(
                file_id='1_HFBNRk-FUOO1nquQVfWbD1IiqnWNzOW',
                dest_path=data_path,
            )
        elif '2m' in data_name:
            # https://drive.google.com/file/d/1zqh0MSl3eNeW6NP0O5aotSUM6Ihg47qz/view?usp=sharing
            gdd.download_file_from_google_drive(
                file_id='1zqh0MSl3eNeW6NP0O5aotSUM6Ihg47qz',
                dest_path=data_path,
            )
        else:  # 5m
            # https://drive.google.com/file/d/1WevyoBY_E7mKd2RwL1_GAgeDgtJQOlue/view?usp=sharing
            gdd.download_file_from_google_drive(
                file_id='1WevyoBY_E7mKd2RwL1_GAgeDgtJQOlue',
                dest_path=data_path,
            )
        print("데이터 다운로드 완료!")

    # 데이터셋 불러오기
    column_names = ['user_id', 'item_id', 'rating', 'timestamp', 'title', 'people', 'country', 'genre']
    movie_data = pd.read_csv(data_path, names=column_names)

    if implicit:
        # print("Binarize ratings greater than or equal to %.f" % 0) ###############################################
        # movie_data = movie_data[movie_data['ratings'] >= 0] ###############################################
        movie_data['rating'] = 1

    # 전체 데이터셋의 user, item 수 확인
    user_list = list(movie_data['user_id'].unique())
    item_list = list(movie_data['item_id'].unique())

    num_users = len(user_list)
    num_items = len(item_list)
    num_ratings = len(movie_data)

    user_id_dict = {old_uid: new_uid for new_uid, old_uid in enumerate(user_list)}
    movie_data.user_id = [user_id_dict[x] for x in movie_data.user_id.tolist()]
    print(f"# of users: {num_users},  # of items: {num_items},  # of ratings: {num_ratings}")

    # movie title와 id를 매핑할 dict를 생성
    item_id_title = movie_data[['item_id', 'title']]
    item_id_title = item_id_title.drop_duplicates()
    idx2title = {}
    for i, c in zip(item_id_title['item_id'], item_id_title['title']):
        idx2title[i] = c

    # user 별 train, valid, test 나누기
    movie_data = movie_data[['user_id', 'item_id', 'rating', 'timestamp']]
    movie_data = movie_data.sort_values(by=['user_id', 'timestamp'], ascending=True)

    movie_data_np = movie_data.to_numpy()

    user_train = dict()
    user_valid = dict()
    user_test = dict()

    for (u, i, r, t) in movie_data_np:
        if user_train.get(u) is None:
            user_train[u] = []
        user_train[u].append(i)

    for u in user_train:
        user_valid[u] = [user_train[u][-2]]
        user_test[u] = [user_train[u][-1]]
        user_train[u] = user_train[u][:-2]

    return user_train, user_valid, user_test, num_users, num_items


def load_data_session(data_name):
    train_path = f'./data/melon_train_{data_name}.tsv'
    test_path = f'./data/melon_test_{data_name}.tsv'

    train_df = pd.read_csv(train_path, sep='\t')
    test_df = pd.read_csv(test_path, sep='\t')

    return train_df, test_df


def load_data_CTR(data_name, use_features, pos_threshold=6):
    data_path = './data/%s' % (data_name)
    if not os.path.exists(data_path):
        if 'small' in data_name:
            # https://drive.google.com/file/d/1_HFBNRk-FUOO1nquQVfWbD1IiqnWNzOW/view?usp=sharing
            gdd.download_file_from_google_drive(
                file_id='1_HFBNRk-FUOO1nquQVfWbD1IiqnWNzOW',
                dest_path=data_path,
                showsize=True,
            )
        elif '2m' in data_name:
            # https://drive.google.com/file/d/1hGtY8X9ERgwgUH37CIx-o7lryY6V286e/view?usp=sharing
            gdd.download_file_from_google_drive(
                file_id='1hGtY8X9ERgwgUH37CIx-o7lryY6V286e',
                dest_path=data_path,
                showsize=True,
            )
        else:  # 5m
            # https://drive.google.com/file/d/1C-qJgsu5cvzZ3ajcXCknG6tHwG_OC4lg/view?usp=sharing
            gdd.download_file_from_google_drive(
                file_id='1C-qJgsu5cvzZ3ajcXCknG6tHwG_OC4lg',
                dest_path=data_path,
                showsize=True,
            )
        print("데이터 다운로드 완료!")

    # 데이터셋 불러오기
    column_names = ['user_id', 'item_id', 'rating', 'timestamp', 'title', 'people', 'country', 'genre']
    movie_data = pd.read_csv(data_path, names=column_names)
    movie_data = movie_data.fillna('[]')

    # 평점이 X점 이상인 데이터는 1로, 나머지는 0으로 설정한다.
    movie_data['rating'] = movie_data['rating'].apply(lambda x: 1 if x >= pos_threshold else 0)

    # 전체 데이터셋의 user, item 수 확인
    user_list = list(movie_data['user_id'].unique())
    item_list = list(movie_data['item_id'].unique())

    num_users = len(user_list)
    num_items = len(item_list)
    num_ratings = len(movie_data)

    # 전체 데이터셋을 돌면서 모든 종류의 영화 장르, 국가, 배우 확인
    all_genre_dict = {'None': 0}
    all_country_dict = {'None': 0}
    all_people_dict = {'None': 0}

    dict_path = data_path + '_dict'
    if not os.path.exists(dict_path):
        try:
            if 'small' in data_name:
                raise Exception('Make small data dictionary')
            if '2m' in data_name:
            # https://drive.google.com/file/d/1yY_T_efF0n4JZpyArcEHuhighVzJt954/view?usp=sharing
                gdd.download_file_from_google_drive(
                    file_id='1yY_T_efF0n4JZpyArcEHuhighVzJt954',
                    dest_path=dict_path,
                    showsize=True,
                )
            elif '5m' in data_name:  # 5m
            # https://drive.google.com/file/d/1VBWAiogRzjvp4TgqG5zIMPxj11BuiTS8/view?usp=sharing
                gdd.download_file_from_google_drive(
                    file_id='1VBWAiogRzjvp4TgqG5zIMPxj11BuiTS8',
                    dest_path=data_path,
                    showsize=True,
                )
            all_genre_dict, all_country_dict, all_people_dict = pickle.load(open(dict_path, 'rb'))
        except:
            for index, row in tqdm(movie_data.iterrows(), total=len(movie_data), desc='check genre, country, people', dynamic_ncols=True):
                genres = row["genre"]
                coutries = row["country"]
                people = row["people"]
                genres = ast.literal_eval(genres)
                coutries = ast.literal_eval(coutries)
                people = ast.literal_eval(people)

                for genre in genres:
                    if all_genre_dict.get(genre) is None:
                        all_genre_dict[genre] = len(all_genre_dict)
                for country in coutries:
                    if all_country_dict.get(country) is None:
                        all_country_dict[country] = len(all_country_dict)
                for person in people:
                    if all_people_dict.get(person) is None:
                        all_people_dict[person] = len(all_people_dict)
            pickle.dump([all_genre_dict, all_country_dict, all_people_dict], open(dict_path, 'wb'), protocol=4)
    else:
        all_genre_dict, all_country_dict, all_people_dict = pickle.load(open(dict_path, 'rb'))

    num_genres = len(all_genre_dict)
    num_countries = len(all_country_dict)
    num_people = len(all_people_dict)

    # 장르, 국가, 배우, 제목과 아이템id를 매핑할 dict를 생성합니다. 이미 있는 아이템은 추가하지 않습니다.
    idx2title = {}
    idx2genre = {}
    idx2country = {}
    idx2people = {}

    item_data = movie_data[['item_id', 'title', 'people', 'country', 'genre']]
    item_data = item_data.drop_duplicates()
    for idx, title, people, country, genre in zip(item_data['item_id'], item_data['title'], item_data['people'], item_data['country'], item_data['genre']):
        idx2title[idx] = title
        idx2genre[idx] = people
        idx2country[idx] = country
        idx2people[idx] = genre

    # train, valid, test 나누기
    train_valid, test = train_test_split(movie_data, test_size=0.2, stratify=movie_data['rating'], random_state=1234)
    train, valid = train_test_split(train_valid, test_size=0.1, stratify=train_valid['rating'], random_state=1234)

    # 전체 데이터셋을 돌면서 matrix 생성하는 함수를 정의합니다.
    num_fields = len(use_features)

    def df_to_array(df):
        final_array = np.zeros((len(df), num_fields))
        for index, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df), desc='convert df to array', dynamic_ncols=True):
            features = []

            for feature in use_features:
                # user_id
                if feature == "user_id":
                    user_id = row["user_id"]
                    features.append(user_id)
                # item_id
                if feature == "item_id":
                    item_id = row["item_id"]
                    features.append(item_id)
                # genre
                if feature == "genre":
                    genres = row["genre"]
                    genres = ast.literal_eval(genres)
                    genre_id = all_genre_dict[genres[0]] if len(genres) > 0 else 0
                    features.append(genre_id)

                # country
                if feature == "country":
                    coutries = row["country"]
                    coutries = ast.literal_eval(coutries)
                    country_id = all_country_dict[coutries[0]] if len(coutries) > 0 else 0
                    features.append(country_id)

                # people
                if feature == "people":
                    people = row["people"]
                    people = ast.literal_eval(people)
                    people_id = all_people_dict[people[0]] if len(people) > 0 else 0
                    features.append(people_id)

            final_array[index] = features

        return final_array

    array_path = data_path + '_np'
    if not os.path.exists(array_path):
        try:
            if 'small' in data_name:
                raise Exception('Make small data array')
            if '2m' in data_name:
            # https://drive.google.com/file/d/19F-_E6Fs0rYym_NiL_5O0MSTiJlh-OeZ/view?usp=sharing
                gdd.download_file_from_google_drive(
                    file_id='19F-_E6Fs0rYym_NiL_5O0MSTiJlh-OeZ',
                    dest_path=array_path,
                    showsize=True,
                )
            elif '5m' in data_name:  # 5m
            # https://drive.google.com/file/d/1swE9YVLjX3q7tnibmIGyxfzgFm181q66/view?usp=sharing
                gdd.download_file_from_google_drive(
                    file_id='1swE9YVLjX3q7tnibmIGyxfzgFm181q66',
                    dest_path=array_path,
                    showsize=True,
                )
            train_arr, valid_arr, test_arr = pickle.load(open(array_path, 'rb'))
        except:
            train_arr = df_to_array(train)
            valid_arr = df_to_array(valid)
            test_arr = df_to_array(test)
            pickle.dump([train_arr, valid_arr, test_arr], open(array_path, 'wb'), protocol=4)
    else:
        train_arr, valid_arr, test_arr = pickle.load(open(array_path, 'rb'))

    train_rating = train['rating'].values
    valid_rating = valid['rating'].values
    test_rating = test['rating'].values
    field_dims = []
    for feature in use_features:
        if feature == 'user_id': field_dims.append(num_users)
        if feature == 'item_id': field_dims.append(num_items)
        if feature == 'genre': field_dims.append(num_genres)
        if feature == 'country': field_dims.append(num_countries)
        if feature == 'people': field_dims.append(num_people)

    return train_arr, train_rating, valid_arr, valid_rating, test_arr, test_rating, field_dims


# Precision, Recall, NDCG@K 평가
# input
#    - pred_u: 예측 값으로 정렬 된 item index
#    - target_u: test set의 item index
#    - top_k: top-k에서의 k 값
# output: prec_k, recall_k, ndcg_k의 점수
def compute_metrics(pred_u, target_u, top_k):
    pred_k = pred_u[:top_k]
    num_target_items = len(target_u)

    hits_k = [(i + 1, item) for i, item in enumerate(pred_k) if item in target_u]
    num_hits = len(hits_k)

    idcg_k = 0.0
    for i in range(1, min(num_target_items, top_k) + 1):
        idcg_k += 1 / math.log(i + 1, 2)

    dcg_k = 0.0
    for idx, item in hits_k:
        dcg_k += 1 / math.log(idx + 1, 2)

    prec_k = num_hits / top_k
    recall_k = num_hits / min(num_target_items, top_k)
    ndcg_k = dcg_k / idcg_k

    return prec_k, recall_k, ndcg_k


def eval_implicit(model, train_data, test_data, top_k):
    prec_list = []
    recall_list = []
    ndcg_list = []

    if 'Item' in model.__class__.__name__:
        num_users, num_items = train_data.shape
        pred_matrix = np.zeros((num_users, num_items))

        for item_id in range(len(train_data.T)):
            train_by_item = train_data[:, item_id]
            missing_user_ids = np.where(train_by_item == 0)[0]  # missing user_id

            pred_u_score = model.predict(item_id, missing_user_ids)
            pred_matrix[missing_user_ids, item_id] = pred_u_score

        for user_id in range(len(train_data)):
            train_by_user = train_data[user_id]
            missing_item_ids = np.where(train_by_user == 0)[0]  # missing item_id

            pred_u_score = pred_matrix[user_id, missing_item_ids]
            pred_u_idx = np.argsort(pred_u_score)[::-1]
            pred_u = missing_item_ids[pred_u_idx]

            test_by_user = test_data[user_id]
            target_u = np.where(test_by_user >= 0.5)[0]

            prec_k, recall_k, ndcg_k = compute_metrics(pred_u, target_u, top_k)
            prec_list.append(prec_k)
            recall_list.append(recall_k)
            ndcg_list.append(ndcg_k)
    else:
        for user_id in range(len(train_data)):
            train_by_user = train_data[user_id]
            missing_item_ids = np.where(train_by_user == 0)[0]  # missing item_id

            pred_u_score = model.predict(user_id, missing_item_ids)
            pred_u_idx = np.argsort(pred_u_score)[::-1]  # 내림차순 정렬
            pred_u = missing_item_ids[pred_u_idx]

            test_by_user = test_data[user_id]
            target_u = np.where(test_by_user >= 0.5)[0]

            prec_k, recall_k, ndcg_k = compute_metrics(pred_u, target_u, top_k)
            prec_list.append(prec_k)
            recall_list.append(recall_k)
            ndcg_list.append(ndcg_k)

    return np.mean(prec_list), np.mean(recall_list), np.mean(ndcg_list)


def eval_explicit(model, train_data, test_data):
    rmse_list = []
    if 'Item' in model.__class__.__name__:
        num_users, num_items = train_data.shape
        pred_matrix = np.zeros((num_users, num_items))

        for item_id in range(len(train_data.T)):
            train_by_item = test_data[:, item_id]
            missing_user_ids = np.where(train_by_item >= 0.5)[0]

            pred_u_score = model.predict(item_id, missing_user_ids)
            pred_matrix[missing_user_ids, item_id] = pred_u_score

        for user_id in range(len(train_data)):
            test_by_user = test_data[user_id]
            target_u = np.where(test_by_user >= 0.5)[0]
            target_u_score = test_by_user[target_u]

            pred_u_score = pred_matrix[user_id, target_u]

            rmse = mean_squared_error(target_u_score, pred_u_score)
            rmse_list.append(rmse)
    else:
        for user_id in range(len(train_data)):
            test_by_user = test_data[user_id]
            target_u = np.where(test_by_user >= 0.5)[0]
            target_u_score = test_by_user[target_u]

            pred_u_score = model.predict(user_id, target_u)

            # RMSE 계산
            rmse = mean_squared_error(target_u_score, pred_u_score)
            rmse_list.append(rmse)

    return np.mean(rmse_list)


def eval_sequential(model, train_data, valid_data, test_data, usernum, itemnum, top_k=100, mode='valid'):
    [train_data, valid_data, test_data, usernum, itemnum] = copy.deepcopy([train_data, valid_data, test_data, usernum, itemnum])

    NDCG = 0.0
    eval_user = 0.0
    HT = 0.0
    users = range(usernum)
    for u in tqdm(users, desc=f'{mode}', dynamic_ncols=True):
        if len(train_data[u]) < 1 or len(train_data[u]) < 1: continue

        seq = np.zeros([model.maxlen], dtype=np.int32)
        idx = model.maxlen - 1
        if mode == 'test':
            seq[idx] = valid_data[u][0]

        for i in reversed(train_data[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = train_data[u]
        target_item = test_data[u][0] if mode == 'test' else valid_data[u][0]
        if mode == 'test':
            rated.append(valid_data[u][0])
        item_idx = list(range(itemnum))

        predictions = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions[rated] = -np.inf

        sorted_items = np.argsort(-predictions.cpu().detach().numpy())

        rank = np.where(sorted_items == target_item)[0][0]
        eval_user += 1

        if rank < top_k:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / eval_user, HT / eval_user

def eval_implicit_CTR(model, test_data, test_label):

    predict_test=model.predict(test_data)

    AUC=roc_auc_score(test_label, predict_test)
    logloss=log_loss(test_label, predict_test)
    return AUC, logloss

def eval_session(model, train_df, test_df, top_k=20):
    session_key='SessionId'
    item_key='ItemId'
    time_key='Time'

    actions=len(test_df)
    sessions=len(test_df[session_key].unique())
    print('START evaluation of ', actions, ' actions in ', sessions, ' sessions')

    items_to_predict=train_df[item_key].unique()
    prev_iid, prev_sid=-1, -1

    recall_list=[]
    mrr_list=[]

    for i in tqdm(range(actions), desc="Eval...", dynamic_ncols=True):
        # Get sid, iid, ts of current row
        sid=test_df[session_key].values[i]
        iid=test_df[item_key].values[i]
        ts=test_df[time_key].values[i]

        # if new session
        if prev_sid != sid:
            prev_sid=sid
        else:
            preds=model.predict_next(sid, prev_iid, items_to_predict, timestamp=ts)
            preds[np.isnan(preds)]=0
            preds.sort_values(ascending=False, inplace=True)

            # Get top_k items
            top_k_preds=preds.iloc[:top_k]
            if iid in top_k_preds.index:
                rank=top_k_preds.index.get_loc(iid) + 1
                hit=1
            else:
                rank, hit=0, 0
            recall_list.append(hit)
            mrr_list.append(rank)
        embed()
    recall=np.mean(recall_list)
    mrr=np.mean(mrr_list)
    return recall, mrr
