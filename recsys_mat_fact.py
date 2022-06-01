from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from itertools import repeat
import csv

#train_data_raw = pd.read_csv("./train.csv", on_bad_lines='skip')
ingredient_name = pd.read_csv("./nikita_request_lemmatize/node_ingredient.csv", names=['a', 'b'], encoding='latin-1')
ingredient_name_list = ingredient_name['a'].tolist()


question_data = pd.DataFrame(0, index=range(7848), columns=ingredient_name_list)
question_reference_lst = []
count = 0
with open("./nikita_request_lemmatize/validation_completion_question.csv", newline='') as csvfile:
    raw_df_svd_preds = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in raw_df_svd_preds:
        list_data = row[0].split(",")
        tmp_result = list(repeat(0, len(ingredient_name_list)))

        question_reference_lst.append(list_data)

        for idx, ingredient in enumerate(list_data):
            tmp_result[int(ingredient)] = 1
        question_data.loc[count] = tmp_result
        count += 1

completion_task_answer = []
count = 0
with open("./nikita_request_lemmatize/validation_completion_answer_extracted.csv", newline='') as csvfile:
    raw_df_svd_preds_answer = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in raw_df_svd_preds_answer:
        data = row[0].split(",")
        tmp_result = list(repeat(0, len(ingredient_name_list)))

        completion_task_answer.append(data[0])
        count += 1

train_data = pd.DataFrame(0, index=range(23547), columns=ingredient_name_list)
count = 0
with open('//nikita_request_lemmatize/train.csv', newline='') as csvfile:
    raw_train_data = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in raw_train_data:
        list_data = row[0].split(",")

        tmp_result = list(repeat(0, len(ingredient_name_list)))
        for ingredient in list_data[:-1]:   # Assumed that type of cuisine is not a factor in completion task -- TODO
            tmp_result[int(ingredient)] = 1
        train_data.loc[count] = tmp_result
        count += 1

print(train_data.shape) # (23547, 6714)

total_data = pd.concat([question_data, train_data])

matrix = total_data.values
ingredient_chosen_mean = np.mean(matrix, axis=1)
matrix_recipe_mean = matrix - ingredient_chosen_mean.reshape(-1, 1)

print(matrix)
print(matrix.shape)
print(ingredient_chosen_mean.shape)
print(matrix_recipe_mean.shape)
print(pd.DataFrame(matrix_recipe_mean, columns=total_data.columns).head())

for k in range(32):
    U, sigma, Vt = svds(matrix_recipe_mean, k=k+1)
    sigma = np.diag(sigma)

    svd_ingredient_predicted = np.dot(np.dot(U, sigma), Vt) + ingredient_chosen_mean.reshape(-1, 1)

    df_svd_preds = pd.DataFrame(svd_ingredient_predicted, columns=ingredient_name_list)
    #print(df_svd_preds.head())
    #print(df_svd_preds.shape)

    def recommend_ingredients(df_svd_preds, user_id, num_recommendations=1):
        # 현재는 index로 적용이 되어있으므로 user_id - 1을 해야함.
        user_row_number = user_id # 여기서는 미적용 -- DONGJAE

        # 최종적으로 만든 pred_df에서 사용자 index에 따라 영화 데이터 정렬 -> 영화 평점이 높은 순으로 정렬 됌
        #predictions = df_svd_preds.iloc[user_row_number].tolist().index(max(df_svd_preds.iloc[user_row_number].tolist()))
        predictions = df_svd_preds.iloc[user_row_number].tolist()

        final_prediction_idx = -1   # NULL
        final_prediction_val = -1000000000000000

        for i, pred in enumerate(predictions):
            if final_prediction_val < pred and (str(i) not in question_reference_lst[user_row_number]):
                final_prediction_val = pred
                final_prediction_idx = i

        return str(final_prediction_idx)

    correct_cnt = 0
    for i in range(7848):
        prediction = recommend_ingredients(df_svd_preds, i, 1)
        if prediction == completion_task_answer[i]:
            correct_cnt += 1

    print("K = " + str(k+1))
    print(correct_cnt)