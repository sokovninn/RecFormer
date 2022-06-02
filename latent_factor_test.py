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

train_data_raw = pd.read_csv("./train.csv", on_bad_lines='skip')
ingredient_name = pd.read_csv("./node_ingredient.csv", names=['a', 'b'])
ingredient_name_list = ingredient_name['a'].tolist()

train_data = pd.DataFrame(0, index=range(23547), columns=ingredient_name_list)

count = 0
with open('train.csv', newline='') as csvfile:
    raw_train_data = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in raw_train_data:
        list_data = row[0].split(",")

        tmp_result = list(repeat(0, len(ingredient_name_list)))
        for ingredient in list_data[:-1]:
            tmp_result[int(ingredient)] = 1
        train_data.loc[count] = tmp_result
        count += 1

# for i in range(23547):
#     tmp_result = list(repeat(0, len(ingredient_name)))
#     lst = train_data_raw.iloc[i]
#     for ingredient in lst[:-1]:
#         tmp_result[int(ingredient)] = i
#     train_data.loc[i] = tmp_result

#print(train_data.head())

SVD = TruncatedSVD(n_components=16)
matrix = SVD.fit_transform(train_data.transpose())
corr = np.corrcoef(matrix)
corr2 = corr[:200, :200]
plt.figure(figsize=(16, 10))
sns.heatmap(corr2)
plt.show()

kimchi = ingredient_name_list.index("pesto sauce")
corr_kimchi = corr[kimchi]
ingredient_title = train_data.columns
print(list(ingredient_title[(corr_kimchi >= 0.90)])[:50])

# for ingred in ingredient_name_list:
#     ingred_idx = ingredient_name_list.index(ingred)
#     coor_ingred = list(corr[ingred_idx])
#     coor_ingred.sort()
#     print(ingred +": ", end=" ")
#     print(coor_ingred[-1], end=", ")
#     print(coor_ingred[-2])