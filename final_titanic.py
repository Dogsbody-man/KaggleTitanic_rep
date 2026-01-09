import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader

train_df = pd.read_csv('titanic/train.csv')
test_df = pd.read_csv('titanic/test.csv')
origin_train = train_df.copy()
origin_test = test_df.copy()
train_y = train_df['Survived']


'''заполню пробелы в данных'''
'''пропуск в Fare в test'''
mean_Fare = train_df['Fare'].mean()
test_df.loc[(test_df['Fare'].isnull()), 'Fare']=mean_Fare

'''пропуск Embarked в train заменим самым частым - S'''
train_df.loc[(train_df['Embarked'].isnull()), 'Embarked']='S'

'''чтобы заполнить Age надо поработать с именами, сгруппировать'''
'''из имен выделим только титлы по сплиту'''
for i in train_df:
    train_df['Title'] = train_df.Name.str.extract(r'([A-Za-z]+)\.')
for i in test_df:
    test_df['Title'] = test_df.Name.str.extract(r'([A-Za-z]+)\.')

'''всего титлов 18 в train и test в сумме, сгруппируем до 4'''
train_df['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                    ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'], inplace=True)
test_df['Title'].replace(['Ms','Dr','Col','Rev','Dona'],
                    ['Miss','Mr','Other','Other','Mrs'], inplace=True)

'''заменим сразу имена на цифры (label encoding) и заодно Sex'''
for dataset in [train_df, test_df]:
    dataset['Sex'].replace({'male': 0, 'female': 1}, inplace=True)
    dataset['Title'].replace({'Mr': 0, 'Mrs': 1, 'Miss' : 2, 'Master': 3, 'Other': 4}, inplace=True)
   

'''Заполним Age средним значеним по значению для Tilte и Pclass вместе в train'''
for dataset in [train_df, test_df]:
    dataset.loc[(dataset.Age.isnull()) & (dataset.Title==3) & (dataset.Pclass==1), 'Age']=5
    dataset.loc[(dataset.Age.isnull()) & (dataset.Title==3) & (dataset.Pclass==2), 'Age']=2
    dataset.loc[(dataset.Age.isnull()) & (dataset.Title==3) & (dataset.Pclass==3), 'Age']=5.5
    dataset.loc[(dataset.Age.isnull()) & (dataset.Title==2) & (dataset.Pclass==1), 'Age']=30
    dataset.loc[(dataset.Age.isnull()) & (dataset.Title==2) & (dataset.Pclass==2), 'Age']=23
    dataset.loc[(dataset.Age.isnull()) & (dataset.Title==2) & (dataset.Pclass==3), 'Age']=16
    dataset.loc[(dataset.Age.isnull()) & (dataset.Title==0) & (dataset.Pclass==1), 'Age']=42
    dataset.loc[(dataset.Age.isnull()) & (dataset.Title==0) & (dataset.Pclass==2), 'Age']=33
    dataset.loc[(dataset.Age.isnull()) & (dataset.Title==0) & (dataset.Pclass==3), 'Age']=29
    dataset.loc[(dataset.Age.isnull()) & (dataset.Title==1) & (dataset.Pclass==1), 'Age']=41
    dataset.loc[(dataset.Age.isnull()) & (dataset.Title==1) & (dataset.Pclass==2), 'Age']=34
    dataset.loc[(dataset.Age.isnull()) & (dataset.Title==1) & (dataset.Pclass==3), 'Age']=34
    dataset.loc[(dataset.Age.isnull()) & (dataset.Title==4) & (dataset.Pclass==2), 'Age']=51
    dataset.loc[(dataset.Age.isnull()) & (dataset.Title==4) & (dataset.Pclass==2), 'Age']=43

'''One hot encoding для Embarked'''
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoding_train = encoder.fit_transform(train_df[['Embarked']]).astype(int)
encoding_test = encoder.transform(test_df[['Embarked']]).astype(int)
train_df[encoder.get_feature_names_out()] = encoding_train
test_df[encoder.get_feature_names_out()] = encoding_test

'''Создам IsAlone из SibSp и Parch'''
for dataset in [train_df, test_df]:
    dataset['Family'] = dataset['SibSp'] + dataset['Parch']
    dataset['IsAlone']=0
    dataset.loc[dataset.Family==0, 'IsAlone']=1
    
'''нормализация и масштабирование Age'''
scaler = QuantileTransformer(n_quantiles=100)
train_scaled = scaler.fit_transform(train_df[['Age']])
test_scaled = scaler.transform(test_df[['Age']])
train_df['Age'] = train_scaled
test_df['Age'] = test_scaled

'''заменим в Fare значения на 1-4 путем того, что медиана это 14$'''
for dataset in [train_df, test_df]:
    dataset['Fare_label'] = 0
    dataset.loc[dataset['Fare']<=7.91, 'Fare_label']=0
    dataset.loc[(dataset['Fare']>7.91) & (dataset['Fare']<=14.454), 'Fare_label']=1
    dataset.loc[(dataset['Fare']>14.454) & (dataset['Fare']<=31), 'Fare_label']=2
    dataset.loc[(dataset['Fare']>31) & (dataset['Fare']<=513), 'Fare_label']=3
    dataset['Fare_label'] = dataset['Fare_label'].astype(int)
    dataset.drop(['Name', 'PassengerId', 'Cabin', 'Family', 'Ticket', 'Embarked', 'Fare'], axis=1, inplace=True)
    dataset['Age'] = round(dataset['Age'], 3)
train_df.drop(['Survived'], axis=1, inplace=True)


'''обучение первой модели на train и проверка score без кросс валидации'''
model = LogisticRegression()
# model.fit(train_df, train_y)
# probs = model.predict_proba(test_df)[:, 1]
# pred = model.predict(test_df)

'''после gridsearch лучшая модель knn'''
model2 = KNeighborsClassifier(n_neighbors=9, metric='manhattan', weights='uniform')
# model2.fit(train_df, train_y)
# probs2 = model2.predict_proba(test_df)[:, 1]
# pred2 = model2.predict(test_df)

'''тренировка DecisionTreeClassifier'''
model3 = DecisionTreeClassifier()
# model3.fit(train_df, train_y)
# probs3 = model3.predict_proba(test_df)
# pred3 = model3.predict(test_df)

'''после gridsearch лучшая модель RandomForest'''
model4 = RandomForestClassifier(max_depth=20, max_features='log2', min_samples_leaf=3,
                              min_samples_split=7, n_estimators=300)
# model4.fit(train_df, train_y)
# probs4 = model4.predict_proba(test_df)[:, 1]
# pred4 = model4.predict(test_df)

'''после gridsearch лучшая модель xgboost'''
model5 = XGBClassifier(n_estimators=100, reg_alpha=1.0, reg_lambda=5, learning_rate=0.1, 
                    max_depth=11, gamma=2, min_child_weight=1, subsample=1.0, colsample_bytree=1)
# model5.fit(train_df, train_y)
# probs5 = model5.predict_proba(test_df)[:, 1]
# pred5 = model5.predict(test_df)

'''все параметры для xgboost'''
# params_grid_xgboost = {'max_depth': [3, 6, 7, 9, 11, 12],              # макс глубина дерева
#                     'learning_rate': [0.001, ],                        # темп обучения 
#                     'n_estimators': [100, 200, 300, 400, 500, 600],    # кол-во моделей
#                     'min_child_weight': [1, 3, 5, 7],                  # минимальное сумма весов в листьях(запрещает split, если меньше)
#                     'gamma': [0.1, 0.2, 0.5, 1, 2],                    # минимальное улучшение loss для split (0.1, 0.2, 0.5, 1, 2)
#                     'subsample': [0.8, 1.0],                           # доля данных для каждого дерева(0.5 - 1) - даются случайные данные
#                     'colsample_bytree': [0.6, 0.8, 1],                 # доля признаков для дерева(Случайный отбор столбцов перед построением каждого дерева)(0.5 - 1)
#                     'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],             # коэффициент для L1 регуляризации     
#                     'reg_lambda': [0.01, 0.1, 0.5, 1, 2, 5, 10],       # коэффициент для L2 регуляризации
#                     'tree_method': ['hist']}                           # алгоритм построения деревьев('auto', 'exact', 'approx')

'''после gridsearch лучшая модель lightgbm'''
# model6 = lightgbm.LGBMClassifier(n_estimators=500) 
# model6.fit(train_df, train_y)
# pred6 = model6.predict(test_df)

'''поиск лучших гиперпараметров'''
# def search_hp(X, y, model, param_grid, param_scale=None):
    
#     CV_model = GridSearchCV(estimator=model,
#                             param_grid=param_grid,
#                             cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1234),
#                             scoring='f1',
#                             n_jobs=-1,
#                             verbose=0)
#     CV_model.fit(X, y)
   
#     return CV_model.best_score_, CV_model.best_estimator_, CV_model.cv_results_, CV_model.best_params_
# param_grid = {'max_depth': [11],
#             'learning_rate': [0.01], 
#             'n_estimators': [100],
#             'reg_lambda': [5],
#             'reg_alpha': [1.0],
#             'min_child_weight': [1, 3, 5, 7],
#             'gamma': [0.1, 0.2, 0.5, 1, 2],
#             'subsample': [0.8, 1.0],
#             'colsample_bytree': [0.6, 0.8, 1],
#             'tree_method': ['hist']}
# best_score, best_xg, results, params = search_hp(train_df, train_y, 
#                                                 model5, param_grid)

# print(best_score, best_xg, params)   


'''NN'''
'''подготовка данных для подачи в NN model'''
# def data_for_nn(train_x, train_y, test_x, batch_size):
#     train_labels_np = train_y.values
#     train_samples_np = train_x.values
#     test_samples_np = test_x.values

#     input_size = train_samples_np.shape[1]

#     train_labels = torch.tensor(train_labels_np, dtype=torch.float)
#     train_samples = torch.tensor(train_samples_np, dtype=torch.float)
#     test_samples = torch.tensor(test_samples_np, dtype=torch.float)

#     dataset = TensorDataset(train_samples, train_labels)

#     train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(test_samples, batch_size=batch_size)

#     return train_loader, test_loader, input_size

# train_loader, test_loader, input_size = data_for_nn(train_df, train_y, test_df, batch_size=32)

'''тренировка модели, взята из дз и чуть переделана под бинарную классификацию'''
# def train_model(model, train_loader, loss, optimizer, num_epochs):    
#     loss_history = []
#     train_history = []

#     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

#     for epoch in range(num_epochs):
#         model.train() 
        
#         loss_accum = 0
#         correct_samples = 0
#         total_samples = 0
        
#         for i_step, (x, y) in enumerate(train_loader):
#             prediction = model(x)   
#             prediction = prediction.squeeze() 
#             loss_value = loss(prediction, y)
#             optimizer.zero_grad()
#             loss_value.backward()
#             optimizer.step()
            
#             predicted_classes = (prediction > 0.5).float()
#             correct_samples += torch.sum(predicted_classes == y)
#             total_samples += y.shape[0]
            
#             loss_accum += loss_value

#         ave_loss = loss_accum / (i_step + 1)
#         train_accuracy = float(correct_samples) / total_samples
        
#         lr_scheduler.step()     
        
#     return loss_history, train_history

'''nn model (1 -DO, 2 -BN, 256-128, adam)'''
# nn_model = nn.Sequential(
#         nn.Linear(input_size, 256),
#         nn.BatchNorm1d(256),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.25),
#         nn.Linear(256, 128),
#         nn.BatchNorm1d(128),
#         nn.ReLU(inplace=True),
#         nn.Linear(128, 1),
#         nn.Sigmoid()
#         )
# optimizer = optim.Adam(nn_model.parameters(), lr=1e-3)
# loss = nn.BCELoss().type(torch.FloatTensor)
# loss_history, train_history = train_model(nn_model, train_loader, loss, optimizer, 15)

'''предикт для отправки на LB'''
# def predict_on_test(model, loader):
#     model.eval()
#     pred = []
#     probs = []
#     with torch.no_grad():
#         for test_samples in loader:
#             prediction_by_batch = model(test_samples)
#             probs.append(prediction_by_batch)
#             predicted_classes = (prediction_by_batch > 0.5).int()
#             pred.append(predicted_classes)
#         all_pred = torch.cat(pred, dim=0)
#         all_probs = torch.cat(probs, dim=0)
#     return all_pred.numpy().flatten(), all_probs.numpy().flatten()

# pred7, probs7 = predict_on_test(nn_model, test_loader)

'''усреднение(NN + XGBClassifier + KNN + LogisticRegression + RandomForestClassifier)'''
# average_probs = (probs + probs2 + probs4 + probs5 + probs7) / 5
# pred_on_ave_probs = (average_probs > 0.5).astype(int)

'''голосование'''
# all_pred = [pred, pred2, pred4, pred5, pred7]
# all_pred = np.sum(all_pred, axis=0)
# voting_pred = [1 if preda > 2 else 0 for preda in all_pred]


'''стейкинг через LinearRegression c Ridge регуляризацией'''
stacking_model = Ridge(alpha=1.0)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = [model, model2, model3, model4, model5]

models_pred_train = np.zeros((len(train_df), 5))

for train_idx, val_idx in skf.split(train_df, train_y):
    for num, machine in enumerate(models):
        machine.fit(train_df.iloc[train_idx], train_y.iloc[train_idx])
        fold_probs = machine.predict_proba(train_df.iloc[val_idx])[:, 1]
        models_pred_train[val_idx, num] = fold_probs

stacking_model.fit(models_pred_train, train_y)

models_pred_test = np.zeros((len(test_df), 5))
for num, machine in enumerate(models):
    machine.fit(train_df, train_y)

    predict_test = machine.predict_proba(test_df)[:, 1]
    models_pred_test[:, num] = predict_test

stacking_predictions = stacking_model.predict(models_pred_test)

stacking_predictions = [1 if preda > 0.5 else 0 for preda in stacking_predictions]



submission = pd.DataFrame({
    'PassengerId': origin_test['PassengerId'], 
    'Survived': stacking_predictions
})
submission.to_csv('submission.csv', index=False)                        