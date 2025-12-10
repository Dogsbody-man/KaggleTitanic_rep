import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import accuracy_score
from some_function import drop_features, split_data_type

#Загружаю данные
train_x = pd.read_csv('titanic/train.csv')
test_x = pd.read_csv('titanic/test.csv')
test_original = test_x.copy()
#создаю тренировочный таргет и массив для более удобной работы
train_y = train_x['Survived']
combine = [train_x, test_x]

#разделю категоричные и числовые призныки и данные п ним . Хотя это особо и не пригодилось
numeric_indices = [0, 1, 2, 5, 6, 7, 9]
categorical_indices = [3, 4, 8, 10]
categorical_data, numeric_data = split_data_type(train_x, categorical_indices, numeric_indices)

#Начну запонять пропуски в Fare и Embarked
mean_fare = train_x['Fare'].mean()
for dataset in combine:
    dataset['Fare'] = dataset['Fare'].astype(float)
    dataset.loc[dataset['Fare'].isnull(), 'Fare']=mean_fare
    dataset['Embarked'].fillna('S', inplace=True)

was_array = ['Mr', 'Sir', 'Don', 'Jonkheer', 'Col', 'Major', 'Rev', 'Capt', 'Dr', 
            'Master', 'Miss', 'Mrs', 'Ms', 'Mme', 'Mlle', 'the Countess', 'Lady', 'Dona']
will_array = ['Mr', 'Mr', 'Mr', 'Mr', 'Other', 'Mr', 'Other', 'Mr', 'Mr', 
            'Master', 'Miss', 'Mrs', 'Miss', 'Miss', 'Miss', 'Mrs', 'Mrs', 'Mrs']
title_mapping = dict(zip(was_array, will_array))

#Заменю титлы на подходящие чтобы сократить
for dataset in combine:
    dataset['Title'] = dataset['Name'].map(lambda x: x.split(',')[1].split('.')[0])
    dataset.reset_index(drop=True, inplace=True)
    dataset['Title_clean'] = dataset['Title'].astype(str).str.strip()
    dataset['Title_new'] = dataset['Title_clean'].replace(title_mapping)

#Заполню пропуски в возрасте средним по титлу
for dataset in combine:
    dataset.loc[(dataset.Age.isnull()) & (dataset.Title_new == 'Master'), 'Age']=5
    dataset.loc[(dataset.Age.isnull()) & (dataset.Title_new == 'Miss'), 'Age']=22
    dataset.loc[(dataset.Age.isnull()) & (dataset.Title_new == 'Mr'), 'Age']=33
    dataset.loc[(dataset.Age.isnull()) & (dataset.Title_new == 'Mrs'), 'Age']=36
    
#Скомбенирую колонны SibSp и Parch. Из этого создам isalone. заодно label encoding
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] 
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 0, 'IsAlone'] = 1

    dataset['Sex'] = dataset['Sex'].map({'male' : 0, 'female' : 1})
    dataset['Embarked'] = dataset['Embarked'].map({'S' : 1, 'C' : 2, 'Q' : 3})
    dataset['Fare'] = dataset['Fare'].round(2)

    dataset['Fare_cat']=0
    dataset.loc[dataset['Fare']<=7.91,'Fare_cat']=0
    dataset.loc[(dataset['Fare']>7.91)&(dataset['Fare']<=14.454),'Fare_cat']=1
    dataset.loc[(dataset['Fare']>14.454)&(dataset['Fare']<=31),'Fare_cat']=2
    dataset.loc[(dataset['Fare']>31)&(dataset['Fare']<=513),'Fare_cat']=3

    dataset['Age_band']=0
    dataset.loc[dataset['Age']<=16,'Age_band']=0
    dataset.loc[(dataset['Age']>16)&(dataset['Age']<=32),'Age_band']=1
    dataset.loc[(dataset['Age']>32)&(dataset['Age']<=48),'Age_band']=2
    dataset.loc[(dataset['Age']>48)&(dataset['Age']<=64),'Age_band']=3
    dataset.loc[dataset['Age']>64,'Age_band']=4

#One hot encoding для имен
encoder = OneHotEncoder(sparse_output=False, drop='first') 
encoding_train = encoder.fit_transform(train_x[['Title_new']]).astype(int)
encoding_test = encoder.transform(test_x[['Title_new']]).astype(int)
train_x[encoder.get_feature_names_out()] = encoding_train
test_x[encoder.get_feature_names_out()] = encoding_test




#Дропаю колонны, которые уже не нужны и из тренировочных таргет
combine = drop_features([train_x, test_x], ['Fare', 'Title_new', 'PassengerId', 'Title_clean', 'Cabin',
                                             'Title', 'Ticket', 'Name', 'Age'])
train_x = drop_features(train_x, ['Survived'])

# вот тут сделаю нормализацию age и fare. но сначала проверю score без них
# transformer = StandardScaler()
# train_df_new1 = train_x[['Age']]
# test_df_new1 = test_x[['Age']]
# transformed_data = transformer.fit_transform(train_df_new1)
# transformed_data_y = transformer.transform(test_df_new1)
# train_x['Age'] = transformed_data[:, 0]
# test_x['Age'] = transformed_data_y[:, 0]

print(test_x.head())

second_model = LogisticRegression()
first_model = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=5, min_samples_split=11,   
                                    min_samples_leaf=4, max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1)  
# cv_stratified = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)   
# cv_score = cross_val_score(second_model, train_x, train_y, cv=cv_stratified, scoring='accuracy')
first_model.fit(train_x, train_y)
pred = first_model.predict(test_x)
# accuracy = accuracy_score(tr_y, pred)
# first_model.fit(train_x, train_y)
# prediction = first_model.predict(test_x)

submission = pd.DataFrame({
    'PassengerId': test_original['PassengerId'],  # сохрани заранее!
    'Survived': pred
})
submission.to_csv('submission.csv', index=False)
# print(accuracy)
# print(cv_score.mean())
# print(cv_score.std())

# train_errors = []
# val_errors = []

# for n in range(10, 201, 10):  # от 10 до 200 деревьев с шагом 10
#     rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    
#     # Быстрая кросс-валидация
#     from sklearn.model_selection import cross_validate
#     scores = cross_validate(
#         rf, train_x, train_y,
#         cv=cv_stratified,
#         scoring='accuracy',
#         return_train_score=True,
#         n_jobs=-1
#     )
    
#     train_errors.append(scores['train_score'].mean())
#     val_errors.append(scores['test_score'].mean())

# # Визуализация
# plt.plot(range(10, 201, 10), train_errors, 'o-', label='Train')
# plt.plot(range(10, 201, 10), val_errors, 's-', label='Validation')
# plt.xlabel('Number of Trees')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)
# plt.show()