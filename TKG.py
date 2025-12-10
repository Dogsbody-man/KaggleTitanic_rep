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
from sklearn import svm

#Загружаю данные
train_x = pd.read_csv('titanic/train.csv')
test_x = pd.read_csv('titanic/test.csv')
test_original = test_x.copy()
#создаю тренировочный таргет
train_y = train_x['Survived']
mean_fare = train_x['Fare'].mean()
test_x.loc[test_x['Fare'].isnull(), 'Fare']=mean_fare

train_x['Initial']=0
test_x['Initial']=0
for i in train_x:
    train_x['Initial'] = train_x.Name.str.extract('([A-Za-z]+)\.')
for i in test_x:
    test_x['Initial'] = test_x.Name.str.extract('([A-Za-z]+)\.')
train_x['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                    ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'], inplace=True)
test_x['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                    ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mrs'], inplace=True)
for dataset in [train_x, test_x]:            
    dataset.loc[(dataset.Age.isnull()) & (dataset.Initial == 'Master'), 'Age']=5
    dataset.loc[(dataset.Age.isnull()) & (dataset.Initial == 'Miss'), 'Age']=22
    dataset.loc[(dataset.Age.isnull()) & (dataset.Initial == 'Mr'), 'Age']=33
    dataset.loc[(dataset.Age.isnull()) & (dataset.Initial == 'Mrs'), 'Age']=36
    dataset.loc[(dataset.Age.isnull()) & (dataset.Initial =='Other'),'Age']=46
    dataset['Embarked'].fillna('S',inplace=True)
    dataset['Age_band']=0
    dataset.loc[dataset['Age']<=16,'Age_band']=0
    dataset.loc[(dataset['Age']>16)&(dataset['Age']<=32),'Age_band']=1
    dataset.loc[(dataset['Age']>32)&(dataset['Age']<=48),'Age_band']=2
    dataset.loc[(dataset['Age']>48)&(dataset['Age']<=64),'Age_band']=3
    dataset.loc[dataset['Age']>64,'Age_band']=4
    dataset['Family_Size']=0
    dataset['Family_Size']=dataset['Parch']+dataset['SibSp']
    dataset['Alone']=0
    dataset.loc[dataset.Family_Size==0,'Alone']=1
    dataset['Fare_cat']=0
    dataset.loc[dataset['Fare']<=7.91,'Fare_cat']=0
    dataset.loc[(dataset['Fare']>7.91)&(dataset['Fare']<=14.454),'Fare_cat']=1
    dataset.loc[(dataset['Fare']>14.454)&(dataset['Fare']<=31),'Fare_cat']=2
    dataset.loc[(dataset['Fare']>31)&(dataset['Fare']<=513),'Fare_cat']=3
    dataset['Sex'].replace(['male','female'],[0,1],inplace=True)
    dataset['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
    dataset['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
    dataset.drop(['Name','Age','Ticket','Fare','Cabin','PassengerId'],axis=1,inplace=True)

train_x.drop(['Survived'], axis=1,inplace=True)

# train, test= train_test_split(train_x, test_size=0.3, random_state=0, stratify=train_x['Survived'])
# train_X=train[train.columns[1:]]
# train_Y=train[train.columns[:1]]
# test_X=test[test.columns[1:]]
# test_Y=test[test.columns[:1]]


# model1 = svm.SVC(kernel='rbf',C=1,gamma=0.1)
# model1.fit(train_X, train_Y)
# prediction1 = model1.predict(test_X)
# score1 = accuracy_score(prediction1, test_Y)
# print(score1)

model2 = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=5, min_samples_split=11,   
                                    min_samples_leaf=4, max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1)
# model2.fit(train_X, train_Y)
# prediction2 = model2.predict(test_X)
# score2 = accuracy_score(prediction2, test_Y)
# print(score2)


model3 = LogisticRegression()
model3.fit(train_x, train_y)
prediction3 = model3.predict(test_x)
# score3 = accuracy_score(prediction3, test_y)
# print(score3)
submission = pd.DataFrame({
    'PassengerId': test_original['PassengerId'],  # сохрани заранее!
    'Survived': prediction3
})
submission.to_csv('submission.csv', index=False)
model4 = DecisionTreeClassifier()
# model4.fit(train_X, train_Y)
# prediction4 = model4.predict(test_X)
# score4 = accuracy_score(prediction4, test_Y)
# print(score4)

model5 = KNeighborsClassifier()
# model5.fit(train_X, train_Y)
# prediction5 = model5.predict(test_X)
# score5 = accuracy_score(prediction5, test_Y)
# print(score5)
# cv_stratified = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)   
# cv_score = cross_val_score(model3, train_x, train_y, cv=cv_stratified, scoring='accuracy')
# print(cv_score.mean())
# print(cv_score.std())
