# 라이브러리 및 데이터 불러오기

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

wine = load_wine()

# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다.
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.

''' 코드 작성 바랍니다 '''

df = pd.DataFrame(data=wine.data, columns= wine.feature_names)
df['target'] = wine.target

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

####### A 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=42)

# 하이퍼파라미터 후보군 설정
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 5],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 2, 4]
}
# GridSearchCV 설정 및 학습 실행
grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 4. 결과 출력
# 최적 하이퍼파라미터와 최고 점수(Accuracy) 출력
print(f"Best Hyper-parameter: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")

# 5. 변수 중요도 시각화
# 최적 모델과 변수 중요도 추출
best_model = grid_search.best_estimator_
importances = best_model.feature_importances_
feature_names = X.columns

# 시각화 설정
plt.figure(figsize=(12, 6))
plt.bar(feature_names, importances)
plt.title('Feature Importance')  # 플롯 제목
plt.xlabel('Feature')            # x축 제목
plt.ylabel('Importances')        # y축 제목

# 변수명을 45도 기울여서 표시
plt.xticks(rotation=45, ha='right')

# 레이아웃을 조정하여 변수명이 잘리지 않게 함
plt.tight_layout()

# 플롯 보여주기
plt.show()

####### B 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''

