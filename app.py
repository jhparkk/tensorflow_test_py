
import pandas as pd
import numpy as np
import tensorflow as tf

data = pd.read_csv('./gpascore.csv')
#print(data)
#print(data['gpa']) # gpa 컬럼만 출력
#print(data['gpa'].mean()) # gpa 컬럼의 평균값 출력
#print(data['gpa'].min()) # gpa 컬럼의 최소값 출력
#print(data['gpa'].max()) # gpa 컬럼의 최대값 출력
#print(data['gpa'].std()) # gpa 컬럼의 표준편차 출력
#print(data['gpa'].var()) # gpa 컬럼의 분산 출력
#print(data['gpa'].describe()) # gpa 컬럼의 기초통계량 출력
#print(data['gpa'].count()) # gpa 컬럼의 데이터 개수 출력

#print(data.isnull().sum()) # 빈값이 있는지 확인
data = data.dropna() # 빈값이 있으면 제거
#data = data.fillna(100) # 빈값이 있으면 100으로 채움

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'), # 레이어1 (node count=64, 활성함수=tanh)
    tf.keras.layers.Dense(128, activation='tanh'), # 레이어2 (node count=128, 활성함수=tanh)
    tf.keras.layers.Dense(1, activation='sigmoid'), # 레이어3 (node count=1, 활성함수=sigmoid)   
])

# 모델 지정 및 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  

# 모델 학습  (x:학습데이터, y:실제값, 학습 횟수)
# admit    gre    gpa     rank
#-------------------------------
# 0        380    3.21    3
# 1        660    3.67    3
# 1        800    4       1
#
# x : gre, gpa, rank   [ [380,3.21,3], [660,3.67,3], [800,4,1] ]
# y : admit [ 0, 1, 1 ]

y_train = data['admit'].values
x_train = []
for i, rows in data.iterrows():
    x_train.append([rows['gre'], rows['gpa'], rows['rank']])

model.fit(np.array(x_train), np.array(y_train), epochs=1000)  # 학습 횟수 10회 - w 값을 도출

# 모델 예측
x = [[750, 3.70, 3], [400, 3.41, 1], [860, 5.67, 3]]

predict_data = model.predict(x)
print(predict_data)

exit()
