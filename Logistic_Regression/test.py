import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ham sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Load data tu file csv
data = pd.read_csv('dataset.csv').values
N, d = data.shape
x = data[:, 0:d-1].reshape(-1, d-1)
y = data[:, 2].reshape(-1, 1)

# Vẽ data scatter
plt.scatter(x[:10, 0], x[:10, 1], c = 'red', edgecolors='none', s=30, label = 'cho vay')
plt.scatter(x[10:, 0], x[10:, 1], c = 'blue', edgecolors='none', s=30, label = 'tu choi')
plt.legend()
plt.xlabel('Lương (triệu)')
plt.ylabel('Kinh nghiệm (năm)')
plt.title('Ảnh minh họa')

# Thêm 1 cột toàn số 1 vào bên trái ma trận x, khai báo các giá trị bàn đầu của ma trân w
x = np.hstack((np.ones((N,1)), x))
w = np.array([0., 0.1, 0.1]).reshape(-1,1)

# Số lần lặp bước 2
numOfIteration = 1000
cost = np.zeros((numOfIteration,1))
learning_rate = 0.05

for i in range(1,numOfIteration):
    y_predict = sigmoid(np.dot(x,w))
    cost[i] = -np.sum(np.multiply(y, np.log(y_predict)) + np.multiply(1-y, np.log(1-y_predict)))
    w = w - learning_rate * np.dot(x.T, y_predict - y)
    # print(cost[i])
    
# Vẽ đường phân cách
t = 0.5
plt.plot((4, 10),(-(w[0]+4*w[1]+ np.log(1/t-1))/w[2], -(w[0] + 10*w[1]+ np.log(1/t-1))/w[2]), 'g')

luong = 2
time = 2
plt.scatter(luong, time, c='black', s=35)

val_test = 1/(1+np.exp(-(w[0] + w[1]*luong + w[2]*time)))
if val_test > 0.5:
    print('cho vay')
else:
    print('tu choi')

plt.show()
