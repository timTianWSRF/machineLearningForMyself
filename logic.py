import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 定义sigmoid函数
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


# 逻辑回归计算参数的核心
# 会涉及numpy矩阵运算
# 改进随机梯度下降法
def logicRegression(data, label, num):
    dataMatrix = data.to_numpy()
    labelMat = label.to_numpy()
    m, n = dataMatrix.shape
    weights = np.ones(n)
    for i in range(num):
        dataIndex = list(range(m))
        for j in range(m):
            alpha = 4.0/(1.0+i+j)+0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = labelMat[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


# 可视化模型x1、y1是faster的距离（副作用）和收入/生活费（正作用）
# x2、y2是lower的距离（副作用）和收入/生活费（正作用）
def visualize_model(x1, y1, x2, y2):
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax = fig.add_subplot(111)
    ax.set_xlabel("$distance$")
    ax.set_xticks(range(0, 3000, 500))
    ax.set_ylabel("$money$")
    ax.set_yticks(range(0, 4000, 500))
    ax.scatter(x1, y1, color="b", alpha=0.4)
    ax.scatter(x2, y2, color="r", alpha=0.4)
    plt.legend(shadow=True)
    plt.show()


def draw(result, data, label):
    data = np.array(data)
    label = np.array(label)
    m,n = data.shape
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i in range(m):
        if int(label[i]) == 1:
            x1.append(data[i, 1])
            y1.append(data[i, 2])
        else:
            x2.append(data[i, 1])
            y2.append(data[i, 2])
    fig = plt.figure(figsize=(8, 8), dpi=80)
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, color="b", alpha=0.4)
    ax.scatter(x2, y2, color="r", alpha=0.4)
    ax.set_xlabel("$distance$")
    ax.set_xticks(range(0, 3000, 500))
    ax.set_ylabel("$money$")
    ax.set_yticks(range(0, 4000, 500))
    x = range(0, 3000, 500)
    y = (result[0]+result[1]*x)/result[2]
    ax.plot(x, y)
    plt.show()


if __name__ == '__main__':
    # 打开文件操作
    os.chdir('D:\\')
    # 读取实验集
    data = pd.read_excel('附件1.xlsx', sep=',')
    result = data['III']
    distance = data['II']
    money = data['VI']
    X = data['IV']
    Y = data['X']
    mistake = data['V']
    test1 = pd.DataFrame({'result': result, 'distance': distance, 'money': money, 'mistake': mistake})

    # 删去因为取票，而不得买错票的
    # faster是买高铁票的人，而且是买对的
    # lower是买普快的人，也是买对的
    test1 = test1[(test1.mistake == 0)]
    faster = test1[(test1.result == 1)]
    lower = test1[test1.result == 0]

    # 整理数据
    faster = pd.DataFrame({'distance': faster['distance'], 'money': faster['money']})
    lower = pd.DataFrame({'distance': lower['distance'], 'money': lower['money']})
    # 丢弃有误数据
    lower = lower.drop(index=129)

    # 可视化步骤，红单点标签值为0，蓝点为1
    # visualize_model(faster['distance'], faster['money'], lower['distance'], lower['money'])

    # 准备逻辑回归的数据集
    m, n = test1.shape
    datas = pd.DataFrame({'X0': np.array([1]*m), 'X1': test1['distance'], 'X2': test1['money']})
    labels = pd.DataFrame({'label': test1['result']})
    # 运行逻辑回归并打印结果
    result = logicRegression(datas, labels, 200)
    print(result)
    draw(result, datas, labels)
