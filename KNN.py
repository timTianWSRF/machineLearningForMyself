import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 这个函数用于对列表的每个元素进行计数
def findListNum(li):
    li = list(li)
    set1 = set(li)
    dict1 = {}
    for item in set1:
        dict1.update({item: li.count(item)})
    return dict1


def KNN(data2, data, k):
    datatemp = pd.DataFrame({'x': data['x'], 'y': data['y']}).to_numpy()
    size, temp =data.shape
    distance = (((np.tile(data2, (size, 1)) - datatemp)**2).sum(axis=1))**0.5
    data = pd.DataFrame({'x': data['x'], 'y': data['y'], 'flag': data['flag'], 'distance': distance})
    data.sort_values("distance", inplace=True)
    data = data.head(k)
    result = findListNum(data.loc[:, 'flag'])
    try:
        if result[0.0] > result[1.0]:
            c = 0.0
        else:
            c = 1.0
    except KeyError:
        for key in result.keys():
            c = key
    return c


if __name__ == '__main__':
    data = pd.read_excel("D:\\附件1.xlsx")
    data = pd.DataFrame({'distance': data['II'], 'time': data['IV'], 'moneyIn': data['VI'], 'moneyOut': data['VIII']})
    data = data.loc[data['moneyIn'] < 4000]

    fig = plt.figure(figsize=(8, 8), dpi=80)
    ax = fig.add_subplot(111)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title('KNN')
    ax.scatter(data['distance']/5, data['time']*40, color='b', alpha=0.4)
    ax.scatter(data['moneyOut'], data['moneyIn'], color='y', alpha=0.4)
    plt.show()

    l = len(data['distance'])
    A = np.array([0]*l)
    B = np.array([1]*l)
    data1 = pd.DataFrame({'x': data['distance']/5, 'y': data['time']*40, 'flag': A})
    data2 = pd.DataFrame({'x': data['moneyOut'], 'y': data['moneyIn'], 'flag': B})
    newData = np.vstack((data1, data2))
    newData = pd.DataFrame(newData)
    newData.columns = ['x', 'y', 'flag']
    print(newData)

    fig = plt.figure(figsize=(8, 8), dpi=80)
    ax = fig.add_subplot(111)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title('KNN')
    ax.scatter(newData['x'], newData['y'], color='b', alpha=0.4)
    plt.show()

    data3 = pd.read_excel("D:\\附件2.xls")
    data31 = pd.DataFrame({'x': data3['II']/5, 'y': data3['IV']*40})
    data32 = pd.DataFrame({'x': data3['VIII'], 'y': data3['VI']})
    data4 = np.vstack((data31, data32)).tolist()
    print(data4)
    for i in data4:
        print(KNN(i, newData, 5))

