import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def findDistance(x, y):
    return np.sqrt(np.sum(np.power(x - y, 2)))


def findPoints(data, k):
    m, n = np.shape(data)
    points = np.mat(np.zeros((k, n)))
    for i in range(n):
        min = np.min(data[:, i])
        I = float(np.max(data[:, i]) - min)
        points[:, i] = min + I * np.random.rand(k, 1)
    return points


def kMeans(data, k):
    m, n = np.shape(data)
    cluster = np.mat(np.zeros((m, 2)))
    points = findPoints(data, k)
    flag = True
    while flag:
        flag = False
        for i in range(m):
            minDistance = np.inf
            minIndex = -1
            for j in range(k):
                distance = findDistance(points[j, :], data[i, :])
                if distance < minDistance:
                    minDistance = distance
                    minIndex = j
            if cluster[i, 0] != minIndex:
                flag = True
            cluster[i, :] = minIndex, minDistance ** 2
        for p in range(k):
            pts = data[np.nonzero(cluster[:, 0].A == p)[0]]
            points[p, :] = np.mean(pts, axis=0)
    return points, cluster


def dichotomyKMeans(data, k):
    m, n = np.shape(data)
    cluster = np.mat(np.zeros((m, 2)))
    points = np.mean(data, axis=0).tolist()[0]
    pointsList = [points]
    for i in range(m):
        cluster[i, 1] = findDistance(points, data[i, :])**2

    while len(pointsList) < k:
        SSE = np.inf

        for j in range(len(pointsList)):
            pts = data[np.nonzero(cluster[:, 0].A == j)[0], :]
            pointsMatrix, informationOfData = kMeans(pts, 2)
            SSESplit = np.sum(informationOfData[:, 1])
            SSENoSplit = np.sum(cluster[np.nonzero(cluster[:, 0].A != j)[0], 1])

            tempLowestSEE = SSESplit + SSENoSplit
            if tempLowestSEE < SSE:
                splitPoints = j
                newPoints = pointsMatrix
                newInformationOfData = informationOfData
                SSE = tempLowestSEE

        newInformationOfData[np.nonzero(newInformationOfData[:, 0].A == 1)[0], 0] = len(pointsList)
        newInformationOfData[np.nonzero(newInformationOfData[:, 0].A == 0)[0], 0] = splitPoints
        pointsList[splitPoints] = newPoints[0, :]
        pointsList.append(newPoints[1, :])
        cluster[np.nonzero(cluster[:, 0].A == splitPoints)[0], :] = newInformationOfData

    try:
        return np.mat(pointsList), cluster
    except ValueError:
        return np.mat(np.array(list(map(lambda x: [int(x[0]), x[1]],
                                        [np.matrix.tolist(i)[0] for i in pointsList])))), cluster


if __name__ == '__main__':
    data = pd.read_csv("E:\\result.csv")
    data = pd.DataFrame({'x': data['value'], 'y': data['price']})
    data = data.to_numpy()

    k = 2
    a, b = dichotomyKMeans(data, k)
    print(a)
    print(type(a))

    fig = plt.figure(figsize=(16, 16), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_xlabel("$value$")
    ax.set_xticks(range(0, 250000, 25000))
    ax.set_ylabel("$price$")
    ax.set_yticks(range(0, 85000, 5000))
    ax.set_title('K-means')

    for i in range(k):
        pts = data[np.nonzero(b[:, 0].A == i)[0], :]
        markerStyle = ['o', '^', 'h']
        colors = ['b', 'y', 'g']
        Marker = markerStyle[i % len(markerStyle)]
        Color = colors[i % len(colors)]
        ax.scatter(np.matrix(pts[:, 0]).A[0], np.matrix(pts[:, 1]).A[0], marker=Marker, s=90, color=Color, alpha=0.2)
    ax.scatter(a[:, 0].flatten().A[0], a[:, 1].flatten().A[0], marker='*', s=900, color='r', alpha=0.9)
    plt.show()



