import pandas as pd
import numpy as np


# 这个函数用于对列表的每个元素进行计数
def findListNum(li):
    li = list(li)
    set1 = set(li)
    dict1 = {}
    for item in set1:
        dict1.update({item: li.count(item)})
    return dict1


def calculateEntropy(data):
    rows, columns = data.shape
    result = {}
    entropy = 0
    for c in data.columns:
        temp = findListNum(c)
        result.update(temp)
    for key in result:
        entropy -= (float(result[key])/rows)*np.math.log(float(result[key]), 2)
    return entropy


def dropAndSplit(data, columnName, value):
    temp = data[data[columnName] == value]
    temp = temp.drop(columns=columnName)
    return temp


def makeAChoice(data):
    entropyAtFirst = calculateEntropy(data)
    gainOfEntropy = 0
    column = 0
    for col in data.columns:
        print(col)
        AllOfEntropy = 0
        values = set(data[col])
        for value in values:
            tryToSplit = dropAndSplit(data, col, value)
            AllOfEntropy += tryToSplit.shape[0]/data.shape[0] * calculateEntropy(tryToSplit)
        tempGainEntropy = entropyAtFirst - AllOfEntropy
        # print(tempGainEntropy)
        # print(tempGainEntropy)
        if tempGainEntropy > gainOfEntropy:
            gainOfEntropy = tempGainEntropy
            column = col
    return column


def createTree(data, result):
    labelResult = result.T.to_numpy().tolist()
    print(labelResult)

    if data.shape[1] == 1:
        # print(data.columns[0])
        nums = findListNum(data[data.columns[0]])
        # print(nums)
        if nums[0] > nums[1]:
            c = 0
        else:
            c = 1
        return c


    labelOfWhatChose = makeAChoice(data)
    discisionTree = {labelOfWhatChose: {}}
    values = set(data[labelOfWhatChose].tolist())
    print(values)
    for value in values:
        print(value)
        discisionTree[labelOfWhatChose][value] = createTree(data, result)
    return discisionTree


if __name__ == '__main__':
    data = pd.read_excel("D:\\附件1.xlsx")
    testData = pd.DataFrame({'submit': data['VII'], 'comfortable': data['IX'], 'mistake': data['V']})
    result = pd.DataFrame({'result': data['III']})
    tree = createTree(testData, result)
    print(tree)