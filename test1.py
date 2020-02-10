# import matplotlib.pylab as plt
# import random as ra
# import numpy as np
# import matplotlib as m
#
# m.rcParams['font.sans-serif'] = ['KaiTi']
# m.rcParams['font.serif'] = ['KaiTi']
#
# plt.figure(3)
# x_index = np.arange(10)   #柱的索引
# x_data = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J')
#
# y1_data = ()
# y2_data = ()
# for i in range(10):
#     y1_data += (ra.randint(1, 100),)
#     y2_data += (ra.randint(1, 100),)
# bar_width = 0.35   #定义一个数字代表每个独立柱的宽度
#
# #参数：左偏移、高度、柱宽、透明度、颜色、图例
# rects1 = plt.bar(x_index, y1_data, width=bar_width, alpha=0.4, color='y', label='示例一')
# rects2 = plt.bar(x_index + bar_width, y2_data, width=bar_width, alpha=0.5, color='b', label='示例二')
# #关于左偏移，不用关心每根柱的中心不中心，因为只要把刻度线设置在柱的中间就可以了
# plt.xticks(x_index + bar_width/2, x_data)   #x轴刻度线
# plt.legend()    #显示图例
# plt.tight_layout()  #自动控制图像外部边缘，此方法不能够很好的控制图像间的间隔
# plt.show()
#
# import matplotlib.pylab as plt
# import random as ra
# import numpy as np
# import matplotlib as m
#
# m.rcParams['font.sans-serif'] = ['KaiTi']
# m.rcParams['font.serif'] = ['KaiTi']
#
# plt.figure()
# x_index = np.arange(10)
# x_data = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J')
#
# y1_data = ()
# y2_data = ()
# for i in range(10):
#     y1_data += (ra.randint(1, 100),)
#     y2_data += (ra.randint(1, 100),)
# bar_width = 0.35
#
# rects1 = plt.bar(x_index, y1_data, width=bar_width, alpha=0.4, color='y', label='示例一')
# rects2 = plt.bar(x_index + bar_width, y2_data, width=bar_width, alpha=0.5, color='b', label='示例二')
# plt.xticks(x_index, x_data)
# plt.legend()
# plt.tight_layout()
# plt.show()





#plt.scatter()
# plt.scatter(x, y, s=20, c=None, marker='o', cmap=None, norm=None, vmin=None, vmax=None,
# alpha=None, linewidths=None, verts=None, edgecolors=None, hold=None, data=None, **kwargs)

# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.random.randn(1000)
# y = np.random.randn(1000)
# plt.scatter(x, y, marker='h', s=np.random.randn(1000)*100, c=y, edgecolor='black')
# plt.grid(True, linestyle='--')
# plt.show()
# # s：散点的大小
# # c：散点的颜色
# # vmin,vmax：亮度设置，标量
# # cmap：colormap

# import matplotlib.pyplot as plt
# import matplotlib
# import random as ra
#
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False
#
# data = [ra.randint(0, 100) for i in range(1000)]
#
# a = [i for i in data if i < 30]
# b = [i for i in data if i >= 30 and i<=60]
# c = [i for i in data if i > 60]
#
# aSize = len(a)/1000
# bSize = len(b)/1000
# cSize = len(c)/1000
#
# label_list = ["小于30", "介于30与60", "大于60"]    # 各部分标签
# size = [aSize, bSize, cSize]    # 各部分大小
# color = ["pink", "yellow", "blue"]     # 各部分颜色
# explode = [0, 0.05, 0]   # 各部分突出值
# """
# 绘制饼图
# explode：设置各部分突出
# label:设置各部分标签
# labeldistance:设置标签文本距圆心位置，1.1表示1.1倍半径
# autopct：设置圆里面文本
# shadow：设置是否有阴影
# startangle：起始角度，默认从0开始逆时针转
# pctdistance：设置圆内文本距圆心距离
# 返回值
# l_text：圆内部文本，matplotlib.text.Text object
# p_text：圆外部文本
# """
# patches, l_text, p_text = plt.pie(size, explode=explode, colors=color, labels=label_list, labeldistance=1.1, autopct="%1.1f%%", shadow=False, startangle=90, pctdistance=0.6)
# plt.axis("equal")    # 设置横轴和纵轴大小相等，这样饼才是圆的
# plt.legend()
# plt.show()

# import random as ra
# import matplotlib.pyplot as plt
#
# x = ra.sample(range(0, 10), 10)
# y = ra.sample(range(10, 20), 10)
#
# x.sort()
# plt.plot(x, y, 'ro-')
# plt.show()

# from pandas import Series
# import pandas as pd
#
# ser1 = pd.Series([1, 2, 3, 4, 5])
# ser1.index = ['A', 'B', 'C', 'D', 'E']
# print(ser1)
#
# ser2 = Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
# print(ser2)

# import numpy as np
#
# print(None == None)
# print(np.nan == np.nan)
# print(type(np.nan))

# import pandas as pd
#
# data = pd.read_csv("D:\\result.csv")
# print(type(data))
# print(data)

# import pandas as pd
#
# idInfo = pd.read_csv("D:\\result.csv", encoding='gbk')
# print(idInfo.head(3))
# print(idInfo.columns)
# print(idInfo.shape)

# print(idInfo.loc[0])
# print(idInfo.loc[2:4])

# print(idInfo[["value", "model"

# c = idInfo["value"]/100
# idInfo["value"] = c
# print(idInfo["value"])


# print(idInfo["price"].max())
# print(idInfo["price"].min())
# print(idInfo["price"].mean())

# print(idInfo.sort_values("value", inplace=False, ascending=True))
# print(idInfo.sort_values("value", inplace=False, ascending=False))

# import csv
#
# csvFile = open("D:\\1.csv", 'w')
# write = csv.writer(csvFile)
# write.writerow([1.0, 1.0, 'exampleOne', 'exampleTwo'])
# rows = [[2.0, 2.0, 'exampleThree', 'exampleFour'], [3.0, 3.0, 'exampleFive', 'exampleSix']]
# write.writerows(rows)
# csvFile = open("D:\\result.csv", 'r')
# read = csv.reader(csvFile)
# for i in read:
#     print(i)

# import pandas as pd
# file = pd.read_excel('D:\\附件1.xlsx', skipfooter=170)
# print(file.head(20))

import re

# str = '''The Gift of Life
# On the very first day, God created the cow. He said to the cow, "Today I have created you! As a cow, you must go to the field with the farmer all day long. You will work all day under the sun! I will give you a life span of 50 years."
# The cow objected, "What? This kind of a tough life you want me to live for 50 years? Let me have 20 years, and the 30 years I'll give back to you." So God agreed.
# On the second day, God created the dog. God said to the dog, "What you are supposed to do is to sit all day by the door of your house. Any people that come in, you will have to bark at them! I'll give you a life span of 20 years."
# The dog objected, "What? All day long to sit by the door? No way! I'll give you back my other 10 years of life!" So God agreed.
# On the third day, God created the monkey. He said to the monkey, "Monkeys have to entertain people. You've got to make them laugh and do monkey tricks. I'll give you 20 years life span."
# The monkey objected. "What? Make them laugh? Do monkey faces and tricks? Ten years will do, and the other 10 years I'll give you back." So God agreed.
# On the fourth day, God created man and said to him, "Your job is to sleep, eat, and play. You will enjoy very much in your life. All you need to do is to enjoy and do nothing. This kind of life, I'll give you a 20 year life span."
# The man objected. "What? Such a good life! Eat, play, sleep, do nothing? Enjoy the best and you expect me to live only for 20 years? No way, man! Why don't we make a deal? Since the cow gave you back 30 years, and the dog gave you back 10 years and the monkey gave you back 10 years, I will take them from you! That makes my life span 70 years, right?" So God agreed.
# AND THAT'S WHY...
# In our first 20 years, we eat, sleep, play, enjoy the best and do nothing much. For the next 30 years, we work all day long, suffer and get to support the family. For the next 10 years, we entertain our grandchildren by making monkey faces and monkey tricks. And for the last 10 years, we stay at home, sit by the front door and bark at people!
# '''

# result2 = re.match('(T)([a-zA-Z]' ')*', str).group()
# print(result2)

# for i in range(1, 4):
#     print(re.search('([a-zA-Z])*(ea)([a-zA-Z])*', str).group(i))

# print(type(result))
# print(result)
#
# print(type(result1))
# print(result1)
# for i in result1:
#     print(i)
#
# print(type(result3))
# print(result3)
# for i in result2:
#     print(i.group())

# str3 = 'from|form|for'
# aimStr = 'format'
#
# result3 = re.search(str3, aimStr).group()
# print(result3)

# import numpy as np
#
# # 生成两个矩阵并打印
# print("生成两个矩阵")
# a = np.arange(1, 11).reshape(2, 5)
# b = np.arange(12, 27).reshape(5, 3)
# print(a)
# print()
# print(b)
# print()
#
# print("矩阵相乘示例一")
# print(a.dot(b))
# print()
# print("矩阵相乘示例二")
# print(np.dot(a, b))
# print()

# # 生成一个个二维数组
# print("打印初始二维数组")
# a = np.arange(1, 33).reshape(4, 8)

# print(a)
# print()
#
# # 以3行、4行为界划分，划出3个数组
# print(np.hsplit(a, (3, 4)))
# print()
# # 以3行、5行、7行为界划分，划出4个数组
# print(np.hsplit(a, (3, 5, 7)))
# print()

# # 返回转置矩阵
# print("返回转置矩阵")
# print(c.T)
# print()
#
# # 返回所有元素
# print("返回所有元素")
# print(c.ravel())
# print()
#
# # 再以一个三维矩阵为例返回所有元素
# # 生成一个三维数组
# print("打印初始三维数组")
# c1 = np.arange(1, 19).reshape(2, 3, 3)
# print(c1)
# print()
# # 返回所有元素
# print("返回所有元素")
# print(c1.ravel())
# print()


# # 迭代一个维度
# print("迭代一个维度")
# for firstAxis in c:
#     print(firstAxis)
# print()
#
# # 迭代两个维度
# print("迭代两个维度")
# for firstAxis in c:
#     for secondAxis in firstAxis:
#         print(secondAxis)
#
# # 迭代三个维度
# print("迭代三个维度，示例1")
# for firstAxis in c:
#     for secondAxis in firstAxis:
#         for thirdAxis in secondAxis:
#             print(thirdAxis)
#
# print("迭代三个维度，示例2")
# for element in c.flat:
#     print(element)


# print("示例一")
# print(c[1, :, :])
# print(c[1, ...])
# print()
# print("示例二")
# print(c[..., 2])
# print(c[:, :, 2])



# def func(x, y):
#     return 5*x+y
#
# # 打印矩阵
# b = np.fromfunction(func, (5, 4), dtype=int)
# print(b)
#
# # 多维数组索引
# print('多维数组索引')
# print(b[3, 3])
#
# # 多维数组切片
# print('多维数组切片')
# print(b[:, 0])
# print(b[0, :])
# print(b[3, 3])
# print(b[-1])
# print(b[-1, :])


# #打印数组
# a = np.arange(100)
# print(a)
#
# # 索引操作
# print('索引操作')
# # 打印序号为3的元素
# print(a[3])
# # 打印序号前30的元素
# # 注意使用的是序号而不是迭代器
# for i in range(30):
#     print(a[i], end=' ')
#     print(' ')
#
# # 切片操作
# print('切片操作')
# # 正序号10到20切片
# print(a[10:20])
# # 间隔5，切片用逆序号
# print(a[:-1: 5])
# # 可以用切片修改再打印示例
# a[:3] = -1
# print(a[:5])
# print()
#
# # 迭代操作
# print('迭代操作')
# # 这里没有使用序号而是迭代器
# for i in a:
#     print(i)


# print(np.empty((4, 5)))
# print(np.zeros((4, 5)))
# print(np.ones((4, 5)))
# print(np.pi)
# print(np.exp(1))

# a = np.arange(0, 53, 3).reshape(3, -1)
# print(a)
# b = np.linspace(0, 53, 3).reshape(3, -1)
# print(b)
# c = np.linspace(0, 53, 18).reshape(3, -1)
# print(c)
# #


# print(np.hsplit(b, 2))
# print(np.hsplit(b, (1, 2, 3, 4)))



# a = np.arange(1, 26).reshape(5, 5)
# A = np.matrix(a)
# eigenvalue, featurevector=np.linalg.eig(A)
# print(eigenvalue)
# print(featurevector)

# import numpy as np
#
# # 初始化一个非奇异矩阵(数组)
# # 因为奇异矩阵不可逆
# print("求逆矩阵")
# a = np.array([[1, 2], [3, 4]])
# print(np.linalg.inv(a))  # 相当于MATLAB中 inv() 函数
#
# # 使用.I方法求逆更方便
# A = np.matrix(a)
# print(A.I)
# print()
#
# # 求矩阵特征值和特征向量
# print('求矩阵特征值和特征向量')
# eigenvalue, featurevector = np.linalg.eig(A)
# print('矩阵特征值：', end='')
# print(eigenvalue)
# print('矩阵特征向量', end='')
# print(featurevector)
#
# # Python Hello World代码示例
# # 错误示例，缺少引号
# print(Hello World!!!)

# '''
# 这是一个多行注释
# 这个程序用来展示一段问候语
# 输入自己的名字机器便向你打招呼！
# '''
# name = input("What's your name?")
# print("Hello, "+name+". Nice to meet you. I'm Python.")

# a = 100
# b = False
# c = True
#
# print(a+c)# 100+1
# print(a+b)# 100+0
# print(b+c)# 0+1
# print(a+b+c)# 100+0+1
#
# del a
# del b, c

# h1 = 6.626070156E-34
# h2 = 6.626070156e-34
# print(h1 is h2)

# a = 1 + 2j
# print(a.conjugate())

# while(True):
#     a = int(input("输入一个指定范围的数："))
#     if(a >= 1 and a<=10 or a >= 15):
#         print(True)
#     else:
#         print(False)


# a = int(input("输入被除数整数："))
# b = int(input("输入除数整数："))
# print("下溢式除法“//”结果", end='')
# print(a//b)
# print("数学除法“/”结果", end='')
# print(a/b)

# a = int(input("输入幂："))
# b = int(input("输入指数："))
# print("幂指函数{}^{}=".format(a, b), end='')
# print(a**b)



# a = 100
# # b = 100.00
# # c = '100.00'
# # d = 100 + 0j
# #
# # class e:
# #     pass
# # ee = e
# #
# # print(type(a))
# # print(type(b))
# # print(type(c))
# # print(type(d))
# # print(type(ee))

# a = 100.00
# print(isinstance(a, int))
# print(isinstance(a, float))


# a = 100
# b = 100.00
# c = 100 + 0j
#
# class e:
#     pass
# d = e
#
# example = [a, b, c, d]
# lis = [int, float, type, complex]
# for i in example:
#     for j in lis:
#         if(isinstance(i, j) is True):
#             print("{} is {}.".format(i, j))

# # 用给定元素初始化列表
# lis = [1, 2, 3, 4, 5]
# # 用for循环和迭代器生成
# lis2 = []
# for i in range(1, 6):
#     lis2.append(i)
#
# print(lis)
# print(lis2)
# # is 实际比较的是两个列表，
# # 存放的地址而不是里面包含的值。
# # 比较包含的值用‘==’即可。
# print(lis is lis2)
# print(lis == lis2)

# lis = [1, 2, 3, 4, 5]
# example = [3, 6]
# for i in example:
#     if i in lis:
#         print('{}存在列表中。'.format(i))
#     elif i not in lis:
#         print('{}不在列表中。'.format(i))


# lis = [1, 2, 3, 4, 5]
# lis2 = lis*3
# print(lis2)

# 初始化列表
# lis = ['*']*10
#
# # 含中括号
# for i in range(1, len(lis)):
#     print(lis[:i])
#
# # 不含中括号
# for i in range(1, len(lis)):
#     for j in lis[:i]:
#         print(j, end='')
#     print()

# a = (1, 2, 3, 4, 5)
# print('-各组都是一样的--')
# # 它们是一样的
# print(a[1::1])
# print(a[1:5:])
# print(a[1::])
#
# # 分割用
# print("-"*15)
#
# # 它们是一样的
# print(a[0:5:2])
# print(a[0::2])
# print(a[:5:2])
# print(a[::2])
#
# # 分割用
# print("-"*15)
#
# # 它们是一样的
# print(a[-5:-1:])
# print(a[0:4:])
# print(a[0:4])
# print(a[:4])

# a = (1, 2, 3, 4)
# b = (5,)
# c = (7, 8, 9)
#
# alist = [1, 2, 3, 4]
# bnum = 5
# blist = [5]
# clist = [7, 8, 9]
#
# print(alist)
#
# print('-'*15)
# print(alist+blist)
# alist.append(bnum)
# print(alist)
#
# print('-'*15)
# print(alist+clist)
# print(alist.extend(clist))
#
# print('-'*15)
# print(a+b)
# print(a+c)
# print(a+b+c)

# # 给出键值对创建一个字典
# dic1 = {'a': 1, 'b': 2, 'c': 3}
# print(dic1)
# print('-'*15)# 分割用
#
# # 甚至可以直接用大括号来创建一个空的字典
# dic2 = {}
# print(dic2)
# print('-'*15)# 分割用
#
# # 用内建方法dict()来创建
# dic3 = dict((['a', 1], ['b', 2], ['c', 3]))
# print(dic3)
# print('-'*15)# 分割用
#
# # 用内建方法fromkeys()来创建
# dic4 = {}.fromkeys(('a', 'b', 'c'))
# print(dic4)
# print('-'*15)# 分割用

# dic = {'a': 1, 'b': 2, 'c': 3}
#
# # 通过遍历键来得到值
# for i in dic.keys():
#     print('键为{}，其值为{}。'.format(i, dic[i]))
# print('-'*15)# 分割用
#
# # Python3可通过遍历字典本身来得到值
# for i in dic:
#     print('键为{}，其值为{}。'.format(i, dic[i]))
# print('-'*15)# 分割用
#
# # 直接通过键来得到值
# print(dic['a'])
# print(dic['b'])
# print(dic['c'])
# print('-'*15)# 分割用

# dic = {'a': 1, 'b': 2, 'c': 3}
#
# # a是字典dic的一个键
# print('a' in dic)
# print('a' not in dic)
# print('-'*15)# 分割用
#
# # d不是字典dic的一个键
# print('d' in dic)
# print('d' not in dic)

# dic1 = {'a': 11, 'b': 22, 'c': 33, 'd': 44, 'e': 55}
# # print(len(dic1))
# # print('-'*15)# 分割用
# #
# # str1 = str(dic1)
# # print(str1)
# # print(type(str1))
# # print('-'*15)# 分割用
# #
# # print(type(dic1))


# # 字符串创建示例
# str1 = 'abcdefg'
# str2 = "hijklmn"
# print(type(str1) == type(str2))
# print('-'*15)# 分割用
#
# # 字符串删除操作
# del str2
# try:
#     print(str2)
# except NameError:
#     print('目标对象已被删除')
# print('-' * 15)  # 分割用
#
# # 字符串更新操作
# str1 += "hijklmn"
# print(str1)
# print('-' * 15)  # 分割用
#
# print(str1[12])
# print(str1[:12]+'M'+str1[13:])
# print('-' * 15)  # 分割用
#
# print(str1[:5])
# print('ABCDE'+str1[5:])
# print('-'*15)# 分割用

#
# import string
#
# # 注：python2中的string成员letters在python3中改为了ascii_letters
#
# alphas = string.ascii_letters + '_'
#
# nums = string.digits
#
# print('合法标识符检查。。。')
#
# print('测试字符串长度至少为2')
#
# while True:
#
#     myInput = input('键入字符串')
#
#     if len(myInput) > 1:
#
#         if myInput[0] not in alphas:
#
#             print('##标识符首位非法##')
#
#         elif True:
#
#             allChar = alphas + nums
#
#             for otherChar in myInput[1:]:
#
#                 if otherChar not in allChar:
#
#                     print('##剩余字符串中存在非法标识符##')
#
#                     break
#
#                 else:
#
#                     break
#
#         print('检查结束')
#


# # 这是正确的驼峰式写法
# myNameIs = 1
# dataFrame = 2
# rabinMiller = 3
#
# # 这些都不是规范的驼峰式写法
# MyNameIs = 1
# my_Name_Is = 2
# _rabinMiller = 3

# import os
#
# os.chdir('f:\\')
#
# try:
#     a = open('1.py')
# except IOError:
#     print('没有文件。')
#
# try:
#     a = open('1.py')
# finally:
#     print('有没有都一样。')

# lis = [False, True]
# for a in lis:
#     if a == True:
#         print('True判断，a为{}。'.format(a))
#     else:
#         print('False判断，a为{}。'.format(a))

# import random as ra
#
# RealValue = ra.randint(0, 100)
# while True:
#     guessValue = int(input("输入一个值猜测这个随机数："))
#     if guessValue > RealValue:
#         print("你猜大了！")
#     elif guessValue < RealValue:
#         print("你猜小了！")
#     else:
#         print("你猜对了！！！")
#         break

# a = 0
# while a < 10:
#     print('当前a的值为{}'.format(a))
#     a += 1

# lis = ['list', 'TIM', 'Tom', 'Riddle']
# tup = ('tuple', 'TIM', 'Tom', 'Riddle')
# for i in lis:
#     print(i)
# print('-'*15)# 分割线
# for i in tup:
#     print(i)

# # 一般函数的定义
# def tellMeYourName():
#     return 'Tim'
#
# print(tellMeYourName())
#
# #匿名函数lambda的定义
# tellMeYourNameV2 = lambda: 'TIM'
# print(tellMeYourNameV2())

#
# from random import randint as ri
# print(n for n in [ri(1, 99)for i in range(9)]if n % 2)

# from random import randint
#
# def odd(n):
#     return n % 2
#
# allNums = []
#
# for eachNum in range(9):
#     allNums.append(randint(1, 99))
#
# print(filter(odd, allNums))


# print('the total is:', reduce((lambda x,y:x+y),range(5)))


# from bs4 import BeautifulSoup
# import os
# import requests
# import sys
#
#
# class Logger(object):
#     def __init__(self, fileN="Default.log"):
#         self.terminal = sys.stdout
#         self.log = open(fileN, "a", encoding="utf-8")
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         pass
#
# sys.stdout = Logger("E:\\example1.txt")
# for i in range(0, 30):
#     print(i)

# def main():
#     os.chdir("E:\\")
#     result = requests.get("https://www.kbb.com/cars-for-sale/cars/used-cars/?distance=none")
#     r = result.text
#     soup = BeautifulSoup(r, 'lxml')
#     pretty = soup.prettify()
#     sys.stdout = Logger("E:\\pretty.txt")
#     print(pretty)
#
#
# if __name__ == '__main__':
#     main()

# print('Hello Word!')
# print 'Hello Word!'

# import pandas as pd
# import numpy as np

# a = pd.DataFrame([[0, 1, 2, 3, 4, np.nan, 6, 7, 8, 9],
# #               [10, 11, 12, 13, 14, np.nan, 16, 17, 18, 19],
# #               [20, 21, 22, 23, 24, np.nan, 26, 27, 28, 29]]
# #               )
# #
# # for col in a.columns:
# #     print(a[col])


# import pandas as pd
#
# a = pd.DataFrame({'A': 1,
#                   'B': 2.0,
#                   'C': 3+3j,
#                   'D': 'StringExample',
#                   'E': pd.Timestamp('20200201')}, index=[1])
# print(a.dtypes)

# import pandas as pd
# import numpy as np
#
# dateTime = pd.date_range('20200101', periods=7)
# a = pd.DataFrame({'A': 1,
#                   'B': 2.0,
#                   'C': 3+3j,
#                   'D': 'StringExample',
#                   'E': np.array([True]*7)}, dateTime)
# print(a)
# print('-'*60)# 分割线
#
# print(a.head(5))
# print('-'*60)# 分割线
#
# print(a.tail(5))
# print('-'*60)# 分割线
#
# print(a.index)
# print('-'*60)# 分割线
#
# print(a.columns)
# print('-'*60)# 分割线

# import numpy as np
# import pandas as pd
#
# dateTime = pd.date_range('20200101', periods=7)
# a = pd.DataFrame(np.random.randn(7, 5), index=dateTime, columns=list('ABCDE'))
#
# print(a)
# # 全部大于0的数，产生含有NaN的b
# b = a[a > 0]
# print(b)
#
# # 增（填充法）
# # 示例全部填为1
# print(b.fillna(value=1))
# print('-'*50)
#
# # 删
# print(b.dropna(how='any'))
# print('-'*50)
#
# # 改
# # 先用布尔值标记空缺再改
# print(pd.isna(b))
# print('-'*50)
# # 修改第一行第一列
# b.iloc[0, 0] = 999
# print(b)

# # 以行标签排序，以第一维度为例
# print(a.sort_index(axis=1, ascending=False))
# print('-'*30)# 分割线
#
# # 按列标签排序，以A为例
# print(a.sort_values(by='A'))

# import psutil
#
# # 输出每个CPU的汇总
# print(psutil.cpu_times(percpu=True))

# 也可以分开来单独输出
# print(psutil.cpu_times().user)
# print(psutil.cpu_times().system)
# print(psutil.cpu_times().idle)
# print(psutil.cpu_times().interrupt)
# print(psutil.cpu_times().idle)

# import psutil
#
# # 该句返回值时从import语句开始导入到结束
# print(psutil.cpu_percent(interval=1))
# # 下一句，计算这两个语句间，这一段时间的CPU利用率
# print(psutil.cpu_percent(interval=0))

# import psutil
#
# print(psutil.cpu_freq())
#
# import matplotlib.pyplot as plt
#
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# fig = plt.figure(figsize=(8, 8), dpi=100)
# ax = fig.add_subplot(111)
# ax.set_xlabel('预测错误的代价')
# ax.set_ylabel('预测正确的收益')
# ax.set_title('ROC')
# pos1 = [0, 1]
# pos2 = [0, 1]
# ax.plot(pos1, pos2, '--', color='r')
# plt.show()

# from sklearn.cluster import KMeans
# import pandas as pd
#
#
# def train_K_means(data):
#     model = KMeans(n_clusters=2, max_iter=200, n_init=10, algorithm="full")
#     model.fit(data)
#     return model
#
#
# if __name__ == '__main__':
#     data = pd.read_csv("E:\\result.csv")
#     data = pd.DataFrame({'x': data['value'], 'y': data['price']})
#     result = train_K_means(data)
#     print(result)
#     print(type(result))


# 这个函数用于对列表的每个元素进行计数
# def findListNum(li):
#     li = list(li)
#     set1 = set(li)
#     dict1 = {}
#     for item in set1:
#         dict1.update({item: li.count(item)})
#     return dict1
#
#
# dic = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
# print(findListNum(dic))

dic = [[1, 1, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0], [1, 1, 9, 1]]
c = [i[-1] for i in dic]
print(c[0])