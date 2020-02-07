# 以下导入三方库
# pandas库相关，用于读取csv文件
# CSV文件介绍：https://baike.baidu.com/item/CSV/10739?fr=aladdin
import pandas as pd
# statsmodels库相关
# 用于定义线性回归中一个被称为“切比雪夫准则（也称“最小一乘法”）”的损失函数
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

# matpoltlib库相关
# 用于绘制图表
from pylab import *
import matplotlib.pyplot as plt

# sklearn库相关
# 用于定义线性回归中一个被称为“最小二乘准则（也称“最小二乘法”）”的损失函数
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model


# OLS回归模型具体实现
# 定义损失函数“最小二乘准则（也称“最小二乘法”）”
def trainOLS(x, y):
    # 调用sklearn中的linear_model中的现成的最小二乘法模型，名为LinearRegression
    model = linear_model.LinearRegression()
    # 进行模型拟合操作
    model.fit(x, y)
    # 打印模型计算出的参数
    # intercept_是参数b（第一张图片中标题为OLS的图表里那个斜线的的截距值）
    print(model.intercept_)
    # coef_是参数a（第一张图片中标题为OLS的图表里那个斜线的的斜率值）
    print(model.coef_)
    # 返回预测值，存放在变量re
    re = model.predict(x)
    # 将re作为本函数trainOLS的返回值
    return re


# LAD回归模型具体实现
# 定义损失函数“切比雪夫准则（也称“最小一乘法”）”
def trainLAD(x, y):
    # 加入全1列作为扰动项系数
    X = sm.add_constant(x)
    # 调用statsmodels的现成的“分位数回归模型”
    model = QuantReg(y, X)
    # 用分位数回归做替代，进行拟合。可行的原因如下两行
    # 分位数回归参数q为0.5时，等同于“切比雪夫准则（也称“最小一乘法”）”
    # 因为切比雪夫准则是分位数回归的一种特殊情况
    model = model.fit(q=0.5)
    # 返回预测值，存放在变量re
    re = model.predict(X)
    # 将re作为本函数trainOLS的返回值
    return re


#定义模型可视化函数
def visualizeModel(x, y, ols, lad):
    # 定义显示框的长宽figsize和分辨率dpi
    fig = plt.figure(figsize=(12, 6), dpi=80)
    # 定义了两个图标ax2和ax3以及其放置的位置
    # 121 一行二列第一个位置
    # 122 一行二列第二个位置
    ax2 = fig.add_subplot(121)
    ax3 = fig.add_subplot(122)

    # 定义图ax2的x轴和y轴显示名称以及显示范围(0到250000，每个刻度间隔25000），还有图表的大标题为OLS
    ax2.set_xlabel("$x$")
    ax2.set_xticks(range(0, 250000, 25000))
    ax2.set_ylabel("$y$")
    ax2.set_title('OLS')

    # 定义图ax2的x轴和y轴显示名称以及显示范围(0到250000，每个刻度间隔25000），还有图表的大标题为LAD
    ax3.set_xlabel("$x$")
    ax3.set_xticks(range(0, 250000, 25000))
    ax3.set_ylabel("$y$")
    ax3.set_title('LAD')

    # 定义点的颜色是（b）lue蓝色，阿尔法值为0.4，图示显示为实验数据
    ax2.scatter(x, y, color='b', marker='d', alpha=0.4, label='实验数据')
    # 定义线的图示显示是预测数据
    ax2.plot(x, ols, color='r', linestyle='-.', label='预测数据')

    # 图表ax3样式同ax2
    ax3.scatter(x, y, color='y', marker='D', alpha=0.4, label='实验数据')
    ax3.plot(x, lad, color='y', linestyle=':', label='预测数据')

    # 显示阴影
    plt.legend(shadow=True)
    # 将结果显示到屏幕
    plt.show()


def OLSvsLADv1(data):
    # 将value重命名为features，将price重命名为label
    features = ["value"]
    data["label"] = data["price"]
    # 返回的ols是OLS回归的预测值
    ols = trainOLS(data[features], data["label"])
    # 打印出OLS回归模型评估指标拟合优度R方的值
    print('OLS模型拟合优度: %.2f' % r2_score(data["label"], ols))
    # 返回的lad是ALD回归的预测值
    lad = trainLAD(data[features], data["label"])
    # 打印出LAD回归模型评估指标拟合优度R方的值
    print('LAD模型拟合优度: %.2f' % r2_score(data["label"], lad))
    # 调用可视化函数将图标送至屏幕
    visualizeModel(data[features], data["label"], ols, lad)



if __name__ == "__main__":
    # 设置中文，避免matplotlib显示乱码
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(12, 6), dpi=80)
    # 切换默认路径
    # 打开htmlspider.py文件输出结果：result.csv文件
    data = pd.read_csv("E:\\result.csv", encoding='gbk')
    # 去重操作，消去重复值
    data = data.dropna(axis=0, how='any')
    data = data.loc[data['price'] < 7000]


    # 这一步是是核心
    # 将数据传出OLSvsLADv1函数中，以分别调用trainOLS和trainLAD模型预测结果
    # 接着对两个模型求R方值（拟合优度值），作为模型评估的依据之一
    # 最后调用可视化函数，以图表形式显示。
    # R2称为“决定系数（拟合优度）”，数值介于0到1.
    # 越趋向0说明模型结果越不好，越趋向1则说明模型结果越好
    OLSvsLADv1(data)

    # 以下到结束是“残差评估”具体实现
    # 传入值
    X = data[['value']].values
    y = data[['price']].values

    fig1 = plt.figure(figsize=(8, 8), dpi=100)
    ax1 = fig1.add_subplot(111)
    ax1.scatter(X, y, color='b', marker='d', alpha=0.4, label='实验数据')
    ax1.set_xlabel("$value$")
    ax1.set_xticks(range(0, 250000, 25000))
    ax1.set_ylabel("$price$")
    ax1.set_yticks(range(0, 10000, 1000))



    # 计算残差估计的自变量
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # 使用调用sklearn中的linear_model中的现成的最小二乘法模型，名为LinearRegression
    slr = LinearRegression()
    # 进行模型拟合
    slr.fit(X_train, y_train)
    # 返回预测值
    # y_train_pred是用训练值所得出的预测值的y轴坐标
    # y_test_pred是用测试值所得出的预测值的y轴坐标
    y_train_pred = slr.predict(X_train)
    y_test_pred = slr.predict(X_test)
    # 这里到倒数第二行为止，定义残差评估的可视化图的样式
    # 定义训练数据的点的样式：蓝色，形状o样式，图示为训练数据
    plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='训练数据')
    # 绘制垂直线
    plt.axhline(y=0, c='k')
    # 定义测试数据的点的样式：浅绿，形状s样式，图示为测试数据
    plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='测试数据')
    # 定义和纵坐标的名称
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    # 图示规定显示在左上
    plt.legend(loc='upper left')
    # 绘制残差评估的参考线
    # 样式为红色，显示范围是x轴-15000到25000的范围
    plt.hlines(y=0, xmin=3500, xmax=5000, colors='red')
    # 显示残差估计图
    plt.show()


