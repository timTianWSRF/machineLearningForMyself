import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# 生成随机数据
def generateData2():
    np.random.seed(4999)
    x1 = np.array(range(0, 20))
    x2 = np.array(range(20, 40))/3
    error = np.round(np.random.randn(20), )
    # 生成的初始表达式:y=0.5*x1+0.3*x2+b
    y = 0.5 * x1 + 0.3 * x2 + error
    return pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})


# 多元线性回归
def multivariableLinearRegression(data):
    xi = pd.DataFrame({'x1': data['x1'], 'x2': data['x2']})
    y = data['y'].values
    model = sm.OLS(y, xi)
    result = model.fit()
    # intercept是参数b
    # coef是参数x1和x2
    print(result.sumry())
    return result


def visualizeModel(data):
    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig)

    x1 = data['x1']
    x2 = data['x2']
    X, Y = np.meshgrid(x1, x2)
    Z = data[['y']]

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='Blues')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("多元线性回归拟合")
    plt.show()


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    data = generateData2()
    ols = multivariableLinearRegression(data)
    visualizeModel(data)

