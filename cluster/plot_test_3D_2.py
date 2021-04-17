import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


xs=[0,1,2,3,4,5]
ys=[6,7,8,9,0,5]
zs=[24,53,64,12,61,23]
xs1 = np.random.randint(30,40,100)
ys1 = np.random.randint(20,30,100)
zs1 = np.random.randint(10,20,100)

fig = plt.figure() # 创建一个画布figure，然后在这个画布上加各种元素。
ax = Axes3D(fig) # 将画布作用于 Axes3D 对象上。
# 基于ax变量绘制三维图
# xs表示x方向的变量
# ys表示y方向的变量
# zs表示z方向的变量，这三个方向上的变量都可以用list的形式表示
# m表示点的形式，o是圆形的点，^是三角形（marker)
# c表示颜色（color for short）
ax.scatter(xs, ys, zs, c='r', marker='^', label='散点')  # 点为红色三角形

# 显示图例
ax.legend()
# 设置坐标轴
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('Scatter Plot')

# 显示图形
plt.show()


fig = plt.figure() # 创建一个画布figure，然后在这个画布上加各种元素。
ax = Axes3D(fig) # 将画布作用于 Axes3D 对象上。
# 基于ax变量绘制三维图
# xs表示x方向的变量
# ys表示y方向的变量
# zs表示z方向的变量，这三个方向上的变量都可以用list的形式表示
# m表示点的形式，o是圆形的点，^是三角形（marker)
# c表示颜色（color for short）
ax.plot(xs1, ys1, zs1, label='曲线')
# 显示图例
ax.legend()
# 设置坐标轴
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('Scatter Plot')


# 显示图形
plt.show()