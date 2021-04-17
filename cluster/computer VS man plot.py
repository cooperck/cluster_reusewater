from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xlrd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import metrics
from sklearn.metrics import silhouette_samples
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator
from scipy.optimize import linear_sum_assignment

#x轴为聚类算法，y轴为案例，z轴为purit、NMI、ARI
xs=[1,2,3,4,5,6,7]
xs_1=['KMeans','DBSCAN','AGNES','M15','M12','M10','M4']
ys=[1,2,3]
ys1=[1,1,1,1,1,1,1]
ys2=[2,2,2,2,2,2,2]
ys3=[3,3,3,3,3,3,3]
ys_1=['JM','JDD','FRX']
ys_2=['JDD']
ys_3=['FRX']
purity_zs1=[1.00,1.00,1.00,1.00,1.00,1.00,0.80]
purity_zs2=[0.88,1.00,0.88,0.88,0.75,0.75,0.63]
purity_zs3=[0.95,1.00,1.00,0.79,1.00,0.68,0.74]
NMI_zs1=[1.00,1.00,1.00,1.00,1.00,1.00,0.78]
NMI_zs2=[0.81,1.00,0.81,0.81,0.44,0.71,0.61]
NMI_zs3=[0.90,1.00,1.00,0.42,1.00,0.46,0.58]
ARI_zs1=[1.00,1.00,1.00,1.00,1.00,1.00,0.32]
ARI_zs2=[0.71,1.00,0.71,0.71,0.27,0.56,0.55]
ARI_zs3=[0.98,1.00,1.00,0.41,1.00,0.29,0.47]



#画purity
fig = plt.figure()  # 创建一个画布figure，然后在这个画布上加各种元素。
ax = Axes3D(fig)  # 将画布作用于 Axes3D 对象上。
ax.plot(xs, ys1, purity_zs1, c='r', marker='X', label='JM Purity')# 点为红色X形
plt.xticks(xs,xs_1)
plt.yticks(ys,ys_1)
ax.legend()

ax.plot(xs, ys2, purity_zs2, c='b', marker='o', label='JDD Purity')# 点为蓝色o形
plt.xticks(xs,xs_1)
#plt.yticks(ys2,ys_2)
ax.legend()

ax.plot(xs, ys3, purity_zs3, c='g', marker='^', label='FRX Purity')# 点为绿色^形
plt.xticks(xs,xs_1)
#plt.yticks(ys3,ys_3)
ax.legend()

# 设置坐标轴
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
ax.set_xlabel('Algorithms or Human Ages')
ax.set_ylabel('Cases')
ax.set_zlabel('Purity')
ax.set_title('Purity of Computer Vs Manual Results')
for x, y, z in zip(xs, ys1, purity_zs1):
    ax.text(x, y, z, str(np.around(z, 2)), ha='center', va='bottom', fontsize=8.5)
for x, y, z in zip(xs, ys2, purity_zs2):
    ax.text(x, y, z, str(np.around(z, 2)), ha='center', va='bottom', fontsize=8.5)
for x, y, z in zip(xs, ys3, purity_zs3):
    ax.text(x, y, z, str(np.around(z, 2)), ha='center', va='bottom', fontsize=8.5)
  # 显示图形
plt.show()



#画NMI
fig = plt.figure()  # 创建一个画布figure，然后在这个画布上加各种元素。
ax = Axes3D(fig)  # 将画布作用于 Axes3D 对象上。
ax.plot(xs, ys1, NMI_zs1, c='r', marker='X', label='JM NMI')# 点为红色X形
plt.xticks(xs,xs_1)
plt.yticks(ys,ys_1)
ax.legend()

ax.plot(xs, ys2, NMI_zs2, c='b', marker='o', label='JDD NMI')# 点为蓝色o形
plt.xticks(xs,xs_1)
#plt.yticks(ys2,ys_2)
ax.legend()

ax.plot(xs, ys3, NMI_zs3, c='g', marker='^', label='FRX NMI')# 点为绿色^形
plt.xticks(xs,xs_1)
#plt.yticks(ys3,ys_3)
ax.legend()

# 设置坐标轴
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
ax.set_xlabel('Algorithms or Human Ages')
ax.set_ylabel('Cases')
ax.set_zlabel('NMI')
ax.set_title('NMI of Computer Vs Manual Results')
for x, y, z in zip(xs, ys1, NMI_zs1):
    ax.text(x, y, z, str(np.around(z, 2)), ha='center', va='bottom', fontsize=8.5)
for x, y, z in zip(xs, ys2, NMI_zs2):
    ax.text(x, y, z, str(np.around(z, 2)), ha='center', va='bottom', fontsize=8.5)
for x, y, z in zip(xs, ys3, NMI_zs3):
    ax.text(x, y, z, str(np.around(z, 2)), ha='center', va='bottom', fontsize=8.5)
  # 显示图形
plt.show()




#画ARI
fig = plt.figure()  # 创建一个画布figure，然后在这个画布上加各种元素。
ax = Axes3D(fig)  # 将画布作用于 Axes3D 对象上。
ax.plot(xs, ys1, ARI_zs1, c='r', marker='X', label='JM ARI')# 点为红色X形
plt.xticks(xs,xs_1)
plt.yticks(ys,ys_1)
ax.legend()

ax.plot(xs, ys2, ARI_zs2, c='b', marker='o', label='JDD ARI')# 点为蓝色o形
plt.xticks(xs,xs_1)
#plt.yticks(ys2,ys_2)
ax.legend()

ax.plot(xs, ys3, ARI_zs3, c='g', marker='^', label='FRX ARI')# 点为绿色^形
plt.xticks(xs,xs_1)
#plt.yticks(ys3,ys_3)
ax.legend()

# 设置坐标轴
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
ax.set_xlabel('Algorithms or Human Ages')
ax.set_ylabel('Cases')
ax.set_zlabel('ARI')
ax.set_title('ARI of Computer Vs Manual Results')
for x, y, z in zip(xs, ys1, ARI_zs1):
    ax.text(x, y, z, str(np.around(z, 2)), ha='center', va='bottom', fontsize=8.5)
for x, y, z in zip(xs, ys2, ARI_zs2):
    ax.text(x, y, z, str(np.around(z, 2)), ha='center', va='bottom', fontsize=8.5)
for x, y, z in zip(xs, ys3, ARI_zs3):
    ax.text(x, y, z, str(np.around(z, 2)), ha='center', va='bottom', fontsize=8.5)
  # 显示图形
plt.show()