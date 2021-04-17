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
ys=[1,2,3,4,5,6,7]
ys_1=['kmeans','DBSCAN','AGNES','M15','M12','M10','M4']
xs=[1,2,3]
xs1=[1,1,1,1,1,1,1]
xs2=[2,2,2,2,2,2,2]
xs3=[3,3,3,3,3,3,3]
xs_1=['JM','JDD','FRX']
xs_2=['JDD']
xs_3=['FRX']
zs1=[1,1,1,1,1,1,0.8]
zs2=[0.88,1,0.88,0.88,0.75,0.75,0.63]
zs3=[0.95,1,0.95,0.79,1,0.68,0.74]

fig = plt.figure()  # 创建一个画布figure，然后在这个画布上加各种元素。
ax = Axes3D(fig)  # 将画布作用于 Axes3D 对象上。
ax.plot(xs1, ys, zs1, c='r', marker='X', label='JM purity')# 点为红色X形
plt.xticks(xs1,xs_1)
plt.yticks(ys,ys_1)
ax.legend()

ax.plot(xs2, ys, zs2, c='b', marker='o', label='JDD purity')# 点为蓝色o形
plt.xticks(xs2,xs_2)
#plt.yticks(ys2,ys_2)
ax.legend()

ax.plot(xs3, ys, zs3, c='g', marker='^', label='FRX purity')# 点为绿色^形
plt.xticks(xs3,xs_3)
#plt.yticks(ys3,ys_3)
ax.legend()

# 设置坐标轴
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
ax.set_xlabel('min samples')
ax.set_ylabel('eps')
ax.set_zlabel('ARI')
ax.set_title('min_samples-eps-ARI for FRX')
for x, y, z in zip(xs1, ys, zs1):
    ax.text(x, y, z, str(np.around(z, 2)), ha='center', va='bottom', fontsize=8.5)
for x, y, z in zip(xs2, ys, zs2):
    ax.text(x, y, z, str(np.around(z, 2)), ha='center', va='bottom', fontsize=8.5)
for x, y, z in zip(xs3, ys, zs3):
    ax.text(x, y, z, str(np.around(z, 2)), ha='center', va='bottom', fontsize=8.5)
  # 显示图形
plt.show()