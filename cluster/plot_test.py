from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import xlrd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances
import scipy.spatial.distance as dist


x = []
y = []
z = []
a = []

for i in range(1,15):
    print(i)
    x.append(i)
    y.append(i*i)
    z.append((i/15)*2000)           #设置圆大小，2000代表1
    a.append(i)                     #设置颜色
sizes=z
colors= a
plt.scatter(x,y, c=colors, s=sizes, alpha=0.3, cmap='rainbow')
plt.plot(x,y,label='test1',marker = "o",markersize=5)
cbar=plt.colorbar()                 #显示颜色条
cbar.set_label("test",fontsize=12)  #显示标签

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.title('题目', fontsize=18)#显示标题
plt.ylabel("Silhouette Coefficient")#显示y标签
plt.xlabel("Number of Clusters")#显示x标签
plt.legend() #加图例
plt.grid()   #加网格

l=3
m=60
plt.axvline(l, color='red', linestyle="--")# 绘制一条平行y轴的虚线（取值为l）
plt.axhline(m, color='red', linestyle="--")# 绘制一条平行x轴的虚线（取值为m）

plt.show()