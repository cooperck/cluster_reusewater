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


x = ['s1','s2']
y = []
z = []
a = []
for i in range(1,3):
    print(i)
    y.append(i*i)
    z.append(i*100)
    a.append(i/2)
sizes=z
colors=a
plt.scatter(x,y, c=colors, s=sizes, alpha=0.3, cmap='viridis')
plt.plot(x,y,label='test1',marker = "o",markersize=5)
cbar=plt.colorbar() #显示颜色条
cbar.set_label("test",fontsize=12)

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.title('题目', fontsize=18)
plt.ylabel("cluster")
plt.xlabel("Silhouette Coefficient")
plt.legend()#加图例
plt.grid()#加网格

plt.show()