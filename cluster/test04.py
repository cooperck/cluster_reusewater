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
sse=[1,2,3,4,5]
sc=[3,4,5,6]
ch=[2,4,6,7]

x = [i for i in range(1, 6)] #聚类数量
print(x)
y = sse
print(y)
z = [i*2000 for i in sc]# 设置圆大小，2000代表1
z.insert(0,0)
print(z)
w = ch  # 设置颜色
w.insert(0,0)
print(w)