from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import xlrd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import metrics

labels_true = [0, 0, 0, 1, 3, 3]
label = np.array(labels_true)
print(np.unique(label))

indedata1 = {}
for p in np.unique(label):  #np.unique用于提取label中的重复数字并排序输出
    indedata1[p] = np.argwhere(label == p)  #返回等于p的数组元组的索引

print(list(indedata1))