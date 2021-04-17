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

def import_excel_matrix(path,n):
  table = xlrd.open_workbook(path).sheets()[n-1] # 获取第n个sheet表
  row = table.nrows # 行数
  col = table.ncols # 列数
  datamatrix = np.zeros((row, col)) # 生成一个nrows行*ncols列的初始矩阵
  for i in range(col): # 对列进行遍历
    cols = np.mat(table.col_values(i)) # 把list转换为矩阵进行矩阵操作
    datamatrix[:, i] = cols # 按列把数据存进矩阵中
  return datamatrix


##测试数据导入情况
data_file = u'D:\博士论文\聚类\聚类测试数据.xls'
a = import_excel_matrix(data_file,1)# 获取测试数据第1个sheet表
print(a)


def hac():
  data = a
  # 开始分类
  ag = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='single')  # affinity距离,linkage连接算法
  predict = ag.fit_predict(data)
  print(predict)  # 打印分类结果
  # 定义一个颜色列表
  colors = ["blue", "yellow", "red", "green"]
  # 替换predict内容为颜色
  predict = [colors[i] for i in predict]
  # 可视化结果
  plt.scatter(data[:, 0], data[:, 1], c=predict)
  plt.show()

  Z = linkage(a,method='single',metric='euclidean')#此处算法一定要与上面ag中的算法一致
  dendrogram(Z)
  plt.show()

if __name__ == '__main__':
    hac()  # 程序结束后运行hac




