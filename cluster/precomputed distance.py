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


##定义一个函数从excel读取数据
def import_excel_matrix(path,n):
  table = xlrd.open_workbook(path).sheets()[n-1] # 获取第n个sheet表
  row = table.nrows # 行数
  col = table.ncols # 列数
  datamatrix = np.zeros((row, col)) # 生成一个nrows行*ncols列的初始矩阵
  for i in range(col): # 对列进行遍历
    cols = np.mat(table.col_values(i)) # 把list转换为矩阵进行矩阵操作
    datamatrix[:, i] = cols # 按列把数据存进矩阵中
  return datamatrix


##定义一个四舍五入矩阵的函数,保留decPts位
def matrixRound(M, decPts):
    row_1 = M.shape[0]  # 行数
    col_1 = M.shape[1]  # 列数
    M_1 = np.zeros(row_1, col_1)  # 生成一个row_1行*col_1列的初始矩阵
    # 对行循环
    for index in range(len(M)):
        # 对列循环
        for _index in range(len(M[index])):
            M_1[index][_index] = round(M[index][_index], decPts)
    return M_1



##测试数据导入情况
data_file = u'D:\博士论文\聚类\聚类测试数据.xls'
a = import_excel_matrix(data_file,1)# 获取测试数据第1个sheet表
print(a)




#使用原装欧氏距离
def hac1():
  data = a
  # 开始分类
  ag = AgglomerativeClustering(n_clusters=3, affinity='euclidean',linkage='single')  # affinity距离,linkage连接算法
  predict = ag.fit_predict(data)
  print(predict)  # 打印分类结果
  # 定义一个颜色列表
  colors = ["blue", "yellow","red", "green" ]
  # 替换predict内容为颜色
  predict = [colors[i] for i in predict]
  # 可视化结果
  plt.scatter(data[:, 0], data[:, 1], c=predict)
  plt.show()
  #绘制嵌套聚类树状图
  tree = linkage(data,method='single',metric='euclidean')#此处算法一定要与上面ag中的算法一致
  dendrogram(tree)
  plt.show()
  #计算CH系数 轮廓系数
  labels = ag.labels_
  print(metrics.calinski_harabasz_score(data, labels))
  print(metrics.silhouette_score(data, labels, metric='euclidean'))#此处算法一定要与上面ag中的算法一致



##定义自定义距离
def hac2():
  data = a
  #计算自定义距离矩阵，此处先用欧氏距离代替，可以在自定义一个距离计算函数代替euclidean_distances函数
  distance_matrix = matrixRound(euclidean_distances(data),4)
  # 开始分类
  ag = AgglomerativeClustering(n_clusters=3, affinity='precomputed',linkage='single')  # affinity距离,linkage连接算法
  predict = ag.fit_predict(distance_matrix)
  print(predict)  # 打印分类结果
  # 定义一个颜色列表
  colors = ["blue", "yellow","red", "green" ]
  # 替换predict内容为颜色
  predict = [colors[i] for i in predict]
  # 可视化结果
  plt.scatter(data[:, 0], data[:, 1], c=predict)
  plt.show()
  #绘制嵌套聚类树状图
  distance_matrix_1 = dist.squareform(distance_matrix)
  print(distance_matrix_1)
  tree = linkage(distance_matrix_1,method='single')#此处算法一定要与上面ag中的算法一致
  dendrogram(tree)
  plt.show()
  #计算CH系数 轮廓系数
  labels = ag.labels_
  print(metrics.calinski_harabasz_score(data, labels))
  print(metrics.silhouette_score(distance_matrix, labels, metric='precomputed'))#此处算法一定要与上面ag中的算法一致


if __name__ == '__main__':

  hac1()  #程序结束后运行hac
  hac2()  # 程序结束后运行hac