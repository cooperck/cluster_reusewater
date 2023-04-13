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


##测试数据导入情况
data_file = u'D:\博士论文\聚类\聚类测试数据.xls'
a_1 = import_excel_matrix(data_file,3)# 获取测试数据第3个sheet表
#print(a_1)
##测试标准化基础数据导入情况
b = import_excel_matrix(data_file,2)# 获取测试数据第2个sheet表
#print(b)


##数据预处理
def logistic(x):  ##定义一个逻辑函数
  return 1 / (0.5 + np.exp(-x))

#def exponential(x):  ##定义一个函数,因为考虑4倍为RO通常的回收率
  #return np.exp(x-0.25)
#  return x*4
for i in range(len(a_1)):
  a_1[i,13]=abs(a_1[i,13]-7) #调整pH
a_2=a_1/b[0,:]#除以上限

 #  for j in range(0, 11):
 #    a_2[i, j] = exponential(a_2[i, j])#对一类污染物超标倍数进行指数倍变换
 # for k in range(14,21):
 #   a_2[i, k]=logistic(a_2[i, k])#色度、物化距离指标进行logistic变换，超标后有一个上限值
#print(a_2)
a=a_2


##定义一个kmeans计算过程，程序结束后运行
def kmeans():
  data=a
  #开始分类

  km =KMeans(n_clusters=2,init='k-means++',algorithm='full')#full-EM算法 'k-means++':启发式选择一个初始聚类中心
  predict = km.fit_predict(data)
  print(predict)#打印分类结果
  #定义一个颜色列表
  colors = ["red","green","yellow","blue"]
  #替换predict内容为颜色
  predict_1 = [colors[i] for i in predict]
  #可视化结果
  plt.scatter(data[:,0],data[:,1],c=predict_1)
  # 可视化聚类中心
  plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=250,marker="*",c="red",label="cluster center")
  plt.show()

  #计算CH系数 轮廓系数
  labels = km.labels_
  print(metrics.calinski_harabasz_score(data, labels))
  print(metrics.silhouette_score(data, labels, metric='euclidean'))#此处算法一定要与上面ag中的算法一致

  # 获取簇的标号
  cluster_labels = np.unique(predict)
  # 获取簇的个数
  n_cluster = cluster_labels.shape[0]
  # 基于euclidean距离计算轮廓系数
  silhoutte_vals = silhouette_samples(data, predict, metric='euclidean')
  y_ax_lower, y_ax_upper = 0, 0
  yticks = []
  for i, c in enumerate(cluster_labels):
    # 获取不同簇的轮廓系数
    c_silhouette_vals = silhoutte_vals[predict == c]
    # 对簇中样本的轮廓系数由小到大进行排序
    c_silhouette_vals.sort()
    # 获取到簇轮廓系数的个数
    y_ax_upper += len(c_silhouette_vals)
    # 获取不同的颜色
    color = cm.jet(i / n_cluster)
    # 绘制水平直方图
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)
    # 获取显示y轴刻度的位置
    yticks.append((y_ax_lower + y_ax_upper) / 2)
    # 下一个y轴的起点位置
    y_ax_lower += len(c_silhouette_vals)
  # 获取轮廓系数的平均值
  silhoutte_avg = np.mean(silhoutte_vals)
  # 绘制一条平行y轴的轮廓系数平均值的虚线
  plt.axvline(silhoutte_avg, color='red', linestyle="--")
  # 设置y轴显示的刻度
  plt.yticks(yticks, cluster_labels + 1)
  plt.ylabel("cluster")
  plt.xlabel("Silhouette Coefficient")
  plt.show()



##定义一个DBSCAN计算过程，程序结束后运行
def dbscan():
  data=a
  #开始分类
  db = DBSCAN(eps = 2, min_samples=2,metric='euclidean')#esp聚类半径，min半径内最少点数，metric距离
  predict = db.fit_predict(data)
  print(predict)#打印分类结果
  #定义一个颜色列表
  colors = ["red","blue", "green","yellow"]
  #替换predict内容为颜色
  predict_1 = [colors[i] for i in predict]
  #可视化结果
  plt.scatter(data[:,0],data[:,1],c=predict_1)
  plt.show()
  #计算CH系数 轮廓系数
  labels = db.labels_
  print(metrics.calinski_harabasz_score(data, labels))
  print(metrics.silhouette_score(data, labels, metric='euclidean'))#此处算法一定要与上面ag中的算法一致


##定义一个AGNES从下向上层次聚类计算过程，程序结束后运行
def hac():
  data = a
  # 开始分类
  ag = AgglomerativeClustering(n_clusters=2, affinity='euclidean',linkage='ward')  # affinity距离,linkage连接算法
  predict = ag.fit_predict(data)
  print(predict)  # 打印分类结果
  # 定义一个颜色列表
  colors = ["blue", "yellow","red", "green" ]
  # 替换predict内容为颜色
  predict_1 = [colors[i] for i in predict]
  # 可视化结果
  plt.scatter(data[:, 0], data[:, 1], c=predict_1)
  plt.show()
  #绘制嵌套聚类树状图 纵坐标为distance
  tree = linkage(data,method='ward',metric='euclidean')#此处算法一定要与上面ag中的算法一致
  dendrogram(tree)
  plt.show()
  #计算CH系数 轮廓系数
  labels = ag.labels_
  print(metrics.calinski_harabasz_score(data, labels))
  print(metrics.silhouette_score(data, labels, metric='euclidean'))#此处算法一定要与上面ag中的算法一致


if __name__ == '__main__':
  kmeans()#程序结束后运行kmeans
  #dbscan()#程序结束后运行dbscan
  #hac()#程序结束后运行hac