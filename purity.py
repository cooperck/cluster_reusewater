import numpy as np
from sklearn import metrics

def purity_calculator(cluster, label):
  # 比较两个list元素多少，元素少的放第一个
  cluster_=list(cluster)#如果不是list，先强行转为list
  label_=list(label)
  a = {}
  for i in cluster_:
    if cluster_.count(i) >= 1:
      a[i] = cluster_.count(i)
  #print(a)
  x = len(a)

  b = {}
  for i in label_:
    if label_.count(i) > 1:
      b[i] = label_.count(i)
  #print(b)
  y = len(b)

  if x >= y:
    cluster, label = label, cluster

  cluster = np.array(cluster)
  label = np.array(label)
  indedata1 = {}
  for p in np.unique(label):  # np.unique用于提取label中的重复数字并排序输出
    indedata1[p] = np.argwhere(label == p)  # 返回等于p的数组元组的索引
  indedata2 = {}
  for q in np.unique(cluster):
    indedata2[q] = np.argwhere(cluster == q)

  count_all = []
  for i in indedata1.values():
    count = []
    for j in indedata2.values():
      a = np.intersect1d(i, j).shape[0]
      count.append(a)
    count_all.append(count)

  return sum(np.max(count_all, axis=0)) / len(cluster)



labels_true_JM=  [1, 1, 2, 2, 2]
labels_true_JDD= [1, 1, 1, 2, 2, 2, 1, 1]
labels_true_FRX=     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2]
labels_pred_FRX_45 = [1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 3]
labels_pred_FRX_36 = [1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 4, 4, 1, 1, 5, 1, 2, 1, 3]
labels_pred_FRX_29 = [1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 3, 1, 2, 1, 4]
labels_pred_JM_29 = [1, 1, 2, 3, 2]
labels_pred_JDD_29 = [1, 3, 1, 2, 2, 4, 5, 1]
labels_pred_JDD_34 = [1, 3, 1, 2, 2, 1, 1, 1]
labels_pred_JDD_36 = [1, 3, 1, 4, 2, 2, 1, 1]
labels_pred_JDD_40 = [1, 3, 1, 2, 2, 2, 1, 1]
labels_pred_JDD_X = [0,1,0,0,0,0,0,0]

labels_pred_ChatGPT_JM = [1, 1, 2, 2, 2]
labels_pred_ChatGPT_JDD = [1, 2, 1, 2, 2, 2, 2, 1]
labels_pred_ChatGPT_FRX = [1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 4, 4, 6, 6, 2, 1, 1, 7, 5]

print('FRX45:',purity_calculator(labels_true_FRX, labels_pred_FRX_45))
print('FRX36:',purity_calculator(labels_true_FRX, labels_pred_FRX_36))
print('FRX29:',purity_calculator(labels_true_FRX, labels_pred_FRX_29))
print('JM29:',purity_calculator(labels_true_JM, labels_pred_JM_29))
print('JDD29:',purity_calculator(labels_true_JDD, labels_pred_JDD_29))
print('JDD34:',purity_calculator(labels_true_JDD, labels_pred_JDD_34))
print('JDD36:',purity_calculator(labels_true_JDD, labels_pred_JDD_36))
print('JDD40:',purity_calculator(labels_true_JDD, labels_pred_JDD_40))

print('JDDX:',purity_calculator(labels_true_JDD, labels_pred_JDD_X))
print('JDD-ChatGPT:',purity_calculator(labels_true_JDD, labels_pred_ChatGPT_JDD))
print('FRX-ChatGPT:',purity_calculator(labels_true_FRX, labels_pred_ChatGPT_FRX))


print('-'*10,'直接调用函数','-'*10)

print('FRX45:',metrics.accuracy_score(labels_true_FRX, labels_pred_FRX_45))
print('FRX36:',metrics.accuracy_score(labels_true_FRX, labels_pred_FRX_36))
print('FRX29:',metrics.accuracy_score(labels_true_FRX, labels_pred_FRX_29))
print('JM29:',metrics.accuracy_score(labels_true_JM, labels_pred_JM_29))
print('JDD29:',metrics.accuracy_score(labels_true_JDD, labels_pred_JDD_29))
print('JDD34:',metrics.accuracy_score(labels_true_JDD, labels_pred_JDD_34))
print('JDD36:',metrics.accuracy_score(labels_true_JDD, labels_pred_JDD_36))
print('JDD40:',metrics.accuracy_score(labels_true_JDD, labels_pred_JDD_40))

print('JDDX:',metrics.accuracy_score(labels_true_JDD, labels_pred_JDD_X))
print('JDD-ChatGPT:',metrics.accuracy_score(labels_true_JDD, labels_pred_ChatGPT_JDD))
print('FRX-ChatGPT:',metrics.accuracy_score(labels_true_FRX, labels_pred_ChatGPT_FRX))