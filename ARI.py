from sklearn import metrics

labels_true_JM=  [1, 1, 2, 2, 2]
labels_true_JDD= [1, 1, 1, 2, 2, 2, 1, 1]
labels_true_FRX= [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2]
labels_pred_FRX_45 = [1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 3]
labels_pred_FRX_35 = [1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 4, 4, 1, 1, 5, 1, 2, 1, 3]
labels_pred_FRX_29 = [1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 3, 1, 2, 1, 4]
labels_pred_JM_29 = [1,1,2,3,2]
labels_pred_JDD_29 = [1, 2, 1, 3, 3, 4, 5, 1]
labels_pred_JDD_34 = [1,3,1,2,2,1,1,1]

labels_pred_ChatGPT_JM = [1, 1, 2, 2, 2]
labels_pred_ChatGPT_JDD = [1, 2, 1, 2, 2, 2, 2, 1]
labels_pred_ChatGPT_FRX = [1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 4, 4, 6, 6, 2, 1, 1, 7, 5]

print(metrics.adjusted_rand_score(labels_true_FRX, labels_pred_FRX_45))
print(metrics.adjusted_rand_score(labels_true_FRX, labels_pred_FRX_35))
print(metrics.adjusted_rand_score(labels_true_FRX, labels_pred_FRX_29))
print('ChatGPT-JM', metrics.adjusted_rand_score(labels_true_JM, labels_pred_ChatGPT_JM))
print('ChatGPT-JDD', metrics.adjusted_rand_score(labels_true_JDD, labels_pred_ChatGPT_JDD))
print('ChatGPT-FRX', metrics.adjusted_rand_score(labels_true_FRX, labels_pred_ChatGPT_FRX))

print('JDD:',metrics.adjusted_rand_score(labels_true_JDD, labels_pred_JDD_29))
print('JM:',metrics.adjusted_rand_score(labels_true_JM, labels_pred_JM_29))

print('JDD34:',metrics.adjusted_rand_score(labels_true_JDD, labels_pred_JDD_34))