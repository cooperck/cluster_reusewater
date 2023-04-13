import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment


def ACC(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true =np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

labels_true_JM=  [1, 1, 2, 2, 2]
labels_true_JDD= [1, 1, 1, 2, 2, 2, 1, 1]
labels_true_FRX= [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2]
labels_pred_FRX_45 = [1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 3]
labels_pred_FRX_36 = [1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 4, 4, 1, 1, 5, 1, 2, 1, 3]
labels_pred_FRX_29 = [1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 3, 1, 2, 1, 4]
labels_pred_JM_29 = [1, 1, 2, 3, 2]
labels_pred_JDD_29 = [1, 3, 1, 2, 2, 4, 5, 1]
labels_pred_JDD_34 = [1, 3, 1, 2, 2, 1, 1, 1]
labels_pred_JDD_36 = [1, 3, 1, 4, 2, 2, 1, 1]
labels_pred_JDD_40 = [1, 3, 1, 2, 2, 2, 1, 1]
labels_pred_JDD_X = [0,0,0,0,0,0,0,0]
labels_pred_JM_X = [0, 0, 1, -1, 1]

labels_pred_ChatGPT_JM = [1, 1, 2, 2, 2]
labels_pred_ChatGPT_JDD = [1, 2, 1, 2, 2, 2, 2, 1]
labels_pred_ChatGPT_FRX = [1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 4, 4, 6, 6, 2, 1, 1, 7, 5]


print('FRX45:',ACC(labels_true_FRX, labels_pred_FRX_45))
print('FRX36:',ACC(labels_true_FRX, labels_pred_FRX_36))
print('FRX29:',ACC(labels_true_FRX, labels_pred_FRX_29))
print('JM29:',ACC(labels_true_JM, labels_pred_JM_29))
print('JDD29:',ACC(labels_true_JDD, labels_pred_JDD_29))
print('JDD34:',ACC(labels_true_JDD, labels_pred_JDD_34))
print('JDD36:',ACC(labels_true_JDD, labels_pred_JDD_36))
print('JDD40:',ACC(labels_true_JDD, labels_pred_JDD_40))

print('JDDX:',ACC(labels_true_JDD, labels_pred_JDD_X))
print('JMX:',ACC(labels_true_JM, labels_pred_JM_X))
print(labels_pred_FRX_29)

print('JDD-ChatGPT:',ACC(labels_true_JDD, labels_pred_ChatGPT_JDD))
print('FRX-ChatGPT:',ACC(labels_true_FRX, labels_pred_ChatGPT_FRX))