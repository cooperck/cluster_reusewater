import math
import numpy as np
from sklearn import metrics


def NMI(A, B):
    # 样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    # 互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A == idA)  # 输出满足条件的元素的下标
            idBOccur = np.where(B == idB)
            idABOccur = np.intersect1d(idAOccur, idBOccur)  # Find the intersection of two arrays.
            px = 1.0 * len(idAOccur[0]) / total
            py = 1.0 * len(idBOccur[0]) / total
            pxy = 1.0 * len(idABOccur) / total
            MI = MI + pxy * math.log(pxy / (px * py) + eps, 2)
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0 * len(np.where(A == idA)[0])
        Hx = Hx - (idAOccurCount / total) * math.log(idAOccurCount / total + eps, 2)
        Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0 * len(np.where(B == idB)[0])
        Hy = Hy - (idBOccurCount / total) * math.log(idBOccurCount / total + eps, 2)
    MIhat = 2.0 * MI / (Hx + Hy)
    return MIhat


if __name__ == '__main__':

    labels_true_JM = np.array([1, 1, 2, 2, 2])
    labels_true_JDD = np.array([1, 1, 1, 2, 2, 2, 1, 1])
    labels_true_FRX = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2])
    labels_pred_FRX_45 = np.array([1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 3])
    labels_pred_FRX_35 = np.array([1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 4, 4, 1, 1, 5, 1, 2, 1, 3])
    labels_pred_FRX_29 = np.array([1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 3, 1, 2, 1, 4])
    labels_pred_JM_29 = np.array([1, 1, 2, 3, 2])
    labels_pred_JDD_29 = np.array([1, 2, 1, 3, 3, 4, 5, 1])
    labels_pred_JDD_34 = np.array([1, 3, 1, 2, 2, 1, 1, 1])

    labels_pred_ChatGPT_JM = np.array([1, 1, 2, 2, 2])
    labels_pred_ChatGPT_JDD = np.array([1, 2, 1, 2, 2, 2, 2, 1])
    labels_pred_ChatGPT_FRX = np.array([1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 4, 4, 6, 6, 2, 1, 1, 7, 5])

    print(NMI(labels_true_FRX, labels_pred_FRX_45))
    print(metrics.normalized_mutual_info_score(labels_true_FRX, labels_pred_FRX_45))  # 直接调用sklearn中的函数
    print(metrics.normalized_mutual_info_score(labels_pred_FRX_45, labels_true_FRX ))
    print('-'*20)
    print(NMI(labels_true_FRX, labels_pred_FRX_35))
    print(metrics.normalized_mutual_info_score(labels_true_FRX, labels_pred_FRX_35))  # 直接调用sklearn中的函数
    print(metrics.normalized_mutual_info_score(labels_pred_FRX_35, labels_true_FRX))
    print('-' * 20)
    print(NMI(labels_true_FRX, labels_pred_FRX_29))
    print(metrics.normalized_mutual_info_score(labels_true_FRX, labels_pred_FRX_29))  # 直接调用sklearn中的函数
    print(metrics.normalized_mutual_info_score(labels_pred_FRX_29, labels_true_FRX))

    print('-' * 20)
    print('JM29:',NMI(labels_true_JM, labels_pred_JM_29))
    print('JM29:',metrics.normalized_mutual_info_score(labels_true_JM, labels_pred_JM_29))  # 直接调用sklearn中的函数
    print('JM29:',metrics.normalized_mutual_info_score(labels_pred_JM_29, labels_true_JM))

    print('-' * 20)
    print('JDD29:', NMI(labels_true_JDD, labels_pred_JDD_29))
    print('JDD29:', metrics.normalized_mutual_info_score(labels_true_JDD, labels_pred_JDD_29))  # 直接调用sklearn中的函数
    print('JDD29:', metrics.normalized_mutual_info_score(labels_pred_JDD_29, labels_true_JDD))

    print('-' * 20)
    print('JDD34:', NMI(labels_true_JDD, labels_pred_JDD_34))
    print('JDD34:', metrics.normalized_mutual_info_score(labels_true_JDD, labels_pred_JDD_34))  # 直接调用sklearn中的函数
    print('JDD34:', metrics.normalized_mutual_info_score(labels_pred_JDD_34, labels_true_JDD))

    print('-' * 20)
    print('JDD-ChatGPT:', NMI(labels_true_JDD, labels_pred_ChatGPT_JDD))
    print('JDD-ChatGPT:', metrics.normalized_mutual_info_score(labels_true_JDD, labels_pred_ChatGPT_JDD))  # 直接调用sklearn中的函数
    print('JDD-ChatGPT:', metrics.normalized_mutual_info_score(labels_pred_ChatGPT_JDD, labels_true_JDD))

    print('-' * 20)
    print('FRX-ChatGPT:', NMI(labels_true_FRX, labels_pred_ChatGPT_FRX))
    print('FRX-ChatGPT:', metrics.normalized_mutual_info_score(labels_true_FRX, labels_pred_ChatGPT_FRX))  # 直接调用sklearn中的函数
    print('FRX-ChatGPT:', metrics.normalized_mutual_info_score(labels_pred_ChatGPT_FRX, labels_true_FRX))