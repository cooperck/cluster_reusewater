# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
y_label=['JM', 'JDD', 'FRX']
x_label=['K-Means', 'DBSCAN', 'AGNES', 'ChatGPT', 'M15', 'M12', 'M10', 'M4']

purity_plot = np.array([[1, 1, 1, 1, 1, 1, 1, 0.8],
                        [0.88, 1, 0.88, 0.75, 0.88, 0.75, 0.75, 0.63],
                        [0.95, 1, 1, 0.53, 0.79, 1, 0.68, 0.74]
                        ])

NMI_plot = np.array([[1, 1, 1, 1, 1, 1, 1, 0.78],
                        [0.81, 1, 0.81, 0.36, 0.81, 0.44, 0.71, 0.61],
                        [0.9, 1, 1, 0.35, 0.42, 1, 0.46, 0.58]
                        ])

ARI_plot = np.array([[1, 1, 1, 1, 1, 1, 1, 0.32],
                        [0.71, 1, 0.71, 0.14, 0.71, 0.27, 0.56, 0.55],
                        [0.98, 1, 1, 0.15, 0.41, 1, 0.29, 0.47]
                        ])

EF_plot = np.array([[1, 1, 1, 1, 1, 1, 1, 0.8],
                        [0.95, 1.00, 0.95, 0.87, 0.95, 0.93, 0.70, 0.47],
                        [0.98, 1.00, 1.00, 0.60, 0.93, 1, 0.64, 0.75]
                        ])

# 第一行JM第二行JDD第三行FRX
plt.xticks(np.arange(len(x_label)),labels=x_label, rotation=45, rotation_mode="anchor",ha="right")
plt.yticks(np.arange(len(y_label)),labels=y_label)
plt.title("NMI of Machine Vs Human Results")

for i in range(len(y_label)):
    for j in range(len(x_label)):
        text= plt.text(j, i, NMI_plot[i,j], ha="center",va="center",color="Black")

plt.imshow(NMI_plot, cmap="Reds")
plt.colorbar()
plt.tight_layout()
plt.show()


