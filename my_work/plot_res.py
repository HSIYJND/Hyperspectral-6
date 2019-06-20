from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt


yt = [1] * 100 + [0] * 100
yp1 = np.random.uniform(0.4, 0.8, (100, 1))
yp2 = np.random.uniform(0.2, 0.6, (100, 1))

yp = np.concatenate([yp1, yp2], axis=0)

a, b, c = roc_curve(yt, yp)
print(a.shape, b.shape, c.shape)

plt.plot(a, b, c='y', lw=2)
plt.show()
