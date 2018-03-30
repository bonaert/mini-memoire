import numpy as np
from sklearn.model_selection import StratifiedKFold

X = np.array([10 * i for i in range(10)])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
skf = StratifiedKFold(n_splits=5)
print()
print()
print()
for train, test in skf.split(X, y):
    print("%s %s" % (train, test))
    print("X: %s %s" % (X[train], X[test]))
    print("Y: %s %s" % (y[train], y[test]))
    # print("%s %s" % (X[train], y[train]))
    # print("%s %s" % (X[test], y[test]))
    print()
