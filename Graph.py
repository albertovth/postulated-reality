
# This code makes a graphical presentation of the decision tree

import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier




Y = [[7,10,17,10,4,2],[8,12,13,10,4,3],[11,12,13,9,3,3],[6,12,15,10,4,3],[8,11,16,9,4,4],[10,12,13,9,4,3],[9,11,17,11,1,3],[9,12,13,10,3,3],[10,10,15,9,4,2],[9,11,13,10,4,3],[8,11,16,10,1,4],[10,9,15,9,4,3],[10,8,16,8,4,3],[10,11,15,10,2,2],[8,13,13,10,2,4],[10,12,14,9,2,3],[11,10,16,9,1,3],[11,11,14,10,1,3],[9,10,16,8,3,3],[10,9,16,9,3,3]]



X = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,1,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,0,1],[0,0,1,0],[1,0,0,0]]



clf =DecisionTreeClassifier()

clf = clf.fit(X, Y)

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)

tree.plot_tree(clf)

fig.savefig('tree.png')

