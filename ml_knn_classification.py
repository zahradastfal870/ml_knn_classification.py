import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets

# Load Iris dataset (use only two features: sepal length, sepal width)
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# List of k values
n_neighbors_list = [1, 3, 5, 10]

# Create mesh for plotting decision boundaries
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Train and plot for each k
for n_neighbors in n_neighbors_list:
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X, y)

    # Predict on mesh
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title(f"Machine Learning • kNN Decision Boundary (k = {n_neighbors})")
    plt.savefig("knn_k1.png")  
    plt.savefig("knn_k3.png")
    plt.savefig("knn_k5.png")
    plt.savefig("knn_k10.png")
