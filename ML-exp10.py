import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

X = np.array([
    [1, 2], [2, 3], [3, 3],
    [8, 8], [9, 8], [8, 9]
])

model = GaussianMixture(n_components=2,
                        random_state=42)

model.fit(X)

labels = model.predict(X)

print("Cluster Labels:", labels)
print("Cluster Means:")
print(model.means_)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()
