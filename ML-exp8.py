import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)

print("Predicted Values:", pred)
print("Mean Squared Error:",
      mean_squared_error(y, pred))

plt.scatter(X, y)
plt.plot(X, pred)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
