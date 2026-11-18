import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])

linear = LinearRegression()
linear.fit(X, y)
p1 = linear.predict(X)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
p2 = model.predict(X_poly)

print("Linear MSE:", mean_squared_error(y, p1))
print("Polynomial MSE:", mean_squared_error(y, p2))

plt.scatter(X, y, label="Actual")
plt.plot(X, p1, label="Linear")
plt.plot(X, p2, label="Polynomial")
plt.legend()
plt.show()
