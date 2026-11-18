import numpy as np

X = np.array([[0, 0], [0, 1],
              [1, 0], [1, 1]])

y = np.array([[0], [1], [1], [0]])

np.random.seed(1)

w1 = np.random.randn(2, 4)
w2 = np.random.randn(4, 1)

lr = 0.5

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for epoch in range(10000):

    h = sigmoid(X @ w1)
    output = sigmoid(h @ w2)

    error = y - output

    d2 = error * output * (1 - output)
    d1 = (d2 @ w2.T) * h * (1 - h)

    w2 += lr * h.T @ d2
    w1 += lr * X.T @ d1

print("Predicted Output:")
print(np.round(output))
