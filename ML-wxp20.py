import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.DataFrame({
    'Month': [1, 2, 3, 4, 5, 6],
    'Sales': [10000, 12000, 15000,
              18000, 20000, 25000]
})

X = data[['Month']]
y = data['Sales']

model = LinearRegression()
model.fit(X, y)

future = pd.DataFrame({'Month': [7, 8, 9]})

prediction = model.predict(future)

for i in range(3):
    print("Month", i + 7,
          "Predicted Sales:", prediction[i])
