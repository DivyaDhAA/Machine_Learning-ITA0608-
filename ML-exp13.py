import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.DataFrame({
    'Year': [2015, 2016, 2017, 2018, 2019, 2020],
    'Mileage': [50000, 45000, 40000, 35000, 30000, 25000],
    'Price': [300000, 350000, 400000, 450000, 500000, 550000]
})

X = data[['Year', 'Mileage']]
y = data['Price']

model = LinearRegression()
model.fit(X, y)

new_car = [[2021, 20000]]

print("Predicted Car Price:",
      model.predict(new_car)[0])
