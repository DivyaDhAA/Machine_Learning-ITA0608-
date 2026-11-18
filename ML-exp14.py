import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.DataFrame({
    'Area': [800, 1000, 1200, 1500, 1800, 2000],
    'Bedrooms': [2, 2, 3, 3, 4, 4],
    'Price': [2000000, 2500000, 3000000,
              3800000, 4500000, 5000000]
})

X = data[['Area', 'Bedrooms']]
y = data['Price']

model = LinearRegression()
model.fit(X, y)

new_house = [[1600, 3]]

print("Predicted House Price:",
      model.predict(new_house)[0])
