import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

data = pd.DataFrame({
    'RAM': [2, 3, 4, 6, 8, 12],
    'Battery': [2500, 3000, 3500, 4000, 4500, 5000],
    'Price_Range': [0, 0, 1, 1, 2, 3]
})

X = data[['RAM', 'Battery']]
y = data['Price_Range']

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

new_mobile = [[8, 4500]]

result = model.predict(new_mobile)[0]

print("Predicted Price Category:", result)
