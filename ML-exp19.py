import pandas as pd
from sklearn.naive_bayes import GaussianNB

data = pd.DataFrame({
    'Income': [20000, 30000, 50000, 60000, 80000, 90000],
    'CreditScore': [500, 550, 650, 700, 750, 800],
    'Loan': ['No', 'No', 'Yes', 'Yes', 'Yes', 'Yes']
})

X = data[['Income', 'CreditScore']]
y = data['Loan']

model = GaussianNB()
model.fit(X, y)

customer = [[55000, 680]]

print("Loan Prediction:", model.predict(customer)[0])
