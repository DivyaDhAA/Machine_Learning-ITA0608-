import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.DataFrame({
    'Income': [20000, 30000, 50000, 80000, 90000, 25000],
    'Debt': [15000, 20000, 10000, 5000, 8000, 22000],
    'CreditScore': ['Low', 'Low', 'Medium',
                    'High', 'High', 'Low']
})

X = data[['Income', 'Debt']]
y = data['CreditScore']

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

new_customer = [[60000, 12000]]

print("Credit Score:",
      model.predict(new_customer)[0])
