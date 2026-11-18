import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Rainy', 'Rainy',
                'Overcast', 'Overcast', 'Sunny', 'Rainy'],
    'Temperature': ['Hot', 'Cool', 'Hot', 'Cool',
                    'Hot', 'Cool', 'Hot', 'Cool'],
    'Play': ['No', 'Yes', 'Yes', 'Yes',
             'Yes', 'Yes', 'No', 'Yes']
})

X = pd.get_dummies(data[['Outlook', 'Temperature']])
y = data['Play']

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

sample = pd.DataFrame({
    'Outlook': ['Sunny'],
    'Temperature': ['Cool']
})

sample = pd.get_dummies(sample).reindex(
    columns=X.columns, fill_value=0
)

print("Prediction:", model.predict(sample)[0])

plot_tree(model, feature_names=X.columns,
          class_names=model.classes_, filled=True)
plt.show()
