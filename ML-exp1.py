import pandas as pd

data = pd.DataFrame([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
], columns=['Sky', 'AirTemp', 'Humidity',
            'Wind', 'Water', 'Forecast', 'EnjoySport'])

h = ['0'] * 6

for _, row in data.iterrows():
    if row['EnjoySport'] == 'Yes':
        for i in range(6):
            if h[i] == '0':
                h[i] = row.iloc[i]
            elif h[i] != row.iloc[i]:
                h[i] = '?'

print("Most Specific Hypothesis:", h)
