import pandas as pd

data = pd.read_csv("training.csv")

n = len(data.columns) - 1

S = ['0'] * n
G = [['?'] * n]

for _, row in data.iterrows():
    x = list(row.iloc[:-1])
    label = row.iloc[-1]

    if label == 'Yes':
        G = [g for g in G if all(
            g[i] == '?' or g[i] == x[i]
            for i in range(n))]

        for i in range(n):
            if S[i] == '0':
                S[i] = x[i]
            elif S[i] != x[i]:
                S[i] = '?'

    else:
        new_G = []

        for g in G:
            for i in range(n):
                if S[i] != '?' and S[i] != '0' and g[i] == '?':
                    h = g.copy()
                    h[i] = S[i]
                    if all(
                        h[j] == '?' or h[j] == x[j]
                        for j in range(n)
                    ):
                        continue
                    new_G.append(h)

        G = new_G

print("Specific Boundary S:", S)
print("General Boundary G:", G)
