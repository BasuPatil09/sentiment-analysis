import pandas as pd

df = pd.read_csv("data/IMDB Dataset.csv")
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
df.to_csv("data/IMDB_clean.csv", index=False)
print("Done!", len(df), "reviews")