import pandas as pd

df = pd.read_csv(
    r"C:/Users\Shreyansh Rai/OneDrive\Desktop/Cleaned Dataset\data/cleaned_paysim_lstm.csv"
)

print("Total rows:", len(df))
print("Is nameOrig present?", 'nameOrig' in df.columns)
print("Unique nameOrig count:", df['nameOrig'].nunique())
print(df['nameOrig'].value_counts().head())
