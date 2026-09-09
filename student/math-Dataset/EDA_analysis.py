import pandas as pd

# 1. Load the dataset (using semicolon separator)
df = pd.read_csv('student-mat.csv', sep=';')

print("=" * 40)
print("     DATA INTEGRITY & STATS REPORT     ")
print("=" * 40)

# 2. Check missing values
missing_count = df.isnull().sum().sum()
print(f"Total Missing Values: {missing_count}")

# 3. Check duplicate rows
duplicate_count = df.duplicated().sum()
print(f"Total Duplicate Rows: {duplicate_count}")

# 4. Check out-of-range or extreme values (Weird values)
print("\n--- Outlier / Extreme Value Check ---")
print(f"Absences range: Min = {df['absences'].min()}, Max = {df['absences'].max()}")
print(f"Final Grade (G3) range: Min = {df['G3'].min()}, Max = {df['G3'].max()}")

# 5. Get basic descriptive statistics for numerical columns
print("\n--- Summary Statistics (Grades & Study Time) ---")
print(df[['studytime', 'failures', 'absences', 'G1', 'G2', 'G3']].describe())

print("=" * 40)