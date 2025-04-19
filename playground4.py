import pandas as pd

# Load the dataset
file_path = "./finalset.csv"
df_new = pd.read_csv(file_path)

# Display basic information and first few rows
# Step 1: Convert date column to datetime format
df_new['date'] = pd.to_datetime(df_new['date'], format='%Y-%m-%d', errors='coerce')

# Step 2: Handling Missing Values
# Fill missing sleep minutes with median per user
df_new['total_sleep_minutes'] = df_new.groupby('id')['total_sleep_minutes'].transform(lambda x: x.fillna(x.median()))

# Fill missing weight and BMI with median per user, then with overall median if no user data is available
for col in ['weightkg', 'bmi']:
    df_new[col] = df_new.groupby('id')[col].transform(lambda x: x.fillna(x.median()))
    df_new[col].fillna(df_new[col].median(), inplace=True)  # Fill remaining NaNs with global median

# Step 3: Normalize column names
df_new.columns = df_new.columns.str.strip().str.lower().str.replace(' ', '_')

# Step 4: Sort data by id and date
df_new.sort_values(by=['id', 'date'], inplace=True)




processed_file_path = "./Dataset/DS1.csv"
df_new.to_csv(processed_file_path, index=False)
