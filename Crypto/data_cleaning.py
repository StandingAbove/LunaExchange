import pandas as pd

# 1. Load the data
# We use 'on_bad_lines' just in case there are formatting glitches
df = pd.read_csv('/Users/samgeng14/PycharmProjects/LunaExchange/LDA_bulk_data.csv')

# 2. Get rid of the "spaces" (Empty Rows)
# This removes rows where all values (or most values) are missing
df = df.dropna(how='all') 

# 3. Filter for Bitcoin, Ethereum, and Mining columns
# We want: Date, anything with "BTC", anything with "ETH", and "COST_TO_MINE"
columns_to_keep = [col for col in df.columns if 
                   'BTC' in col or 
                   'ETH' in col or 
                   'Date' in col or 
                   'COST_TO_MINE' in col or 
                   'BTC_REWARD' in col]

df_clean = df[columns_to_keep]

# 4. Fix the Date format (Optional but recommended)
# Your data shows dates like 20080101. This converts them to 2008-01-01
df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%Y%m%d')

# 5. Save the cleaned file
df_clean.to_csv('cleaned_crypto_data.csv', index=False)

print("Cleanup complete! Kept columns:", df_clean.columns.tolist())