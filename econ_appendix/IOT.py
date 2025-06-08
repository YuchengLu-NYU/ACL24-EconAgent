#%%
import pandas as pd

# Read the ZAF2020ttl.csv file into a DataFrame, using first column as index
df = pd.read_csv('ZAF2020ttl.csv', index_col=0)
#%%
# Keep only the household final consumption expenditure column
df = df[['HFCE']]


# %%
# Drop rows after TTL_R
df = df.loc[:'TTL_R']


# %%
# Calculate and display distribution statistics of HFCE
print("Distribution statistics of Household Final Consumption Expenditure:")
print("\nDescriptive Statistics:")
print(df['HFCE'].describe())

# Create a boxplot visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
df.boxplot(column='HFCE')
plt.title('Distribution of Household Final Consumption Expenditure')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.show()

# Create a histogram
plt.figure(figsize=(10,6))
df['HFCE'].hist(bins=20)
plt.title('Histogram of Household Final Consumption Expenditure')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# %%
# Get the 65th percentile value
percentile_65 = df['HFCE'].quantile(0.65)

# Filter rows above 75th percentile and sort in descending order
top_spenders = df[df['HFCE'] > percentile_65].sort_values('HFCE', ascending=False)

print("\nTop 35% of household expenditure categories (above 65th percentile):")
print("65th percentile value:", percentile_65)
print("\nCategories and their expenditure values:")
print(top_spenders)

# %%
