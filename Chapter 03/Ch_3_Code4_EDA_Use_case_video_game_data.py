
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('path_to_your_file/data4_Ch3_vgsales.csv')

# Drop the 'Rank', 'Name', and 'Year' columns
data_dropped = data.drop(columns=['Rank', 'Name', 'Year'])

# Boxplot: Genre vs. Global Sales Distribution
plt.figure(figsize=(14, 8)) 
sns.boxplot(x='Genre', y='Global_Sales', data=data_dropped)
plt.title('Genre vs. Global Sales Distribution')
plt.xlabel('Genre')
plt.ylabel('Global Sales (Millions)')
plt.xticks(rotation=45)
plt.yscale('log')
plt.grid(True, which="both", ls="--")
plt.show()

# Pie Chart: Market Share by Genre
genre_sales = data_dropped.groupby('Genre')['Global_Sales'].sum()
genre_sales_percentage = genre_sales / genre_sales.sum() * 100
plt.figure(figsize=(10, 8))
genre_sales_percentage.plot.pie(autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Market Share by Genre (Percentage of Global Sales)')
plt.ylabel('')
plt.show()

# Pie Chart: Market Share by Top Publishers
top_12_publishers = data_dropped.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False).head(12)
top_publishers_market_share = top_12_publishers / top_12_publishers.sum() * 100
plt.figure(figsize=(10, 8))
top_publishers_market_share.plot.pie(autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Market Share by Top Publishers (Above 200 Million Global Sales)')
plt.ylabel('')
plt.show()

# Barplot: Top 10 Platforms by Total Global Sales
top_10_platforms = data_dropped.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 8))
sns.barplot(x=top_10_platforms.index, y=top_10_platforms.values, palette="viridis")
plt.title('Top 10 Platforms by Total Global Sales')
plt.xlabel('Platform')
plt.ylabel('Global Sales (Millions)')
plt.xticks(rotation=45)
plt.show()

# Overlaid Bar Plot: Regional Sales for Top 10 Platforms
platform_sales_region = data_dropped.groupby('Platform').agg({'NA_Sales': 'sum','EU_Sales': 'sum', 'JP_Sales': 'sum','Other_Sales': 'sum'})
top_platforms_sales_region = platform_sales_region.loc[top_10_platforms.index]
top_platforms_sales_region.plot(kind='bar', figsize=(14, 10), width=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title('Regional Sales for Top 10 Platforms')
plt.xlabel('Platform')
plt.ylabel('Sales (Millions)')
plt.xticks(rotation=45)
plt.legend(title='Region')
plt.show()
