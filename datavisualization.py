import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
tips = sns.load_dataset('tips')  # Built-in dataset in Seaborn
print(tips)
# Line plot
sns.lineplot(x='size', y='total_bill', data=tips)
plt.xlabel("no_of_people")
plt.ylabel("Total_bill")
plt.xticks(x='size')
plt.grid(which='major',axis='y')
plt.title("Seaborn Line Plot")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plot
sns.scatterplot(x='total_bill', y='tip', data=tips, hue='sex', size='size')
plt.title("Seaborn Scatter Plot")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Box plot
sns.boxplot(x='day', y='total_bill', data=tips)
plt.title("total bill by each day")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# print(tips.describe())

# Correlation heatmap with numeric data only
numeric_tips = tips.select_dtypes(include='number')  # Select only numeric columns
corr = numeric_tips.corr()
print(corr) 
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
# evenly sampled time at 200ms intervals

t = np.arange(0., 5., 0.2)
# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t,t**2.5,'yd')
plt.show()

import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [10, 100, 15, 0, 99]

# Creating a line plot
# plt.plot(x, y, color='blue', linestyle='dashdot', marker='D')
plt.plot(x,y,'g--',marker='D')
plt.title("Matches vs Runs scored ")
plt.xlabel("Match Number")
plt.ylabel("Score")
plt.xticks(x)
plt.grid()
# plt.grid(which='minor',axis='y')
# plt.savefig("Lineplot_On_Runs.png", dpi=1000, bbox_inches='tight') #72 low # 150 standard # 300 high # 1000 ultra high
plt.show()

import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [4, 2, 25, 9, 16]

# Scatter plot
plt.scatter(x, y, color='red', marker='^',linewidths=1 )
plt.title("Scatter Plot Example")
plt.xlabel("Match Number")
plt.ylabel("Score")
# plt.grid()
plt.show()

import matplotlib.pyplot as plt

# Sample data
categories = ['A', 'B', 'C', 'D']
sales_data_in_thosuands = [5, 7, 3, 8]

# Bar plot

plt.bar(categories, sales_data_in_thosuands, color='yellow')
plt.title("Bar Plot between category and sales ")
plt.xlabel("Categories")
plt.ylabel("Sales in thousands ")
plt.show()

import matplotlib.pyplot as plt
import numpy as np 

# Random data
data = np.random.randn(10)

# Histogram
plt.hist(data,  color='green', alpha=1, bins=10 ) #, alpha=0.7
plt.title("Histogram Example")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()





