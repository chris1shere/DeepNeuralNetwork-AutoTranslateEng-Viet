import pandas as pd

# Load the data into a pandas DataFrame
data = pd.read_csv('C:/Users/chris/Desktop/PYTHON/English-Viet Translation/data.csv', header=None, names=['english', 'vietnamese'])

# Print the first few rows of the data
print(data.head())
