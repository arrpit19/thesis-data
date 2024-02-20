import pandas as pd
import seaborn as sns



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming dfmae is your DataFrame
dfmaemulti = pd.read_csv("C:/Users/Arpit/Desktop/Maemulti.csv")

print(dfmaemulti)

# Set seaborn style for better aesthetics
sns.set(style="whitegrid", palette="pastel")

# Create the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=dfmaemulti, width=0.6, palette="Set3")

# Customize the plot
plt.title("Box Plot showing mean absolute error for Spatial data", fontsize=16)
plt.xlabel("Models", fontsize=14)
plt.ylabel("Values", fontsize=14)

# Save the plot as an image file (e.g., PNG)
plt.savefig("boxplot_mae.png")

# Show the plot
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming dfmae is your DataFrame
dfmaeaug = pd.read_csv("C:/Users/Arpit/maeaug.csv")

# Set seaborn style for better aesthetics
sns.set(style="whitegrid", palette="pastel")

# Create the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=dfmaeaug, width=0.6, palette="Set3")

# Customize the plot
plt.title("Box Plot showing mean absolute error for temporal augmented data", fontsize=16)
plt.xlabel("Columns", fontsize=14)
plt.ylabel("Values", fontsize=14)

# Save the plot as an image file (e.g., PNG)
plt.savefig("boxplot_maeaug.png")

# Show the plot
plt.show()
