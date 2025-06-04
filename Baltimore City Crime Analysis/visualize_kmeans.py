import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# After clustering
plt.figure()
df = pd.read_csv("kmeans_9.csv")
sns.scatterplot(x=df.y, y=df.x, 
                hue=df.c, 
                palette=sns.color_palette("hls", n_colors=9))
plt.xlabel("latitude")
plt.ylabel("longitude")
plt.title("Baltimore Crime Locations K=9")

plt.savefig("kmeans_9.png")