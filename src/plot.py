import matplotlib.pyplot as plt
import pandas as pd

# Plot Cohn-Kanade predictions
dfCK = pd.read_csv("results12.csv")
plt.plot(dfCK["numClassifiers"], dfCK["accuracy"], 'r.-')

dfJAFFE = pd.read_csv("results3.csv")
plt.plot(dfJAFFE["numClassifiers"], dfJAFFE["accuracy"], 'b.-')


plt.axis([0, 40, 0.4, 0.95])
plt.ylabel("Accuracy")
plt.xlabel("Ensemble size")
plt.savefig("plot.png")