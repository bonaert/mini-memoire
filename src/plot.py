import matplotlib.pyplot as plt
import pandas as pd

# Plot Cohn-Kanade predictions
dir = "remoteResults/"
PDtype = "PD"
dfCK = pd.read_csv(dir + "resultsCK" + PDtype + "0.csv")
plt.plot(dfCK["numClassifiers"], dfCK["accuracy"], 'r.-')

dfJAFFE = pd.read_csv(dir + "resultsJaffe" + PDtype + "0.csv")
plt.plot(dfJAFFE["numClassifiers"], dfJAFFE["accuracy"], 'b.-')


plt.axis([0, 85, 0.1, 0.95])
plt.ylabel("Accuracy")
plt.xlabel("Ensemble size")
plt.savefig("plot" + PDtype + ".png")
