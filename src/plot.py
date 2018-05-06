import matplotlib.pyplot as plt
import pandas as pd


# Plot Cohn-Kanade predictions
def makePlot(PDtype, CKnum, JaffeNum):
    dir = "remoteResults/"
    dfCK = pd.read_csv(dir + "resultsCK" + PDtype + str(CKnum) + ".csv")
    plt.plot(dfCK["numClassifiers"], dfCK["accuracy"], 'r.-')

    dfJAFFE = pd.read_csv(dir + "resultsJaffe" + PDtype + str(JaffeNum) + ".csv")
    plt.plot(dfJAFFE["numClassifiers"], dfJAFFE["accuracy"], 'b.-')

    plt.legend(['CK+ dataset', 'JAFFE dataset'])

    plt.axis([0, 85, 0, 0.95])
    plt.ylabel("Accuracy")
    plt.xlabel("Ensemble size")
    plt.savefig("figures/plot" + PDtype + ".png")
    plt.close()


makePlot("PI", 1, 1)
makePlot("PD", 0, 0)
