import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import pandas as pd
from bandit import Bandit


def run_experiment(mu, sigma, N):

    bandit = Bandit(mu, sigma)

    count = np.arange(0, N)
    data = np.empty(N)
    mean = np.empty(N)
    var = np.empty(N)
    svar = np.empty(N)

    # generate a data point and throw it away
    # so that variance can be subsequently computed
    bandit.play()

    for i in range(N):
        data[i] = bandit.play()
        mean[i], var[i], svar[i] = bandit.get_statistics()

    single_bandit_plots(count, data, mean, var, svar)


# plots using seaborn
def single_bandit_plots(count, data, mean, var, svar):
    description = {
        "iteration": count,
        "value": data,
        "estimated mean": mean,
        "estimated sample variance": var,
        "estimated population variance": svar,
    }
    pddata = pd.DataFrame(description)

    plt.rcParams["figure.figsize"] = [15, 5]

    sns.set(style="darkgrid")
    ax1 = sns.scatterplot(x="iteration", y="value", s=5, data=pddata)
    ax1.set(xscale="linear", yscale="linear")
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_title("Data")

    plt.show()

    sns.set(style="darkgrid")
    ax2 = sns.lineplot(x="iteration", y="estimated mean", data=pddata)
    ax2.set(xscale="linear", yscale="linear")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.set_title("Estimated Mean")

    plt.show()

    sns.set(style="darkgrid")
    ax3 = sns.lineplot(x="iteration", y="estimated sample variance", data=pddata)
    ax3.set(xscale="linear", yscale="linear")
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    ax3.set_title("Estimated Sample Variance")

    plt.show()

    sns.set(style="darkgrid")
    ax4 = sns.lineplot(x="iteration", y="estimated population variance", data=pddata)
    ax4.set(xscale="linear", yscale="linear")
    ax4.set_xlabel("")
    ax4.set_ylabel("")
    ax4.set_title("Estimated Population Variance")

    plt.show()

    sns.set(style="darkgrid")
    ax5 = sns.lineplot(x="iteration", y="estimated mean", data=pddata)
    ax5.set(xscale="log", yscale="linear")
    ax5.set_xlabel("")
    ax5.set_ylabel("")
    ax5.set_title("Estimated Mean (log x-scale)")

    plt.show()

    sns.set(style="darkgrid")
    ax6 = sns.lineplot(x="iteration", y="estimated sample variance", data=pddata)
    ax6.set(xscale="log", yscale="linear")
    ax6.set_xlabel("")
    ax6.set_ylabel("")
    ax6.set_title("Estimated Sample Variance (log x-scale)")

    plt.show()

    sns.set(style="darkgrid")
    ax7 = sns.lineplot(x="iteration", y="estimated population variance", data=pddata)
    ax7.set(xscale="log", yscale="linear")
    ax7.set_xlabel("")
    ax7.set_ylabel("")
    ax7.set_title("Estimated Population Variance (log x-scale)")

    plt.show()


def main():

    seed = None
    N = 10000
    mu = 1.0
    sigma = 10.0

    random.seed(seed)
    run_experiment(mu, sigma, N)


if __name__ == "__main__":
    main()
