# %%
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

path = r"G:\Economics\Finance(Prof.Heidari-Aghajanzadeh)\Data\Speculative Beta\\"

# %%
def beta5(g):
    y = "EReturn"
    x = ["MEReturn", "1MEReturn", "2MEReturn", "3MEReturn", "4MEReturn", "5MEReturn"]
    g = g.dropna()
    # Add a constant term like so:
    if len(g) > 30:
        try:
            model = sm.OLS(g[y], sm.add_constant(g[x])).fit()
            beta = (
                model.params[1]
                + model.params[2]
                + model.params[3]
                + model.params[4]
                + model.params[5]
            )
        except:
            print("inner")
            beta = np.nan

    else:
        beta = np.nan

    return beta


def P12Beta(g):
    g["P12Beta"] = "."
    print(g.name)
    ggdata = g.groupby(["Year", "Month"])
    f = ggdata.first().reset_index()["jalaliDate"]
    for i in range(len(f)):
        if i > 11:
            fg = g.loc[g["jalaliDate"] < f.iloc[i]]
            gg = fg.loc[fg["jalaliDate"] >= f.iloc[i - 12]]
            b = beta5(gg)
            g.loc[(g["jalaliDate"] >= f.iloc[i]), "P12Beta"] = b
    return g


def betamean(g):
    return g.P12Beta.mean()


def returnsum(g):
    return g["return"].sum()


def generate(m, path, path2):
    data = pd.read_parquet(path + "SpeculativeBetaData" + str(m) + ".parquet")
    ddd = pd.read_parquet(path + "SpeculativeBeta" + str(m) + ".parquet")
    t = ddd.groupby("Portfolio")[["Beta", "12Return", "Mreturn"]].mean()
    a = (
        data.groupby(["date", "Portfolio"])
        .size()
        .to_frame()
        .reset_index()
        .groupby("Portfolio")[[0]]
        .mean()
        .round()
    )
    t["Size"] = a[0]
    ddd.groupby("Portfolio")[["Beta", "12Return"]].mean().plot.scatter(
        x="Beta", y="12Return"
    )
    plt.grid()
    plt.savefig(path2 + "BetaReturn" + str(m) + ".jpg")
    plt.savefig(path2 + "BetaReturn" + str(m) + ".eps")
    return ddd, t.T


def generateSpeculative(m, path, path2):
    data = pd.read_parquet(
        path + "SpeculativeBetaSpeculativeData" + str(m) + ".parquet"
    )
    ddd2 = pd.read_parquet(path + "SpeculativeBetaSpeculative" + str(m) + ".parquet")
    t2 = ddd2.groupby(["Speculative", "Portfolio"])[
        ["Beta", "12Return", "Mreturn"]
    ].mean()
    t2
    a2 = (
        data.groupby(["date", "Speculative", "Portfolio"])
        .size()
        .to_frame()
        .reset_index()
        .groupby(["Speculative", "Portfolio"])[[0]]
        .mean()
        .round()
    )
    t2["Size"] = a2[0]
    ddd2[ddd2.Speculative == 1].groupby("Portfolio")[
        ["Beta", "12Return"]
    ].mean().plot.scatter(x="Beta", y="12Return")
    plt.grid()
    plt.title("Speculative")
    plt.savefig(path2 + "SpeculativeBetaReturn" + str(m) + ".jpg")
    plt.savefig(path2 + "SpeculativeBetaReturn" + str(m) + ".eps")
    ddd2[ddd2.Speculative == 0].groupby("Portfolio")[
        ["Beta", "12Return"]
    ].mean().plot.scatter(x="Beta", y="12Return")
    plt.grid()
    plt.title("NonSpeculative")
    plt.savefig(path2 + "NonSpeculativeBetaReturn" + str(m) + ".jpg")
    plt.savefig(path2 + "NonSpeculativeBetaReturn" + str(m) + ".eps")
    return ddd2, t2.T


#%%
path2 = (
    r"D:\Dropbox\Finance(Prof.Heidari-Aghajanzadeh)\Project\Speculative Betas\Report\\"
)
result = pd.DataFrame()
for m in [5, 10, 15, 20]:
    print(m)
    ddd, t = generate(m, path, path2)
    aggregate = pd.DataFrame()
    aggregate = aggregate.append(ddd)
    aggregate["Speculative"] = "None"
    ddd, t = generateSpeculative(m, path, path2)
    aggregate = aggregate.append(ddd).reset_index(drop=True)
    aggregate["PNumber"] = m
    result = result.append(aggregate)

result

#%%

# %%
result.to_csv(path + "speclativePortfoDetails.csv", index=False)
# %%
