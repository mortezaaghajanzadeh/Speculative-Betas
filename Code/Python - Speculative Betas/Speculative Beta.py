# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython
from numpy.core.arrayprint import _make_options_dict

# %%
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt


# %%
path = r"G:\Economics\Finance(Prof.Heidari-Aghajanzadeh)\Data"
path = r"C:\Users\RA\Desktop\RA_Aghajanzadeh\Data\\"
n1 = path + "\Stocks_Prices_1400-02-07" + ".parquet"
df1 = pd.read_parquet(n1)
df1 = df1[~((df1.title.str.startswith("ح")) & (df1.name.str.endswith("ح")))]
df1 = df1[~(df1.name.str.endswith("پذيره"))]
df1 = df1[
    ["jalaliDate", "date", "name", "group_name", "close_price", "volume", "quantity"]
]


# %%
def vv(row):
    X = row.split("-")
    return int(X[0] + X[1] + X[2])


def DriveYearMonthDay(d):
    d["jalaliDate"] = d["jalaliDate"].astype(str)
    d["Year"] = d["jalaliDate"].str[0:4]
    d["Month"] = d["jalaliDate"].str[4:6]
    d["Day"] = d["jalaliDate"].str[6:8]
    d["jalaliDate"] = d["jalaliDate"].astype(int)
    return d


df1["jalaliDate"] = df1["jalaliDate"].apply(vv)

df = df1


# %%
cols = ["close_price", "volume", "quantity"]
for col in cols:
    df[col] = df[col].astype(float)
symbols = [
    "سپرده",
    "هما",
    "وهنر-پذيره",
    "نکالا",
    "تکالا",
    "اکالا",
    "توسعه گردشگری ",
    "وآفر",
    "ودانا",
    "نشار",
    "نبورس",
    "چبسپا",
    "بدکو",
    "چکارم",
    "تراک",
    "کباده",
    "فبستم",
    "تولیددارو",
    "قیستو",
    "خلیبل",
    "پشاهن",
    "قاروم",
    "هوایی سامان",
    "کورز",
    "شلیا",
    "دتهران",
    "نگین",
    "کایتا",
    "غیوان",
    "تفیرو",
    "سپرمی",
    "بتک",
]
df = df.drop(df[df["name"].isin(symbols)].index)
df = df.drop(df[df.group_name == "صندوق سرمایه گذاری قابل معامله"].index)
df = df.drop(df[df.group_name == "فعاليتهاي كمكي به نهادهاي مالي واسط"].index)
df = df.drop(df[(df.name == "اتکای") & (df.close_price == 1000)].index)
df = df.drop_duplicates()
df = (
    df.drop(df.loc[(df["volume"] == 0)].index)
    .sort_values(by=["name", "jalaliDate"])
    .drop(columns=["quantity"])
)
df = DriveYearMonthDay(df)


# %%
path2 = r"D:\TseClient\Data adjusted"
path2 = r"C:\Users\RA\Desktop\RA_Aghajanzadeh\Data\\"
index = pd.read_excel(path2 + "\IRX6XTPI0009.xls")[["<COl14>", "<CLOSE>"]].rename(
    columns={"<COl14>": "jalaliDate", "<CLOSE>": "Index"}
)
index["Market_return"] = index["Index"].pct_change(periods=1) * 100
index = DriveYearMonthDay(index)


# %%
n = path + "\RiskFree.xlsx"
df3 = pd.read_excel(n)
df3 = df3.rename(columns={"Unnamed: 2": "Year"})
df3["YM"] = df3["YM"].astype(str)
df3["YM"] = df3["YM"] + "00"
df3["YM"] = df3["YM"].astype(int)
df4 = index
df4["MRiskFree"] = np.nan
df4["WRiskFree"] = np.nan
df4["DRiskFree"] = np.nan
df4["jalaliDate"] = df4["jalaliDate"].astype(int)
for i in df3.YM:
    df4.loc[df4.jalaliDate >= i, "WRiskFree"] = (
        df3.loc[df3["YM"] == i].iloc[0, 1] / 12 / 52
    )
    df4.loc[df4.jalaliDate >= i, "MRiskFree"] = df3.loc[df3["YM"] == i].iloc[0, 1] / 12
df4["DRiskFree"] = df4["WRiskFree"] / 7
df4.head()


# %%
for i in range(5):
    col = str(i + 1) + "Market_return"
    df4[col] = df4["Market_return"].shift(i + 1)
df4.head()

index = df4

index.head()

# %%
gg = df.groupby(["name"])
df["close_price"] = df.close_price.astype(float)
df["return"] = df["close_price"].pct_change(periods=1) * 100
df.head()


# %%
data = df.merge(index, on=["jalaliDate", "Year", "Month", "Day"]).sort_values(
    by=["name", "jalaliDate"]
)
data.head()


# %%
data["EReturn"] = data["return"] - data["DRiskFree"]
data["MEReturn"] = data["Market_return"] - data["DRiskFree"]
for i in range(5):
    col1 = str(i + 1) + "Market_return"
    col2 = str(i + 1) + "MEReturn"
    data[col2] = data[col1] - data["DRiskFree"]

data.head()

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
    g["std"] = "."
    print(g.name)
    ggdata = g.groupby(["Year", "Month"])
    f = ggdata.first().reset_index()["jalaliDate"]
    for i in range(len(f)):
        if i > 11:
            fg = g.loc[g["jalaliDate"] < f.iloc[i]]
            gg = fg.loc[fg["jalaliDate"] >= f.iloc[i - 12]]
            b = beta5(gg)
            g.loc[(g["jalaliDate"] >= f.iloc[i]), "P12Beta"] = b
            g.loc[(g["jalaliDate"] >= f.iloc[i]), "std"] = gg["return"].std()
    return g


gdata = data.groupby(["name"])
bdata = gdata.apply(P12Beta)

path = r"G:\Economics\Finance(Prof.Heidari-Aghajanzadeh)\Data\Speculative Beta\\"
path = r"C:\Users\RA\Desktop\RA_Aghajanzadeh\Data\\"
bdata.to_csv(path + "FirstRegress" + ".csv")


#%%

#%%
def BeforePortfo():
    path = r"G:\Economics\Finance(Prof.Heidari-Aghajanzadeh)\Data\Speculative Beta\\"
    path = r"C:\Users\RA\Desktop\RA_Aghajanzadeh\Data\\"
    n = path + "FirstRegress" + ".csv"
    bdata = pd.read_csv(n)
    bdata = bdata[~(bdata.name.str.endswith("پذيره"))]
    data = bdata[bdata.P12Beta != "."]
    data = data[~data.P12Beta.isnull()]
    len(set(data.name))
    data["P12Beta"] = data.P12Beta.astype(float)
    data = data[(data.Year < 1399) & (data.Year >= 1393)]
    data = AddMarketCap(path, data)
    return data


def AddMarketCap(path, data):
    # path = r"G:\Economics\Finance(Prof.Heidari-Aghajanzadeh)\Data\\"
    sdf = pd.read_csv(path + "SymbolShrout.csv")
    mapdict = dict(zip(sdf.set_index(["symbol", "date"]).index, sdf.shrout))
    data["shrout"] = data.set_index(["name", "date"]).index.map(mapdict)
    data["shrout"] = data.groupby("name")["shrout"].fillna(method="ffill")
    data.isnull().sum()
    data = data[~data.shrout.isnull()]
    data["marketCap"] = data.shrout * data.close_price
    return data


def DataPortfo(data, m):
    gg = data.groupby(["Year", "Month"])

    tempt1 = (
        gg.apply(portfo, m=m)
        .reset_index(drop=True)
        .set_index(["Year", "Month", "name"])
    )
    mapdict = dict(zip(tempt1.index, tempt1.Portfolio))
    data["Portfolio"] = data.set_index(["Year", "Month", "name"]).index.map(mapdict)
    return data


def portfo(g, m):
    # print(g.name)
    stocks = g.drop_duplicates(subset=["name"])
    stocks["Portfolio"] = 0
    for i in range(m):
        th = stocks.P12Beta.quantile(i * 1 / m)
        stocks.loc[g.P12Beta >= th, "Portfolio"] = i + 1
    stocks = stocks.sort_values(by=["Portfolio"])
    return stocks


def portfolioReturn(g):
    # print(g.name)
    t = g[
        [
            "jalaliDate",
            "Year",
            "Month",
            "Day",
            "Index",
            "Market_return",
            "DRiskFree",
            "MEReturn",
            "1MEReturn",
            "2MEReturn",
            "3MEReturn",
            "4MEReturn",
            "5MEReturn",
        ]
    ].drop_duplicates()
    g["BetaWeighted"] = g["P12Beta"] / g["P12Beta"].sum()
    g["weight"] = g["marketCap"] / g["marketCap"].sum()
    t["equallWeightedReturn"] = g["return"].mean()
    g["betaWeightedreturn"] = g["BetaWeighted"] * g["return"]
    t["betaWeightedreturn"] = g["betaWeightedreturn"].sum()
    g["return"] = g["weight"] * g["return"]
    t["return"] = g["return"].sum()
    t["EReturn"] = t["return"] - t["DRiskFree"]
    return t


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
    # print(g.name)
    ggdata = g.groupby(["Year", "Month"])
    f = ggdata.first().reset_index()["jalaliDate"]
    for i in range(len(f)):
        if i > 11:
            fg = g.loc[g["jalaliDate"] < f.iloc[i]]
            gg = fg.loc[fg["jalaliDate"] >= f.iloc[i - 12]]
            b = beta5(gg)
            g.loc[(g["jalaliDate"] >= f.iloc[i]), "P12Beta"] = b
    return g


def aggregate(gg):
    ddd = gg.first()
    ddd["Mreturn"] = gg["return"].sum()
    ddd["MMarket_return"] = gg["Market_return"].sum()
    ddd["Beta"] = gg.apply(betamean)
    ddd = ddd.reset_index()
    return ddd


def betamean(g):
    return g.P12Beta.mean()


def yearCal(ddd):
    ddd["12Return"] = np.nan
    ddd["12MReturn"] = np.nan
    gg = ddd.groupby("Portfolio")
    for i in gg.groups:
        g = gg.get_group(i)
        ddd.loc[ddd.Portfolio == i, "12Return"] = g.Mreturn.rolling(12).sum().shift(-12)
        ddd.loc[ddd.Portfolio == i, "12MReturn"] = (
            g.MMarket_return.rolling(12).sum().shift(-12)
        )
    return ddd


def DataPortfoSpeculative(data, m):
    gg = data.groupby(["Year", "Month"])
    tempt1 = (
        gg.apply(Speculativeportfo, m=m)
        .reset_index(drop=True)
        .set_index(["Year", "Month", "name"])
    )
    mapdict = dict(zip(tempt1.index, tempt1.Portfolio))
    data["Portfolio"] = data.set_index(["Year", "Month", "name"]).index.map(mapdict)
    mapdict = dict(zip(tempt1.index, tempt1.Speculative))
    data["Speculative"] = data.set_index(["Year", "Month", "name"]).index.map(mapdict)
    return data


def Speculativeportfo(g, m):
    # print(g.name)
    stocks = g.drop_duplicates(subset=["name"])
    stocks["Portfolio"] = 0
    stocks["Speculative"] = 0
    stocks.loc[stocks.ratio > stocks.ratio.median(), "Speculative"] = 1
    result = pd.DataFrame()
    for j in [0, 1]:
        t = stocks[stocks.Speculative == j]
        for i in range(m):
            th = t.P12Beta.quantile(i * 1 / m)
            t.loc[t.P12Beta >= th, "Portfolio"] = i + 1
        result = result.append(t)
    return result


def yearCalSpeculative(ddd):
    data = pd.DataFrame()
    ddd["12Return"] = np.nan
    ddd["12MReturn"] = np.nan
    gg = ddd.groupby(["Speculative", "Portfolio"])
    for i in gg.groups:
        g = gg.get_group(i)
        g["12Return"] = g.Mreturn.rolling(12).sum().shift(-12)
        g["12MReturn"] = g.MMarket_return.rolling(12).sum().shift(-12)
        data = data.append(g)
    return data


def summary(ddd, path2):
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
    t.T
    ddd.groupby("Portfolio")[["Beta", "12Return"]].mean().plot.scatter(
        x="Beta", y="12Return"
    )
    plt.grid()
    plt.savefig(path2 + "BetaReturn.jpg")
    plt.savefig(path2 + "BetaReturn.eps")
    return t.T


def summarySpeculative(ddd, path2):
    t = ddd.groupby(["Speculative", "Portfolio"])[
        ["Beta", "12Return", "Mreturn"]
    ].mean()
    a = (
        data.groupby(["date", "Speculative", "Portfolio"])
        .size()
        .to_frame()
        .reset_index()
        .groupby(["Speculative", "Portfolio"])[[0]]
        .mean()
        .round()
    )
    t["Size"] = a[0]
    t.T
    ddd[ddd.Speculative == 1].groupby("Portfolio")[
        ["Beta", "12Return"]
    ].mean().plot.scatter(x="Beta", y="12Return")
    plt.grid()
    plt.title("Speculative")
    plt.savefig(path2 + "SpeculativeBetaReturn.jpg")
    plt.savefig(path2 + "SpeculativeBetaReturn.eps")
    ddd[ddd.Speculative == 0].groupby("Portfolio")[
        ["Beta", "12Return"]
    ].mean().plot.scatter(x="Beta", y="12Return")
    plt.grid()
    plt.title("NonSpeculative")
    plt.savefig(path2 + "NonSpeculativeBetaReturn.jpg")
    plt.savefig(path2 + "NonSpeculativeBetaReturn.eps")
    return t.T


#%%
m = 5
path2 = r"C:\Users\RA\Desktop\RA_Aghajanzadeh\Last Code\Speculative beta\\"
path = r"G:\Economics\Finance(Prof.Heidari-Aghajanzadeh)\Data\Speculative Beta\\"
path = r"C:\Users\RA\Desktop\RA_Aghajanzadeh\Data\\"
odata = BeforePortfo()
#%%
for m in [5, 10, 15, 20]:
    print("--------------------------------------------------")
    print("------         NoSpeculative" + str(m) + "             -------")
    print("--------------------------------------------------")
    data = pd.DataFrame()
    data = data.append(odata)
    data = DataPortfo(data, m)
    gg = data.groupby(["Portfolio", "date"])
    print("start buliding Portfolio")
    tempt = gg.apply(portfolioReturn).reset_index()
    gg = tempt.groupby(["Portfolio"])
    print("Portfolio Betas")
    pdata = gg.apply(P12Beta)
    pdata = pdata[pdata.P12Beta != "."]
    gg = pdata.groupby(["Portfolio", "Year", "Month"])
    print("Start aggregating")
    ddd1 = aggregate(gg)
    ddd1 = ddd1[["Portfolio", "Year", "Month", "Mreturn", "Beta", "MMarket_return"]]
    ddd1 = yearCal(ddd1)
    print("Summarize")
    # t = summary(ddd1,path2)
    n1 = path + "SpeculativeBetaData" + str(m) + ".parquet"
    data.to_parquet(n1, index=False)
    print("--------------------------------------------------")
    print("------          Speculative" + str(m) + "              -------")
    print("--------------------------------------------------")
    data = pd.DataFrame()
    data = data.append(odata)
    data["std"] = data["std"].astype(float)
    data = data[data["std"] > 0.0]
    data["ratio"] = data["P12Beta"] / data["std"] / data["std"]
    data = DataPortfoSpeculative(data, m)
    gg = data.groupby(["Portfolio", "Speculative", "date"])
    print("start buliding Portfolio")
    tempt2 = gg.apply(portfolioReturn).reset_index()
    gg = tempt2.groupby(["Speculative", "Portfolio"])
    print("Portfolio Betas")
    pdata = gg.apply(P12Beta)
    pdata = pdata[pdata.P12Beta != "."]
    gg = pdata.groupby(["Speculative", "Portfolio", "Year", "Month"])
    print("Start aggregating")
    ddd2 = aggregate(gg)
    ddd2 = ddd2[
        [
            "Speculative",
            "Portfolio",
            "Year",
            "Month",
            "Mreturn",
            "Beta",
            "MMarket_return",
        ]
    ]
    ddd2 = yearCalSpeculative(ddd2)
    print("Summarize")
    # t = summarySpeculative(ddd2,path2)
    n1 = path + "SpeculativeBeta" + str(m) + ".parquet"
    ddd1.to_csv(n1, index=False)
    n1 = path + "SpeculativeBetaSpeculative" + str(m) + ".parquet"
    ddd2.to_parquet(n1, index=False)
    n1 = path + "SpeculativeBetaSpeculativeData" + str(m) + ".parquet"
    data.to_parquet(n1, index=False)
#%%
