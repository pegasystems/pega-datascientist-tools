import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import timedelta
import itertools


def initial_prep(df, referenceTime="pxOutcomeTime"):
    for i in df.columns:
        if "time" in str(i).lower():
            df[i] = pd.to_datetime(df[i])
    df["Date"] = df[referenceTime].dt.strftime("%Y-%m-%d")
    df["Date_date"] = pd.to_datetime(df["Date"])
    df["WeekOfYear"] = [i.weekofyear for i in df["Date_date"]]
    df["Week"] = df["WeekOfYear"] - df["WeekOfYear"].min() + 1
    return df.drop("Date_date", axis=1)


def detect_outlier(df, col):
    data = df[col]
    data = sorted(data)
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)

    return df[(df[col] < lower_range) | (df[col] > upper_range)].reset_index(drop=True)


def get_all_times_inds(df):
    idx_time = (
        pd.DataFrame(
            [
                i.strftime("%Y-%m-%d")
                for i in pd.period_range(min(df["Date"]), max(df["Date"]))
            ]
        )
        .rename(columns={0: "Date"})
        .sort_values("Date")
    )
    return idx_time


def get_total_outcome(df, outcome, rollup):  # pragma: no cover
    if type(outcome) == list:
        _df_all = pd.DataFrame()
        for i in outcome:
            _df = (
                df[df["pyOutcome"] == i]
                .groupby(rollup)
                .count()[["pxInteractionID"]]
                .rename(columns={"pxInteractionID": "Count: " + i})
            )
            _df_all = pd.concat([_df_all, _df], axis=1)
    else:
        _df_all = (
            df[df["pyOutcome"] == outcome]
            .groupby(rollup)
            .count()[["pxInteractionID"]]
            .rename(columns={"pxInteractionID": "Count: " + outcome})
        )
    return _df_all


def get_accept_rate(df, pos, neg, rollup):
    if type(pos) != list:
        pos = pos.split()
    if type(neg) != list:
        neg_t = []
        neg_t.append(neg)
        neg = neg_t
    total = [pos, neg]
    total = list(itertools.chain.from_iterable(total))

    _df = (
        df[df["pyOutcome"].isin(total)]
        .groupby(rollup)
        .count()[["pxInteractionID"]]
        .reset_index()
        .rename(columns={"pxInteractionID": "Total"})
    )
    _df = _df.merge(
        df[df["pyOutcome"].isin(pos)]
        .groupby(rollup)
        .count()[["pxInteractionID"]]
        .reset_index()
        .rename(columns={"pxInteractionID": "Accepted"}),
        on=rollup,
        how="left",
    ).fillna(0)
    _df["Accept Rate (%)"] = _df["Accepted"] * 100.0 / _df["Total"]
    return _df


def get_accept_rate_time(df, pos, neg, level, **kwargs):
    rollup = [level]
    hue = []
    if "hue" in kwargs.keys():
        if type(kwargs["hue"]) == list:
            for i in kwargs["hue"]:
                rollup.append(i)
            hue = kwargs["hue"]
        else:
            rollup.append(kwargs["hue"])
            hue.append(kwargs["hue"])
    else:
        rollup = level
    return (get_accept_rate(df, pos, neg, rollup), rollup, hue)


def plot_daily_accept_rate(df, pos, neg, **kwargs):
    _df, rollup, hue = get_accept_rate_time(df, pos, neg, "Date", **kwargs)
    if len(hue) > 0:
        _df["hue"] = _df[hue].agg("__".join, axis=1)
        kwargs["hue"] = "hue"
    if "allTime" in kwargs.keys():
        inds_df = get_all_times_inds(df)
        _df = get_allDays_df(_df, inds_df, hue)

    get_daily_graph(_df.sort_values("Date"), "Date", "Accept Rate (%)", **kwargs)


def plot_weekly_accept_rate(df, pos, neg, **kwargs):
    _df, rollup, hue = get_accept_rate_time(df, pos, neg, "Week", **kwargs)
    if len(hue) > 0:
        _df["hue"] = _df[hue].agg("__".join, axis=1)
        kwargs["hue"] = "hue"

    get_daily_graph(_df.sort_values("Week"), "Week", "Accept Rate (%)", **kwargs)


def plot_daily_cumulative_accept_rate(df, pos, neg, **kwargs):
    _df, rollup, hue = get_accept_rate_time(df, pos, neg, "Date", **kwargs)

    if "hue" in kwargs.keys():
        _df["Total_cum"] = _df.groupby(hue)["Total"].apply(lambda x: x.cumsum())
        _df["Accepted_cum"] = _df.groupby(hue)["Accepted"].apply(lambda x: x.cumsum())
        _df["hue"] = _df[hue].agg("__".join, axis=1)
        kwargs["hue"] = "hue"
    else:
        _df["Total_cum"] = _df["Total"].cumsum()
        _df["Accepted_cum"] = _df["Accepted"].cumsum()
    _df["Cumulative Accept Rate (%)"] = _df["Accepted_cum"] * 100 / _df["Total_cum"]

    if "allTime" in kwargs.keys():
        inds_df = get_all_times_inds(df)
        _df = get_allDays_df(_df, inds_df, hue)

    get_daily_graph(
        _df.sort_values("Date"), x="Date", y="Cumulative Accept Rate (%)", **kwargs
    )


def get_daily_graph(df, x, y, **kwargs):
    if "figsize" in kwargs.keys():  # pragma: no cover
        figsize = kwargs["figsize"]
    else:
        figsize = (12, 5)
    fig, ax = plt.subplots(figsize=figsize)
    if "hue" in kwargs.keys():
        for huevalue in df[kwargs["hue"]].unique():
            _df = (
                df[df[kwargs["hue"]] == huevalue]
                .merge(df[[x]], how="outer")
                .sort_values(x)
            )
            ax.plot(_df[x].tolist(), _df[y].tolist(), label=huevalue)
        plt.legend()
    else:
        ax.plot(df.sort_values(x)[x].tolist(), df.sort_values(x)[y].tolist())
    if "showOutlier" in kwargs.keys():
        outlier_df = detect_outlier(df, y)
        ax.scatter(
            outlier_df[x],
            outlier_df[y],
            marker="o",
            color="r",
            s=100,
            label="Outlier",
            zorder=3,
        )
        if outlier_df.shape[0] > 0:  # pragma: no cover
            plt.legend()
    if "ylabel" in kwargs.keys():
        ax.set_ylabel(kwargs["ylabel"], fontsize=13)
    else:
        ax.set_ylabel(y, fontsize=13)
    ax.set_xlabel(x, fontsize=13)
    for i in ax.get_xmajorticklabels():
        i.set_rotation(90)
    if "ylim" in kwargs.keys():
        ax.set_ylim(kwargs["ylim"])
    if "shrinkTicks" in kwargs.keys():
        for label in ax.xaxis.get_ticklabels()[::2]:
            if not label == ax.xaxis.get_ticklabels()[-1]:
                label.set_visible(False)
    if "title" in kwargs.keys():
        ax.set_title(kwargs["title"])


def plot_outcome_count_time(df, outcome, time, **kwargs):
    if time.lower() == "daily":  # pragma: no cover
        gra = "Date"
    elif time.lower() == "weekly":
        gra = "Week"
    else:
        print(
            'Please define the time granularity parameter. Either use "daily" or "weekly"'
        )
        print("The following graph shows daily view")
        gra = "Date"
    _df = df[df["pyOutcome"] == outcome]
    rollup = [gra]
    hue = []
    if "hue" in kwargs.keys():
        if type(kwargs["hue"]) == list:
            for i in kwargs["hue"]:
                rollup.append(i)
            hue = kwargs["hue"]
        else:
            rollup.append(kwargs["hue"])
            hue.append(kwargs["hue"])
    _df = _df.groupby(rollup).count().reset_index()
    if len(hue) > 0:
        _df["hue"] = _df[hue].agg("__".join, axis=1)
        kwargs["hue"] = "hue"

    if gra == "Date":
        if "allTime" in kwargs.keys():
            inds_df = get_all_times_inds(df)
            _df = get_allDays_df(_df, inds_df, hue)
    get_daily_graph(
        _df.rename(columns={"pxInteractionID": "Count"}).reset_index(),
        gra,
        "Count",
        **kwargs
    )


def get_allDays_df(_df, inds_df, hue):
    if len(hue) > 0:
        _df = (
            pd.concat(
                [
                    pd.DataFrame(
                        [[_df["hue"].unique()] for i in range(len(inds_df.index))]
                    ).rename(columns={0: "hue"}),
                    inds_df,
                ],
                axis=1,
            )
            .explode("hue")
            .merge(_df, on=["Date", "hue"], how="outer")
        )
    else:
        _df = inds_df.merge(_df, how="outer")
    return _df


def get_total_outcome_share_per_level(df, outcome, level):
    _df = (
        df[df["pyOutcome"] == outcome]
        .groupby(level)
        .count()[["pxInteractionID"]]
        .rename(columns={"pxInteractionID": "Count"})
        .reset_index()
    )
    _df["Total"] = _df["Count"].sum()
    _df[outcome + " Share (%)"] = _df["Count"] * 100 / _df["Total"]
    return _df


def plot_outcome_share_graph(df, outcome, level, hue=None):
    fig, ax = plt.subplots(figsize=(14, 4))
    _df = get_total_outcome_share_per_level(df, outcome, level)
    if hue:
        _df = _df.merge(df[[level, hue]].drop_duplicates(), on=level)
    sort = _df.sort_values(outcome + " Share (%)", ascending=False)[level].tolist()
    sns.barplot(x=level, y=outcome + " Share (%)", data=_df, order=sort, hue=hue)
    for x in ax.get_xmajorticklabels():
        x.set_rotation(90)
    ax.set_xlabel(level, fontsize=13)
    ax.set_ylabel(outcome + " Share (%)", fontsize=13)


def get_outcome_share_time(df, outcome, level, time="daily"):
    if time.lower() == "daily":
        gra = "Date"
    elif time.lower() == "weekly":
        gra = "Week"
    else:
        print(
            'Please define the time granularity parameter. Either use "daily" or "weekly"'
        )
        print("The following graph shows daily view")
        gra = "Date"

    _df = df[df["pyOutcome"] == outcome].reset_index(drop=True)
    outcome_per_gra = (
        _df.groupby([gra])
        .count()[["pxInteractionID"]]
        .rename(columns={"pxInteractionID": "total " + time + " " + outcome})
        .reset_index()
    )
    outcome_per_gra = pd.concat(
        [outcome_per_gra.assign(newCol=c) for c in _df[level].unique()],
        ignore_index=True,
    ).rename(columns={"newCol": level})

    level_outcome_share_gra = (
        _df.groupby([level, gra])
        .count()[["pxInteractionID"]]
        .rename(columns={"pxInteractionID": level + " " + outcome + " Count"})
        .reset_index()
        .merge(outcome_per_gra, on=[level, gra], how="outer")
        .fillna(0)
    )

    level_outcome_share_gra[outcome + " Share (%)"] = (
        level_outcome_share_gra[level + " " + outcome + " Count"]
        * 100
        / level_outcome_share_gra["total " + time + " " + outcome]
    )

    return level_outcome_share_gra


def select_date_range_lookback(df, lookback=3):
    _df = df.reset_index(drop=True)
    _df["myDateTime"] = pd.to_datetime(df["Date"])
    last_day = _df["myDateTime"].max()
    mid_day = last_day - timedelta(lookback)
    first_day = last_day - timedelta(lookback * 2)
    _df = _df[
        (_df["myDateTime"] > first_day) & (_df["myDateTime"] <= last_day)
    ].reset_index(drop=True)
    dates = []
    dates_print = {
        0: (mid_day + timedelta(1)).strftime("%Y-%m-%d")
        + " to "
        + last_day.strftime("%Y-%m-%d"),
        1: (first_day + timedelta(1)).strftime("%Y-%m-%d")
        + " to "
        + mid_day.strftime("%Y-%m-%d"),
    }
    dates = [(mid_day + timedelta(1), last_day), (first_day + timedelta(1), mid_day)]
    _df["Date Range"] = np.piecewise(
        np.zeros(len(_df)),
        [
            (pd.to_datetime(_df["myDateTime"].values) >= vals[0])
            & (pd.to_datetime(_df["myDateTime"].values) <= vals[1])
            for vals in dates
        ],
        list(dates_print.keys()),
    )
    _df.replace({"Date Range": dates_print}, inplace=True)
    return _df.drop("myDateTime", axis=1)


def get_delta_df(df, outcome, level, dates):
    # dates=[('2020-07-28', '2020-07-30'), ('2020-08-04', '2020-08-06')]
    _df = get_outcome_share_time(df, outcome, level, time="daily")
    share_delta = select_date_range_lookback(
        _df.drop(outcome + " Share (%)", axis=1).reset_index(drop=True), lookback=dates
    )

    total_range_outcomes = (
        share_delta[["Date", "Date Range", "total daily " + outcome]]
        .drop_duplicates()
        .groupby("Date Range")
        .sum()
        .reset_index()
        .rename(columns={"total daily " + outcome: "total range " + outcome})
    )
    share_delta = (
        share_delta.drop("total daily " + outcome, axis=1)
        .groupby([level, "Date Range"])
        .sum()
        .reset_index()
    )
    share_delta = share_delta.merge(
        total_range_outcomes, on="Date Range", how="outer"
    ).fillna(0)
    share_delta[outcome + " Share (%)"] = (
        share_delta[level + " " + outcome + " Count"]
        * 100
        / share_delta["total range " + outcome]
    )
    share_delta = (
        share_delta[share_delta["Date Range"] == share_delta["Date Range"].min()]
        .drop([level + " " + outcome + " Count", "total range " + outcome], axis=1)
        .merge(
            share_delta[
                share_delta["Date Range"] == share_delta["Date Range"].max()
            ].drop(
                [level + " " + outcome + " Count", "total range " + outcome], axis=1
            ),
            how="outer",
            on=[level],
            suffixes=["_earlier", "_recent"],
        )
        .fillna(0)
    )
    share_delta["delta"] = (
        share_delta[outcome + " Share (%)_recent"]
        - share_delta[outcome + " Share (%)_earlier"]
    )
    return share_delta


def plot_share_delta_graph(df, outcome, level, dates):
    fig, ax = plt.subplots(figsize=(17, 5))
    _df = get_delta_df(df, outcome, level, dates)
    sns.barplot(
        x=level,
        y="delta",
        data=_df.sort_values("delta", ascending=False),
        ax=ax,
        palette="RdBu_r",
    )  # "RdYlGn_r")
    for i in ax.get_xmajorticklabels():
        i.set_rotation(90)
        i.set_fontsize(12)
    for i in ax.get_ymajorticklabels():
        i.set_fontsize(12)
    ax.set_ylabel(level + " " + outcome + " Share Difference (%)")
    if (
        _df["Date Range_recent"].unique()[0].split(" to ")[0]
        == _df["Date Range_recent"].unique()[0].split(" to ")[1]
    ):
        recent_date = str(
            _df["Date Range_recent"].unique()[0].split(" to ")[0]
        )  # pragma: no cover
    else:
        recent_date = str(_df["Date Range_recent"].unique()[0])
    if (
        _df["Date Range_earlier"].unique()[0].split(" to ")[0]
        == _df["Date Range_earlier"].unique()[0].split(" to ")[1]
    ):
        earlier_date = str(
            _df["Date Range_earlier"].unique()[0].split(" to ")[0]
        )  # pragma: no cover
    else:
        earlier_date = str(_df["Date Range_earlier"].unique()[0])
    ax.set_title(
        "Comparing "
        + recent_date
        + " vs "
        + earlier_date
        + " \n delta=recent data - older data"
    )
