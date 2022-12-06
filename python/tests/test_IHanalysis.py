import sys
import pytest

sys.path.append("python")
from pdstools.ih.IHanalysis import *
from pdstools.utils.cdh_utils import readDSExport

import matplotlib.pyplot as plt

# This is not a good set of tests - just running to make sure we don't run into errors
# The IHanalysis code is not actively maintained, so no point in too exhaustively testing.


@pytest.fixture
def init():
    return readDSExport(
        "Data-pxStrategyResult_pxInteractionHistory_20210101T010000_GMT.zip",
        path="data",
    )


def test_no_errors(init):
    df = initial_prep(init)
    assert df.shape == (127123, 52)
    plot_daily_accept_rate(
        df,
        "Accepted",
        "Rejected",
        **{"hue": ["pyChannel"], "allTime": True, "shrinkTicks": True}
    )
    plot_weekly_accept_rate(
        df, "Accepted", "Rejected", **{"showOutlier": True, "hue": "pyDirection"}
    )
    plot_daily_cumulative_accept_rate(
        df[df["pyName"] == "UPlusPersonal"],
        "Accepted",
        "Rejected",
        **{
            "allTime": True,
            "shrinkTicks": True,
            "showOutlier": True,
            "title": "Proposition: UPlusPersonal",
        }
    )
    plot_daily_cumulative_accept_rate(
        df,
        "Accepted",
        "Rejected",
        **{"allTime": True, "shrinkTicks": True, "showOutlier": True}
    )
    plot_daily_cumulative_accept_rate(
        df,
        "Clicked",
        "NoResponse",
        **{
            "hue": ["pyGroup", "pyDirection", "pyChannel"],
            "allTime": True,
            "shrinkTicks": True,
        }
    )
    plot_outcome_count_time(
        df,
        "Accepted",
        "weekly",
        **{"hue": "pyIssue", "allTime": True, "shrinkTicks": True}
    )

    plot_outcome_count_time(
        df,
        "Accepted",
        "yearly",
        **{
            "hue": ["pyIssue", "pyGroup"],
            "allTime": True,
            "shrinkTicks": True,
            "allTime": True,
        }
    )

    plot_outcome_share_graph(
        df[df["pyChannel"] == "Web"], "Accepted", "pyName", "pyGroup"
    )
    plot_outcome_share_graph(
        df[df["pyChannel"] == "Web"], "Accepted", "pyName", "pyGroup"
    )
    get_outcome_share_time(df, outcome="Accepted", level="pyName", time="yearly")

    click_share_name_daily = get_outcome_share_time(
        df[df["pyChannel"] == "Web"], "Clicked", "pyName", time="daily"
    )
    click_share_name_weekly = get_outcome_share_time(
        df[df["pyChannel"] == "Web"], "Clicked", "pyName", time="weekly"
    )
    get_daily_graph(
        click_share_name_daily[click_share_name_daily["pyName"] == "UPlusGold"],
        "Date",
        "Clicked Share (%)",
        **{"shrinkTicks": True}
    )
    get_daily_graph(
        click_share_name_weekly[click_share_name_weekly["pyName"] == "UPlusGold"],
        "Week",
        "Clicked Share (%)",
        **{"shrinkTicks": True, "ylabel": "test", "ylim": 20}
    )
    click_share_direction_daily = get_outcome_share_time(
        df, "Accepted", "pyDirection", time="daily"
    )
    get_daily_graph(
        click_share_direction_daily,
        "Date",
        "Accepted Share (%)",
        **{"shrinkTicks": True, "hue": "pyDirection"}
    )

    plot_share_delta_graph(
        df[df["pyChannel"] == "SMS"].reset_index(drop=True),
        "Clicked",
        "pyName",
        dates=4,
    )

    plot_df = get_accept_rate(
        df[df["pyDirection"] == "Inbound"], "Accepted", "Rejected", "pyName"
    )

    fig, ax = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"hspace": 0.05}
    )
    sort = plot_df.sort_values("Accept Rate (%)", ascending=False)["pyName"].tolist()
    sns.barplot(x="pyName", y="Accept Rate (%)", data=plot_df, ax=ax[0], order=sort)
    sns.barplot(x="pyName", y="Accepted", data=plot_df, ax=ax[1], order=sort)
    sns.pointplot(x="pyName", y="Total", data=plot_df, ax=ax[1], order=sort)
    for x in ax[1].get_xmajorticklabels():
        x.set_rotation(90)
    ax[0].set_xlabel("")
    ax[1].text(2, 2000, "The bars show the accepts\nThe line shows accept+reject")
    ax[0].set_ylabel("Accept Rate (%)", fontsize=13)
    ax[1].set_ylabel("Accepts", fontsize=13)
    ax[0].set_title("Offers within Inbound direction")
