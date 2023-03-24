# Copyright 2023, Patrick Riley, github: pfrstg

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import voxart


def _ecdf_plus_hist(ax, vals, color):
    ax.hist(vals, color="tab:blue")
    ax.set_ylabel("count")
    ax2 = ax.twinx()
    sns.ecdfplot(ax=ax2, data=vals, color="tab:red", lw=2)


def _ties_plot(ax, vals, label):
    # data = vals.value_counts().value_counts()
    # ax.bar(data.index, data, label=label)
    sns.ecdfplot(ax=ax, data=vals.value_counts(), label=label)
    ax.set_xlabel("number of ties in rank")


def objective_distributions_plot(
    df_filled_results: pd.DataFrame, df_results: pd.DataFrame
) -> plt.Figure:
    """Plots for understanding distribution of objective values.

    Expects dataframes from the results of search(), converted via all_objective_values()
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    _ecdf_plus_hist(
        ax,
        df_filled_results.loc[df_filled_results["filled_is_unique"], "objective_value"],
        color="b",
    )
    ax.set_title("filled objective_value")

    ax = axes[1]
    _ecdf_plus_hist(ax, df_results["objective_value"], color="g")
    ax.set_title("complete objective_value")

    ax = axes[2]
    _ties_plot(ax, df_filled_results["objective_value_rank"], label="filled")
    _ties_plot(ax, df_results["objective_value_rank"], label="complete")
    ax.legend()

    fig.tight_layout()
    plt.close()
    return fig


def overall_progress_plot(df_results: pd.DataFrame):
    """Shows how the objective value evolves over the search."""
    df_results = df_results.sort_values(
        [
            "batch_idx",
            "filled_form_idx",
            "filled_iteration",
            "conn_iteration",
            "idx_in_bottom_location",
        ]
    )
    df_results["idx_overall"] = np.arange(len(df_results))

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    df = df_results[~df_results["is_unique"]]
    ax.plot(
        df["idx_overall"],
        df["objective_value"],
        color="gray",
        linestyle="",
        marker="|",
        alpha=0.75,
    )
    df = df_results[df_results["is_unique"]]
    ax.plot(
        df["idx_overall"],
        df["objective_value"],
        color="b",
        linestyle="",
        marker=".",
        alpha=0.5,
    )
    ax.plot(
        df_results["idx_overall"],
        df_results["objective_value"].cummin(),
        color="g",
        marker="",
        lw=2,
    )
    ax.set_xlabel("overall_idx")
    ax.set_ylabel("objective_value")
    plt.close()
    return fig


def connector_iterations_plot(df_results: pd.DataFrame):
    """Helps to understand if the connector search iterations are sufficient."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Show only the top 25% of ranks
    rank_limit = df_results["objective_value_rank"].max() // 4
    df = df_results[
        df_results["is_unique"] & (df_results["objective_value_rank"] <= rank_limit)
    ]
    axes[0].scatter(df["conn_iteration"], df["objective_value"], marker=".")
    axes[0].set_xlabel("conn_iteration")
    axes[0].set_ylabel("objective_value")

    df_summ = (
        df_results[df_results["idx_in_bottom_location"] == 0]
        .groupby("conn_iteration")["is_unique"]
        .apply(
            lambda s: pd.Series(
                index=pd.Index(name="tmp_is_unique", data=[True, False]),
                data=(np.sum(s == True), np.sum(s == False)),
            )
        )
        .reset_index()
        .rename(columns={"is_unique": "count", "tmp_is_unique": "is_unique"})
    )
    df = df_summ[df_summ["is_unique"] == True]
    axes[1].bar(df["conn_iteration"], df["count"], 0.5, label="unique", color="green")
    bottom = df["count"]
    df = df_summ[df_summ["is_unique"] == False]
    axes[1].bar(
        df["conn_iteration"],
        df["count"],
        0.5,
        label="repeat",
        color="red",
        bottom=bottom,
    )
    axes[1].legend()
    axes[1].set_xlabel("conn_iteration")
    axes[1].set_ylabel("count")
    plt.close()
    return fig


def batch_plot(df_filled_results: pd.DataFrame, df_results: pd.DataFrame):
    """Helps to understand if the batches and sizes are sufficient."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    df = df_results[df_results["idx_in_bottom_location"] == 0]
    ax = axes[0, 0]
    ax.scatter(df["idx_in_batch"], df["objective_value_rank"], marker=".")
    ax.set_xlabel("idx_in_batch")
    ax.set_ylabel("objective_value_rank")

    ax = axes[0, 1]
    ax.scatter(df["idx_in_batch"], df["objective_value"], marker=".")
    ax.set_xlabel("idx_in_batch")
    ax.set_ylabel("objective_value")

    ax = axes[1, 0]
    batch_size = df_filled_results["idx_in_batch"].max() + 1
    df_filled_results["idx"] = (
        df_filled_results["batch_idx"] * batch_size + df_filled_results["idx_in_batch"]
    )

    ax.scatter(
        df_filled_results["idx"], df_filled_results["filled_is_unique"], marker="_"
    )
    for batch in range(0, df["batch_idx"].max()):
        ax.axvline((batch + 1) * batch_size, color="gray", zorder=0)
    ax.set_xlabel("batch + idx_in_batch")
    ax.set_ylabel("filled is unique")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["False", "True"])

    ax = axes[1, 1]
    df_summ = (
        df_filled_results.groupby("filled_iteration")["filled_is_unique"]
        .apply(
            lambda s: pd.Series(
                index=pd.Index(name="tmp_filled_is_unique", data=[True, False]),
                data=(np.sum(s == True), np.sum(s == False)),
            )
        )
        .reset_index()
        .rename(
            columns={
                "filled_is_unique": "count",
                "tmp_filled_is_unique": "filled_is_unique",
            }
        )
    )
    df = df_summ[df_summ["filled_is_unique"] == True]
    ax.bar(df["filled_iteration"], df["count"], 0.5, label="unique", color="green")
    bottom = df["count"]
    df = df_summ[df_summ["filled_is_unique"] == False]
    ax.bar(
        df["filled_iteration"],
        df["count"],
        0.5,
        label="repeat",
        color="red",
        bottom=bottom,
    )
    ax.legend()
    ax.set_xlabel("filled_iteration")
    ax.set_ylabel("count (filled designs)")

    plt.close()
    return fig


def forms_plot(df_results: pd.DataFrame, top_n: int = 50):
    """Shows what forms achieve good objective values"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    df_best_forms = df_results[df_results["objective_value_rank"] < top_n]
    best_unique = (
        df_best_forms.loc[df_best_forms["is_unique"], "filled_form_idx"]
        .value_counts()
        .rename("unique")
    )
    best_repeat = (
        df_best_forms.loc[~df_best_forms["is_unique"], "filled_form_idx"]
        .value_counts()
        .rename("repeat")
    )
    df_merged = pd.DataFrame(best_unique).merge(
        pd.DataFrame(best_repeat), left_index=True, right_index=True, how="outer"
    )

    ax.bar(df_merged.index, df_merged["unique"], color="green", label="unique")
    ax.bar(
        df_merged.index,
        df_merged["repeat"],
        bottom=df_merged["unique"],
        color="red",
        label="repeat",
    )
    ax.set_xticks(df_merged.index)
    ax.set_xlabel("form_idx")
    ax.set_ylabel("count")
    ax.set_title(f"Forms of the top {top_n}")
    ax.legend()

    ax = axes[1]
    sns.ecdfplot(
        ax=ax,
        data=df_results,
        x="objective_value",
        hue="filled_form_idx",
        palette="tab10",
    )
    ax.set_title("Objective value distribution by form")

    fig.tight_layout()
    plt.close()
    return fig
