# Copyright 2023, Patrick Riley, github: pfrstg

import collections
import os

import pytest
from PIL import Image

import voxart

DataFramePair = collections.namedtuple(
    "DataFramePair", ["df_filled_results", "df_results"]
)


@pytest.fixture
def dataframes():
    goal = voxart.Goal.from_image(
        Image.open(os.path.join(os.path.dirname(__file__), "../assets/chiral_7.png"))
    )
    opts = voxart.SearchOptions()
    opts.filled_num_batches = 2
    opts.filled_num_to_pursue = 2
    opts.connector_num_iterations_per = 3
    opts.connector_frace = 0.1
    opts.top_n = 1
    filled_results, results = voxart.search(goal, opts)
    assert filled_results is not None
    assert results is not None
    return DataFramePair(
        filled_results.all_objective_values(), results.all_objective_values()
    )


def test_objective_distributions_plot(dataframes):
    voxart.objective_distributions_plot(
        dataframes.df_filled_results, dataframes.df_results
    )


def test_overall_progress_plot(dataframes):
    voxart.overall_progress_plot(dataframes.df_results)


def test_connector_iterations_plot(dataframes):
    voxart.connector_iterations_plot(dataframes.df_results)


def test_batch_plot(dataframes):
    voxart.batch_plot(dataframes.df_filled_results, dataframes.df_results)


def test_forms_plot(dataframes):
    voxart.forms_plot(dataframes.df_results)
