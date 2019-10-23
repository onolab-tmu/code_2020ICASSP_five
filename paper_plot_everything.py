# Copyright (c) 2019 Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This script takes the output from the simulation and produces a number of plots
used in the publication.
"""
import argparse
import json
import os
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyroomacoustics as pra
import seaborn as sns
from matplotlib.ticker import MaxNLocator

matplotlib.use("TkAgg")


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Plot the data simulated by separake_near_wall"
    )
    parser.add_argument(
        "-p",
        "--pickle",
        action="store_true",
        help="Read the aggregated data table from a pickle cache",
    )
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="Display the plots at the end of data analysis",
    )
    parser.add_argument(
        "dirs",
        type=str,
        nargs="+",
        metavar="DIR",
        help="The directory containing the simulation output files.",
    )

    cli_args = parser.parse_args()
    plot_flag = cli_args.show
    pickle_flag = cli_args.pickle

    parameters = dict()
    algorithms = dict()
    args = []
    df = None

    data_files = []

    parameters = None
    for i, data_dir in enumerate(cli_args.dirs):

        print("Reading in", data_dir)

        # add the data file from this directory
        data_file = os.path.join(data_dir, "data.json")
        if os.path.exists(data_file):
            data_files.append(data_file)
        else:
            raise ValueError("File {} doesn" "t exist".format(data_file))

        # get the simulation config
        with open(os.path.join(data_dir, "parameters.json"), "r") as f:
            new_parameters = json.load(f)

        if parameters is None:
            parameters = new_parameters
        else:
            parameters["algorithm_kwargs"].update(new_parameters["algorithm_kwargs"])

    # algorithms to take in the plot
    algos = algorithms.keys()

    # check if a pickle file exists for these files
    pickle_file = ".five.pickle"
    rt60_file = ".rt60.pickle"

    if os.path.isfile(pickle_file) and pickle_flag:
        print("Reading existing pickle file...")
        # read the pickle file
        df = pd.read_pickle(pickle_file)
        rt60 = pd.read_pickle(rt60_file)

    else:

        # reading all data files in the directory
        records = []
        rt60_list = []
        for file in data_files:
            with open(file, "r") as f:
                content = json.load(f)
                for seg in content:
                    records += seg

        # build the data table line by line
        print("Building table")
        columns = [
            "Algorithm",
            "Sources",
            "Interferers",
            "Mics",
            "RT60",
            "SINR",
            "seed",
            "Iteration",
            "Iteration_Index",
            "Runtime [s]",
            "SDR [dB]",
            "SIR [dB]",
            "SDR Improvement [dB]",
            "SIR Improvement [dB]",
            "Success",
        ]
        table = []
        num_sources = set()

        copy_fields = [
            "algorithm",
            "n_targets",
            "n_interferers",
            "n_mics",
            "rt60",
            "sinr",
            "seed",
        ]

        number_failed_records = 0

        for record in records:

            algo_kwargs = parameters["algorithm_kwargs"][record["algorithm"]]["kwargs"]
            if "callback_checkpoints" in algo_kwargs:
                checkpoints = algo_kwargs["callback_checkpoints"].copy()
                checkpoints.insert(0, 0)
                checkpoints.append(algo_kwargs["n_iter"])
                algo_n_iter = algo_kwargs["n_iter"]
            else:
                checkpoints = list(range(len(record["sdr"])))
                algo_n_iter = 1

            rt60_list.append(record["rt60"])

            # runtime per iteration, per second of audio
            runtime = (
                record["runtime"]
                / record["n_samples"]
                * parameters["room_params"]["fs"]
                / algo_n_iter
            )
            evaltime = (
                record["eval_time"]
                / record["n_samples"]
                * parameters["room_params"]["fs"]
                / algo_n_iter
            )

            if len(record["sdr"]) == 2 and np.isnan(record["sdr"][-1]):
                number_failed_records += 1
                continue

            for i, (n_iter, sdr, sir) in enumerate(
                zip(checkpoints, record["sdr"], record["sir"])
            ):

                entry = [record[field] for field in copy_fields]
                entry.append(n_iter)
                entry.append(i)

                # seconds processing / second of audio
                entry.append(runtime * n_iter)

                try:
                    sdr_i = np.array(record["sdr"][0])  # Initial SDR
                    sdr_f = np.array(sdr)  # Final SDR
                    sir_i = np.array(record["sir"][0])  # Initial SDR
                    sir_f = np.array(sir)  # Final SDR

                    table.append(
                        entry
                        + [
                            np.mean(sdr_f),
                            np.mean(sir_f),
                            np.mean(sdr_f - sdr_i),
                            np.mean(sir_f - sir_i),
                            float(np.mean(sir_f - sir_i) >= 1.0),
                        ]
                    )
                except:
                    continue

        # create a pandas frame
        print("Making PANDAS frame...")
        df = pd.DataFrame(table, columns=columns)
        rt60 = pd.DataFrame(rt60_list, columns=["RT60"])

        df.to_pickle(pickle_file)
        rt60.to_pickle(rt60_file)

        if number_failed_records > 0:
            warnings.warn(f"Number of failed record: {number_failed_records}")

    # Draw the figure
    print("Plotting...")

    # sns.set(style='whitegrid')
    # sns.plotting_context(context='poster', font_scale=2.)
    # pal = sns.cubehelix_palette(8, start=0.5, rot=-.75)

    substitutions = {
        "Algorithm": {
            "five_laplace": "FIVE (Laplace)",
            "five_gauss": "FIVE (Gauss)",
            "overiva_laplace": "OverIVA (Laplace)",
            "overiva_gauss": "OverIVA (Gauss)",
            "ogive_laplace": "OGIVEw (Laplace)",
            "ogive_gauss": "OGIVEw (Gauss)",
            "auxiva_laplace": "AuxIVA (Laplace)",
            "auxiva_gauss": "AuxIVA (Gauss)",
            "pca": "PCA",
        }
    }

    df = df.replace(substitutions)

    df_melt = df.melt(id_vars=df.columns[:-5], var_name="metric")
    df_melt = df_melt.replace(substitutions)

    # Aggregate the convergence curves
    df_agg = (
        df_melt.groupby(
            by=[
                "Algorithm",
                "Sources",
                "Interferers",
                "SINR",
                "Mics",
                "Iteration",
                "metric",
            ]
        )
        .mean()
        .reset_index()
    )

    all_algos = [
        "AuxIVE (Laplace)",
        "AuxIVE (Gauss)",
        "OverIVA (Laplace)",
        "OverIVA (Gauss)",
        "OGIVEw (Laplace)",
        "OGIVEw (Gauss)",
        "AuxIVA (Laplace)",
        "AuxIVA (Gauss)",
    ]

    sns.set(
        style="whitegrid",
        context="paper",
        font_scale=0.75,
        rc={
            # 'figure.figsize': (3.39, 3.15),
            "lines.linewidth": 1.0,
            # 'font.family': 'sans-serif',
            # 'font.sans-serif': [u'Helvetica'],
            # 'text.usetex': False,
        },
    )
    pal = sns.cubehelix_palette(
        4, start=0.5, rot=-0.5, dark=0.3, light=0.75, reverse=True, hue=1.0
    )
    sns.set_palette(pal)

    if not os.path.exists("figures"):
        os.mkdir("figures")

    fig_dir = "figures/{}_{}_{}".format(
        parameters["name"], parameters["_date"], parameters["_git_sha"]
    )

    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    the_metrics = {
        "improvements": ["SDR Improvement [dB]", "SIR Improvement [dB]"],
        "raw": ["SDR [dB]", "SIR [dB]"],
    }

    plt_kwargs = {
        # "improvements": {"ylim": [-5.5, 20.5], "yticks": [-5, 0, 5, 10, 15]},
        # "raw": {"ylim": [-5.5, 20.5], "yticks": [-5, 0, 5, 10, 15]},
        # "runtime": {"ylim": [-0.5, 40.5], "yticks": [0, 10, 20, 30]},
    }

    # Fourth figure
    # Histogram of RT60
    select = np.logical_and(df_melt["SINR"] == 5, df_melt["Iteration"] == 0)
    select = np.logical_and(select, df_melt["Algorithm"] == "AuxIVE (Laplace)")
    select = np.logical_and(select, df_melt["Mics"] == 2)
    select = np.logical_and(select, df_melt["Interferers"] == 10)
    select = np.logical_and(select, df_melt["metric"] == "Success")
    plt.figure(figsize=(3.35, 1.8))
    plt.hist(df_melt[select]["RT60"] * 100.0)
    plt.xlabel("RT60 [ms]")
    plt.ylabel("Frequency")
    sns.despine(offset=10, trim=False, left=True, bottom=True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    fig_fn = os.path.join(fig_dir, f"figure4_rt60_hist.pdf")
    plt.savefig(fig_fn, bbox_inches="tight")
    plt.close()

    # Fifth figure
    # Success as a function of # interferers
    for n_mics in parameters["n_mics_list"]:
        for sinr in parameters["sinr_list"]:
            select = (
                (df_melt["Mics"] == n_mics)
                & (df_melt["SINR"] == sinr)
                & (df_melt["metric"] == "Success")
            )
            n_iter_map = {
                "AuxIVE": 3,
                "OverIVA": 14,
                "OGIVEw": 14,
                "AuxIVA": 14,
                "PCA": 1,
            }
            df_fig5 = pd.DataFrame(columns=df_melt.columns)
            for name, n in n_iter_map.items():
                sel = np.logical_and(select, df_melt["Algorithm"].str.startswith(name))
                sel = np.logical_and(sel, df_melt["Iteration_Index"] == n)
                df_fig5 = df_fig5.append(df_melt[sel])
            plt.figure(figsize=(3.35, 2.2))
            sns.barplot(
                data=df_fig5,
                x="Interferers",
                y="value",
                hue="Algorithm",
                hue_order=all_algos,
            )
            leg = plt.legend(loc="upper left", fontsize="x-small")
            leg.get_frame().set_linewidth(0.2)
            plt.ylabel("Pr($\Delta$SIR $\geq$ 1 dB)")
            sns.despine(offset=10, trim=False, left=True, bottom=True)
            fig_fn = os.path.join(
                fig_dir, f"figure5_success_mics{n_mics}_sinr{sinr}.pdf"
            )
            plt.savefig(fig_fn, bbox_inches="tight", bbox_extra_artists=[leg])
            plt.close()

    # First figure
    # Convergence curves: Time/Iteration vs SDR
    n_mics = 5
    for n_interferers in parameters["n_interferers_list"]:

        select = np.logical_and(
            df_agg["Mics"] == n_mics, df_agg["metric"] == "SDR Improvement [dB]"
        )
        select = np.logical_and(select, df_agg["Interferers"] == n_interferers)

        if len(df_agg[select]) == 0:
            continue

        # select = np.logical_and(df_agg["Interferers"] == 5, select)
        g = sns.FacetGrid(
            df_agg[select],
            col="SINR",
            hue="Algorithm",
            hue_order=all_algos,
            hue_kws=dict(marker=["."] * len(all_algos)),
            xlim=[0.0, 0.5],
        )
        g.map(plt.plot, "Runtime [s]", "value")
        plt.legend()

        fig_fn = os.path.join(
            fig_dir, f"figure1_conv_interf{n_interferers}_mics{n_mics}.pdf"
        )
        plt.savefig(fig_fn, bbox_inches="tight")
        plt.close()

    # Second figure
    # Convergence curves: Time/Iteration vs SDR
    sinr = 5
    for n_interferers in parameters["n_interferers_list"]:

        select = np.logical_and(
            df_agg["SINR"] == sinr, df_agg["metric"] == "SDR Improvement [dB]"
        )
        select = np.logical_and(select, df_agg["Interferers"] == n_interferers)

        if len(df_agg[select]) == 0:
            continue

        # select = np.logical_and(df_agg["Interferers"] == 5, select)
        g = sns.FacetGrid(
            df_agg[select],
            col="Mics",
            hue="Algorithm",
            hue_order=all_algos,
            hue_kws=dict(marker=["."] * len(all_algos)),
            xlim=[0.0, 0.5],
        )
        g.map(plt.plot, "Runtime [s]", "value")
        plt.legend()

        fig_fn = os.path.join(
            fig_dir, f"figure2_conv_interf{n_interferers}_sinr{sinr}.pdf"
        )
        plt.savefig(fig_fn, bbox_inches="tight")
        plt.close()

    # Third figure
    # Classic # of microphones vs metric (box-plots ?)
    n_cols = 1
    full_width = 6.93  # inches, == 17.6 cm, double column width
    half_width = 3.35  # inches, == 8.5 cm, single column width
    # width = aspect * height
    aspect = 3 / 2  # width / height
    height = half_width / aspect / 2.0

    all_algos_fig3 = [
        "AuxIVE (Laplace)",
        "AuxIVE (Gauss)",
        "OverIVA (Laplace)",
        "OverIVA (Gauss)",
        "AuxIVA (Laplace)",
        "AuxIVA (Gauss)",
    ]

    iteration_index = 3  # we run algo 5 times
    for sinr in parameters["sinr_list"]:
        for n_interferers in parameters["n_interferers_list"]:

            select = (
                (df_melt["Iteration_Index"] == iteration_index)
                & (df_melt["Interferers"] == n_interferers)
                & (df_melt["SINR"] == sinr)
            )

            for m_name, metric in the_metrics.items():

                new_select = select & df_melt.metric.isin(metric)

                if len(df_melt[new_select]) == 0:
                    continue

                g = sns.catplot(
                    data=df_melt[new_select],
                    x="Mics",
                    y="value",
                    hue="Algorithm",
                    row="metric",
                    row_order=metric,
                    hue_order=all_algos_fig3,
                    kind="box",
                    legend=False,
                    aspect=aspect,
                    height=height,
                    linewidth=0.5,
                    fliersize=0.3,
                    whis=5.0,
                    sharey="row",
                    # size=3, aspect=0.65,
                    # margin_titles=True,
                )

                if m_name in plt_kwargs:
                    g.set(**plt_kwargs[metric])
                # remove original titles before adding custom ones
                [plt.setp(ax.texts, text="") for ax in g.axes.flat]
                g.set_titles(col_template="SINR={col_name}", row_template="")

                all_artists = []

                # left_ax = g.facet_axis(2, 0)
                left_ax = g.facet_axis(0, 0)
                leg = left_ax.legend(
                    title="Algorithms",
                    frameon=True,
                    framealpha=0.85,
                    fontsize="x-small",
                    loc="center",
                    bbox_to_anchor=[1.5, 1.0],
                )
                leg.get_frame().set_linewidth(0.2)
                all_artists.append(leg)

                sns.despine(offset=10, trim=False, left=True, bottom=True)

                plt.tight_layout(pad=0.1)

                for c, lbl in enumerate(metric):
                    g_ax = g.facet_axis(c, 0)
                    g_ax.set_ylabel(lbl)

                fig_fn = os.path.join(
                    fig_dir, f"figure3_{m_name}_interf{n_interferers}_sinr{sinr}.pdf"
                )
                plt.savefig(fig_fn, bbox_extra_artists=all_artists, bbox_inches="tight")
                plt.close()

    if plot_flag:
        plt.show()
