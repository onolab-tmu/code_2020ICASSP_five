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
This file contains the code to run the systematic simulation for evaluation
of overiva and other algorithms.
"""
import argparse, json, os, sys
import numpy as np
import pyroomacoustics as pra
import rrtools

# Get the data if needed
from get_data import get_data, samples_dir
from room_builder import random_room_builder, callback_noise_mixer

get_data()

# Routines for manipulating audio samples
sys.path.append(samples_dir)
from generate_samples import sampling, wav_read_center

# find the absolute path to this file
base_dir = os.path.abspath(os.path.split(__file__)[0])


def init(parameters):
    parameters["base_dir"] = base_dir


def one_loop(args):
    global parameters

    import time
    import numpy

    np = numpy

    import pyroomacoustics

    pra = pyroomacoustics

    import os
    import sys

    sys.path.append(parameters["base_dir"])

    from auxiva_pca import auxiva_pca, pca_separation
    from five import five
    from ive import ogive
    from overiva import overiva
    from pyroomacoustics.bss.common import projection_back
    from room_builder import callback_noise_mixer, random_room_builder

    # import samples helper routine
    from get_data import samples_dir

    sys.path.append(os.path.join(parameters['base_dir'], samples_dir))
    from generate_samples import wav_read_center

    n_targets, n_interferers, n_mics, sinr, wav_files, room_seed, seed = args

    # this is the underdetermined case. We don't do that.
    if n_mics < n_targets:
        return []

    # set MKL to only use one thread if present
    try:
        import mkl

        mkl.set_num_threads(1)
    except ImportError:
        pass

    # set the RNG seed
    rng_state = np.random.get_state()
    np.random.seed(seed)

    # STFT parameters
    framesize = parameters["stft_params"]["framesize"]
    hop = parameters["stft_params"]["hop"]
    if parameters["stft_params"]["window"] == "hann":
        win_a = pra.hamming(framesize)
    else:  # default is Hann
        win_a = pra.hann(framesize)
    win_s = pra.transform.compute_synthesis_window(win_a, hop)

    # Generate the audio signals

    # get the simulation parameters from the json file
    # Simulation parameters
    sources_var = np.ones(n_targets)

    # total number of sources
    n_sources = n_targets + n_interferers

    # Read the signals
    wav_files = [os.path.join(parameters["base_dir"], fn) for fn in wav_files]
    signals = wav_read_center(wav_files[:n_sources], seed=123)

    # Get a random room
    room, rt60 = random_room_builder(
        signals, n_mics, seed=room_seed, **parameters["room_params"]
    )
    premix = room.simulate(return_premix=True)

    # mix the signal
    n_samples = premix.shape[2]
    mix = callback_noise_mixer(
        premix,
        sinr=sinr,
        diffuse_ratio=parameters["sinr_diffuse_ratio"],
        n_src=n_sources,
        n_tgt=n_targets,
        tgt_std=np.sqrt(sources_var),
        ref_mic=parameters["ref_mic"],
    )

    # sum up the background
    # shape (n_mics, n_samples)
    background = np.sum(premix[n_targets:n_sources, :, :], axis=0)

    # shape (n_targets+1, n_samples, n_mics)
    ref = np.zeros(
        (n_targets + 1, premix.shape[2], premix.shape[1]), dtype=premix.dtype
    )
    ref[:n_targets, :, :] = premix[:n_targets, :, :].swapaxes(1, 2)
    ref[n_targets, :, :] = background.T

    synth = np.zeros_like(ref)

    # START BSS
    ###########

    # shape: (n_frames, n_freq, n_mics)
    X_all = pra.transform.analysis(mix.T, framesize, hop, win=win_a)
    X_mics = X_all[:, :, :n_mics]

    # convergence monitoring callback
    def convergence_callback(
        Y, X, n_targets, SDR, SIR, eval_time, ref, framesize, win_s, algo_name
    ):
        t_in = time.perf_counter()

        # projection back
        z = projection_back(Y, X[:, :, 0])
        Y = Y * np.conj(z[None, :, :])

        from mir_eval.separation import bss_eval_sources

        if Y.shape[2] == 1:
            y = pra.transform.synthesis(Y[:, :, 0], framesize, hop, win=win_s)[:, None]
        else:
            y = pra.transform.synthesis(Y, framesize, hop, win=win_s)

        if algo_name not in parameters["overdet_algos"]:
            new_ord = np.argsort(np.std(y, axis=0))[::-1]
            y = y[:, new_ord]

        m = np.minimum(y.shape[0] - hop, ref.shape[1])

        synth[:n_targets, :m, 0] = y[hop : m + hop, :n_targets].T
        synth[n_targets, :m, 0] = y[hop : m + hop, 0]

        sdr, sir, sar, perm = bss_eval_sources(
            ref[: n_targets + 1, :m, 0], synth[:, :m, 0]
        )
        SDR.append(sdr[:n_targets].tolist())
        SIR.append(sir[:n_targets].tolist())

        t_out = time.perf_counter()
        eval_time.append(t_out - t_in)

    # store results in a list, one entry per algorithm
    results = []

    # compute the initial values of SDR/SIR
    init_sdr = []
    init_sir = []

    convergence_callback(
        X_mics, X_mics, n_targets, init_sdr, init_sir, [], ref, framesize, win_s, "init"
    )

    for full_name, params in parameters["algorithm_kwargs"].items():

        name = params["algo"]
        kwargs = params["kwargs"]

        if name == "auxiva_pca" and n_targets == 1:
            # PCA doesn't work for single source scenario
            continue
        elif name in ["ogive", "five"] and n_targets != 1:
            # OGIVE is only for single target
            continue

        results.append(
            {
                "algorithm": full_name,
                "n_targets": n_targets,
                "n_interferers": n_interferers,
                "n_mics": n_mics,
                "rt60": rt60,
                "sinr": sinr,
                "seed": seed,
                "sdr": [],
                "sir": [],  # to store the result
                "runtime": np.nan,
                "eval_time": np.nan,
                "n_samples": n_samples,
            }
        )

        # this is used to keep track of time spent in the evaluation callback
        eval_time = []

        def cb(Y):
            convergence_callback(
                Y,
                X_mics,
                n_targets,
                results[-1]["sdr"],
                results[-1]["sir"],
                eval_time,
                ref,
                framesize,
                win_s,
                name,
            )

        # avoid one computation by using the initial values of sdr/sir
        results[-1]["sdr"].append(init_sdr[0])
        results[-1]["sir"].append(init_sir[0])

        try:
            t_start = time.perf_counter()

            if name == "auxiva":
                # Run AuxIVA
                # this calls full IVA when `n_src` is not provided
                Y = overiva(X_mics, callback=cb, **kwargs)

            elif name == "auxiva_pca":
                # Run AuxIVA
                Y = auxiva_pca(
                    X_mics, n_src=n_targets, callback=cb, proj_back=False, **kwargs
                )

            elif name == "overiva":
                # Run BlinkIVA
                Y = overiva(
                    X_mics, n_src=n_targets, callback=cb, proj_back=False, **kwargs
                )

            elif name == "overiva2":
                # Run BlinkIVA
                Y = overiva(
                    X_mics, n_src=n_targets, callback=cb, proj_back=False, **kwargs
                )

            elif name == "five":
                # Run AuxIVE
                Y = five(X_mics, callback=cb, proj_back=False, **kwargs)

            elif name == "ilrma":
                # Run AuxIVA
                Y = pra.bss.ilrma(X_mics, callback=cb, proj_back=False, **kwargs)

            elif name == "ogive":
                # Run OGIVE
                Y = ogive(X_mics, callback=cb, proj_back=False, **kwargs)

            elif name == "pca":
                # Run PCA
                Y = pca_separation(X_mics, n_src=n_targets)

            else:
                continue

            t_finish = time.perf_counter()

            # The last evaluation
            convergence_callback(
                Y,
                X_mics,
                n_targets,
                results[-1]["sdr"],
                results[-1]["sir"],
                [],
                ref,
                framesize,
                win_s,
                name,
            )

            results[-1]["eval_time"] = np.sum(eval_time)
            results[-1]["runtime"] = t_finish - t_start - results[-1]["eval_time"]

        except:
            import os, json

            pid = os.getpid()
            # report last sdr/sir as np.nan
            results[-1]["sdr"].append(np.nan)
            results[-1]["sir"].append(np.nan)
            # now write the problem to file
            fn_err = os.path.join(
                parameters["_results_dir"], "error_{}.json".format(pid)
            )
            with open(fn_err, "a") as f:
                f.write(json.dumps(results[-1], indent=4))
            # skip to next iteration
            continue

    # restore RNG former state
    np.random.set_state(rng_state)

    return results


def generate_arguments(parameters):
    """ This will generate the list of arguments to run simulation for """

    rng_state = np.random.get_state()
    np.random.seed(parameters["seed"])

    # Maximum total number of sources
    n_sources = np.max(parameters["n_interferers_list"]) + np.max(
        parameters["n_targets_list"]
    )

    # First we randomly select all the speech samples
    gen_files_seed = int(np.random.randint(2 ** 32, dtype=np.uint32))
    all_wav_files = sampling(
        parameters["n_repeat"],
        n_sources,
        parameters["samples_list"],
        gender_balanced=True,
        seed=gen_files_seed,
    )

    # Pick the seeds to reproducibly build a bunch of random rooms
    room_seeds = np.random.randint(
        2 ** 32, size=parameters["n_repeat"], dtype=np.uint32
    ).tolist()

    args = []

    for n_targets in parameters["n_targets_list"]:
        for n_interferers in parameters["n_interferers_list"]:
            for n_mics in parameters["n_mics_list"]:

                # we don't do underdetermined
                if n_targets > n_mics:
                    continue

                for sinr in parameters["sinr_list"]:
                    for wav_files, room_seed in zip(all_wav_files, room_seeds):

                        # generate the seed for this simulation
                        seed = int(np.random.randint(2 ** 32, dtype=np.uint32))

                        # add the new combination to the list
                        args.append(
                            [
                                n_targets,
                                n_interferers,
                                n_mics,
                                sinr,
                                wav_files,
                                room_seed,
                                seed,
                            ]
                        )

    np.random.set_state(rng_state)

    return args


if __name__ == "__main__":

    rrtools.run(
        one_loop,
        generate_arguments,
        func_init=init,
        base_dir=base_dir,
        results_dir="data/",
        description="Simulation for Independent Vector Extraction via Iterative SINR Maximization  (submitted to ICASSP 2020)",
    )
