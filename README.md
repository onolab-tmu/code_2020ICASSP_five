Fast Independent Vector Extraction by Iterative SINR Maximization
=================================================================

This repository provides implementations and code to reproduce the results
of the paper

> R. Scheibler and N. Ono, [*"Fast Independent Vector Extraction by Iterative SINR Maximization,"*](http://arxiv.org/abs/1910.10654) 2019.

Abstract
--------

We propose fast independent vector extraction (FIVE), a new algorithm that
blindly extracts a single non-Gaussian source from a Gaussian background. The
algorithm iteratively computes beamforming weights maximizing the
signal-to-interference-and-noise ratio for an approximate noise covariance
matrix. We demonstrate that this procedure minimizes the negative
log-likelihood of the input data according to a well-defined probabilistic
model. The minimization is carried out via the auxiliary function technique
whereas, unlike related methods, the auxiliary function is globally minimized
at every iteration. Numerical experiments are carried out to assess the
performance of FIVE. We find that it is vastly superior to competing methods in
terms of convergence speed, and has high potential for real-time applications.

Authors
-------

[Robin Scheibler](http://robinscheibler.org) and [Nobutaka
Ono](http://www.comp.sd.tmu.ac.jp/onolab/index-e.html) are with the Faculty of
Systems Design at [Tokyo Metropolitan University](https://www.tmu.ac.jp/english/index.html).

### Contact

    Robin Scheibler (robin[at]tmu[dot]ac[dot]jp)
    6-6 Asahigaoka
    Hino, Tokyo
    191-0065 Japan

Preliminaries
-------------

The preferred way to run the code is using [anaconda](https://www.anaconda.com/distribution/).
An `environment.yml` file is provided to install the required dependencies.

    # create the minimal environment
    conda env create -f environment.yml

    # switch to new environment
    conda activate 2019_scheibler_five

Test FIVE
---------

The algorithm can be tested and compared to others using the sample
script `example.py`. It can be run as follows.

    $ python ./example.py --help
    The samples directory ./samples seems to exist already. Delete if re-download is needed.
    usage: example.py [-h] [--no_cb] [-b BLOCK]
                      [-a {auxiva,auxiva_pca,overiva,ilrma,five,ogive}]
                      [-d {laplace,gauss}] [-i {eye,eig,ogive}] [-m MICS]
                      [-s SRCS] [-n N_ITER] [--gui] [--save]

    Demonstration of blind source extraction using FIVE.

    optional arguments:
      -h, --help            show this help message and exit
      --no_cb               Removes callback function
      -b BLOCK, --block BLOCK
                            STFT block size
      -a {auxiva,auxiva_pca,overiva,ilrma,five,ogive}, --algo {auxiva,auxiva_pca,overiva,ilrma,five,ogive}
                            Chooses BSS method to run
      -d {laplace,gauss}, --dist {laplace,gauss}
                            IVA model distribution
      -i {eye,eig,ogive}, --init {eye,eig,ogive}
                            Initialization, eye: identity, eig: principal
                            eigenvectors
      -m MICS, --mics MICS  Number of mics
      -s SRCS, --srcs SRCS  Number of sources
      -n N_ITER, --n_iter N_ITER
                            Number of iterations
      --gui                 Creates a small GUI for easy playback of the sound
                            samples
      --save                Saves the output of the separation to wav files

For example, we can run FIVE with 4 microphones.

    python ./example.py -a five -m 4

Reproduce the Results
---------------------

The code can be run serially, or using multiple parallel workers via
[ipyparallel](https://ipyparallel.readthedocs.io/en/latest/).
Moreover, it is possible to only run a few loops to test whether the
code is running or not.

1. Run **test** loops **serially**

        python ./paper_simulation.py ./paper_sim_config.json -t -s

2. Run **test** loops in **parallel**

        # start workers in the background
        # N is the number of parallel process, often "# threads - 1"
        ipcluster start --daemonize -n N

        # run the simulation
        python ./paper_simulation.py ./paper_sim_config.json -t

        # stop the workers
        ipcluster stop

3. Run the whole simulation

        # start workers in the background
        # N is the number of parallel process, often "# threads - 1"
        ipcluster start --daemonize -n N

        # run the simulation
        python ./paper_simulation.py ./paper_sim_config.json

        # stop the workers
        ipcluster stop

The results are saved in a new folder `data/<data>-<time>_five_sim_<flag_or_hash>`
containing the following files

    parameters.json  # the list of global parameters of the simulation
    arguments.json  # the list of all combinations of arguments simulated
    data.json  # the results of the simulation

Figure 1., 2., 3., and 4. from the paper are produced then by running

    python ./paper_plot_figures.py data/<data>-<time>_five_sim_<flag_or_hash>

Data
----

For the experiment, we concatenated utterances from the CMU ARCTIC speech corpus to
obtain samples of at least 15 seconds long. The dataset thus created was stored on zenodo
with DOI [10.5281/zenodo.3066488](https://zenodo.org/record/3066489). The data is automatically
retrieved upon running the scripts, but can also be manually downloaded with the `get_data.py` script.

    python ./get_data.py

It is stored in the `samples` directory.

Use FIVE
--------

Our implementation of the proposed FIVE algorithm lives in the file `five.py`.
It can be used simply like this.

    from five import five

    # STFT tensor, a numpy.ndarray with shape (frames, frequencies, channels)
    X = ...

    # perform separation, output Y has the same shape as X
    Y = five(X)

The function comes with docstrings.

    five(X, n_iter=3, proj_back=True, W0=None, model="laplace", init_eig=False,
        return_filters=False, callback=None, callback_checkpoints=[],
        cost_callback=None)

    This algorithm extracts one source independent from a minimum energy background.
    The separation is done in the time-frequency domain and the FFT length
    should be approximately equal to the reverberation time. The residual
    energy in the background is minimized.

    Two different statistical models (Laplace or time-varying Gauss) can
    be used by specifying the keyword argument `model`. The performance of Gauss
    model is higher in good conditions (few sources, low noise), but Laplace
    (the default) is more robust in general.

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the signal
    n_iter: int, optional
        The number of iterations (default 3)
    proj_back: bool, optional
        Scaling on first mic by back projection (default True)
    W0: ndarray (nfrequencies, nsrc, nchannels), optional
        Initial value for demixing matrix
    model: str
        The model of source distribution 'gauss' or 'laplace' (default)
    init_eig: bool, optional (default ``False``)
        If ``True``, and if ``W0 is None``, then the weights are initialized
        using the principal eigenvectors of the covariance matrix of the input
        data. When ``False``, the demixing matrices are initialized with identity
        matrix.
    return_filters: bool
        If true, the function will return the demixing matrix too
    callback: func
        A callback function called every 10 iterations, allows to monitor
        convergence
    callback_checkpoints: list of int
        A list of epoch number when the callback should be called
    cost_callback: func
        When this callback function is specified, it will be called with
        the value of the cost function as argument

    Returns
    -------
    Returns an (nframes, nfrequencies, 1) array. Also returns
    the demixing matrix (nfrequencies, nchannels, nsources)
    if ``return_values`` keyword is True.

Summary of the Files in this Repo
---------------------------------

    environment.yml  # anaconda environment file

    auxiva_pca.py  # implementation of AuxIVA with PCA dim reduction step
    five.py  # implementation of the proposed FIVE algorithm
    get_data.py  # script that gets the data necessary for the experiment
    ive.py  # implementation of orthogonally constrained independent vector extraction (OGIVE)
    overiva.py  # Implementation of OverIVA
    room_builder.py  # The random room generator used in the simulation
    routines.py  # contains a bunch of helper routines for the simulation

    example.py  # test file for source separation, with audible output
    paper_simulation.py  # script to run exhaustive simulation, used for the paper
    paper_sim_config.json  # simulation configuration file
    paper_plot_figures.py  # plots the figures from the paper
    paper_plot_everything.py  # plots all the output of paper_simulation.py

    data  # directory containing simulation results
    rrtools  # tools for parallel simulation
