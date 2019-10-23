import itertools
import math
from typing import List, Optional, Tuple

import numpy as np
import pyroomacoustics as pra
from numpy.random import rand


def callback_noise_mixer(
    premix,
    sinr=0,
    diffuse_ratio=0,
    ref_mic=0,
    n_src=None,
    n_tgt=None,
    tgt_std=None,
):
    """
    This callback function will rescale all the signals so that the SINR
    is fixed to a given value with a given ratio of diffuse noise
    """

    # first normalize all separate recording to have unit power at microphone one
    p_mic_ref = np.std(premix[:, ref_mic, :], axis=1)
    premix /= p_mic_ref[:, None, None]
    premix[:n_tgt, :, :] *= tgt_std[:, None, None]

    # Total variance of noise components
    var_noise_tot = 10 ** (-sinr / 10) * np.sum(tgt_std ** 2)

    # compute noise variance
    sigma_n = np.sqrt((1 - diffuse_ratio) * var_noise_tot)

    # now compute the power of interference signal needed to achieve desired SINR
    sigma_i = np.sqrt((diffuse_ratio / (n_src - n_tgt)) * var_noise_tot)
    premix[n_tgt:n_src, :, :] *= sigma_i

    # the background
    bg = np.sum(premix[n_tgt:n_src, :, :], axis=0)
    bg += sigma_n * np.random.randn(*premix.shape[1:])

    # Mix down the recorded signals
    mix = np.sum(premix[:n_tgt, :], axis=0) + bg

    return mix


def inv_sabine(t60, room_dim, c):
    """
    given desired t60, (shoebox) room dimension and sound speed,
    computes the reflection coefficient (amplitude) and image source
    order needed. the speed of sound used is the package wide default
    (in :py:data:`pyroomacoustics.parameters.constants`).

    parameters
    ----------
    t60: float
        desired t60 (time it takes to go from full amplitude to 60 db decay) in seconds
    room_dim: list of floats
        list of length 2 or 3 of the room side lengths
    c: float
        speed of sound

    returns
    -------
    reflection: float
        the reflection coefficient (in amplitude domain, to be passed to
        room constructor)
    max_order: int
        the maximum image source order necessary to achieve the desired t60
    """

    # finding image sources up to a maximum order creates a (possibly 3d) diamond
    # like pile of (reflected) rooms. now we need to find the image source model order
    # so that reflections at a distance of at least up to ``c * rt60`` are included.
    # one possibility is to find the largest sphere (or circle in 2d) that fits in the
    # diamond. this is what we are doing here.
    R = []
    for l1, l2 in itertools.combinations(room_dim, 2):
        R.append(l1 * l2 / np.sqrt(l1 ** 2 + l2 ** 2))

    V = np.prod(room_dim)  # area (2d) or volume (3d)
    # "surface" computation is diff for 2d and 3d
    if len(room_dim) == 2:
        S = 2 * np.sum(room_dim)
        sab_coef = 12  # the sabine's coefficient needs to be adjusted in 2d
    elif len(room_dim) == 3:
        S = 2 * np.sum([l1 * l2 for l1, l2 in itertools.combinations(room_dim, 2)])
        sab_coef = 24

    a2 = sab_coef * np.log(10) * V / (c * S * t60)  # absorption in power (sabine)

    if a2 > 1.0:
        raise ValueError(
            "evaluation of parameters failed. room may be too large for required t60."
        )

    reflection = np.sqrt(1 - a2)  # convert to reflection coefficient

    max_order = math.ceil(c * t60 / np.min(R) - 1)

    return reflection, max_order


def random_room_builder(
    source_signals: List[np.ndarray],
    n_mics: int,
    mic_delta: Optional[float] = None,
    fs: float = 16000,
    t60_interval: Tuple[float, float] = (0.150, 0.500),
    room_width_interval: Tuple[float, float] = (6, 10),
    room_height_interval: Tuple[float, float] = (2.8, 4.5),
    source_zone_height: Tuple[float, float] = [1.0, 2.0],
    guard_zone_width: float = 0.5,
    seed: Optional[int] = None,
):
    """
    This function creates a random room within some parameters.

    The microphone array is circular with the distance between neighboring
    elements set to the maximal distance avoiding spatial aliasing.

    Parameters
    ----------
    source_signals: list of numpy.ndarray
        A list of audio signals for each source
    n_mics: int
        The number of microphones in the microphone array
    mic_delta: float, optional
        The distance between neighboring microphones in the array
    fs: float, optional
        The sampling frequency for the simulation
    t60_interval: (float, float), optional
        An interval where to pick the reverberation time
    room_width_interval: (float, float), optional
        An interval where to pick the room horizontal length/width
    room_height_interval: (float, float), optional
        An interval where to pick the room vertical length
    source_zone_height: (float, float), optional
        The vertical interval where sources and microphones are allowed
    guard_zone_width: float
        The minimum distance between a vertical wall and a source/microphone

    Returns
    -------
    ShoeBox object
        A randomly generated room according to the provided parameters
    float
        The measured T60 reverberation time of the room created
    """

    # save current numpy RNG state and set a known seed
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)

    n_sources = len(source_signals)

    # sanity checks
    assert source_zone_height[0] > 0
    assert source_zone_height[1] < room_height_interval[0]
    assert source_zone_height[0] <= source_zone_height[1]
    assert 2 * guard_zone_width < room_width_interval[1] - room_width_interval[0]

    def random_location(
        room_dim, n, ref_point=None, min_distance=None, max_distance=None
    ):
        """ Helper function to pick a location in the room """

        width = room_dim[0] - 2 * guard_zone_width
        width_intercept = guard_zone_width

        depth = room_dim[1] - 2 * guard_zone_width
        depth_intercept = guard_zone_width

        height = np.diff(source_zone_height)[0]
        height_intercept = source_zone_height[0]

        locs = rand(3, n)
        locs[0, :] = locs[0, :] * width + width_intercept
        locs[1, :] = locs[1, :] * depth + depth_intercept
        locs[2, :] = locs[2, :] * height + height_intercept

        if ref_point is not None:
            # Check condition
            d = np.linalg.norm(locs - ref_point, axis=0)

            if min_distance is not None and max_distance is not None:
                redo = np.where(np.logical_or(d < min_distance, max_distance < d))[0]
            elif min_distance is not None:
                redo = np.where(d < min_distance)[0]
            elif max_distance is not None:
                redo = np.where(d > max_distance)[0]
            else:
                redo = []

            # Recursively call this function on sources to redraw
            if len(redo) > 0:
                locs[:, redo] = random_location(
                    room_dim,
                    len(redo),
                    ref_point=ref_point,
                    min_distance=min_distance,
                    max_distance=max_distance,
                )

        return locs

    c = pra.constants.get("c")

    # Create the room
    # Sometimes the room dimension and required T60 are not compatible, then
    # we just try again with new random values
    retry = True
    while retry:
        try:
            room_dim = np.array(
                [
                    rand() * np.diff(room_width_interval)[0] + room_width_interval[0],
                    rand() * np.diff(room_width_interval)[0] + room_width_interval[0],
                    rand() * np.diff(room_height_interval)[0] + room_height_interval[0],
                ]
            )
            t60 = rand() * np.diff(t60_interval)[0] + t60_interval[0]
            reflection, max_order = inv_sabine(t60, room_dim, c)
            retry = False
        except ValueError:
            pass
    # Create the room based on the random parameters
    room = pra.ShoeBox(room_dim, fs=fs, absorption=1 - reflection, max_order=max_order)

    # The critical distance
    # https://en.wikipedia.org/wiki/Critical_distance
    d_critical = 0.057 * np.sqrt(np.prod(room_dim) / t60)

    # default intermic distance is set according to nyquist criterion
    # i.e. 1/2 wavelength corresponding to fs / 2 at given speed of sound
    if mic_delta is None:
        mic_delta = 0.5 * (c / (0.5 * fs))

    # the microphone array is uniformly circular with the distance between
    # neighboring elements of mic_delta
    mic_center = random_location(room_dim, 1)
    mic_rotation = rand() * 2 * np.pi
    mic_radius = 0.5 * mic_delta / np.sin(np.pi / n_mics)
    mic_array = pra.MicrophoneArray(
        np.vstack(
            (
                pra.circular_2D_array(
                    mic_center[:2, 0], n_mics, mic_rotation, mic_radius
                ),
                mic_center[2, 0] * np.ones(n_mics),
            )
        ),
        room.fs,
    )
    room.add_microphone_array(mic_array)

    # Now we will get the sources at random
    source_locs = []

    # Choose the target location at least as far as the critical distance
    # Then the other sources, yet one further meter away
    target_source = random_location(
        room_dim,
        1,
        ref_point=mic_center,
        min_distance=d_critical,
        max_distance=d_critical + 1,
    )
    interferers = random_location(
        room_dim, n_sources - 1, ref_point=mic_center, min_distance=d_critical + 1
    )
    source_locs = np.concatenate((target_source, interferers), axis=1)

    for s, signal in enumerate(source_signals):
        room.add_source(source_locs[:, s], signal=signal)

    # pre-compute the impulse responses
    room.compute_rir()
    t60_actual = pra.experimental.measure_rt60(room.rir[0][0], room.fs)

    # restore numpy RNG former state
    if seed is not None:
        np.random.set_state(rng_state)

    return room, t60_actual
