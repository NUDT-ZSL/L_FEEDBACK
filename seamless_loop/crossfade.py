"""Equal-power sine/cosine crossfade and seamless loop construction.

Given a loop region ``r`` of ``N`` samples split into a *head* ``H`` and a
*tail* ``T`` of ``F`` samples each plus a middle ``M``::

    r = [ H (F) ][ M (N-2F) ][ T (F) ]

the raw wrap discontinuity is the jump ``T[-1] -> H[0]`` that would appear
when the region is glued end to start.  The seamless loop absorbs that jump
inside an equal-power crossfade ``Z`` of length ``F``::

    Z[k] = T[k] * cos(pi/2 * p) + H[k] * sin(pi/2 * p),  p = k / (F - 1)

and the produced loop of length ``N - F`` is::

    loop = [ M ][ Z ]

Properties guaranteed by the normalised windows (endpoints pinned to exact
0.0/1.0):

* ``cos(p)**2 + sin(p)**2 == 1`` -> constant signal power across the whole
  crossfade (equal power, *not* equal amplitude);
* the ``M -> Z`` junction joins source-adjacent samples ``r[N-F-1] -> r[N-F]``
  and the periodic seam ``Z[-1] -> M[0]`` joins source-adjacent samples
  ``r[F-1] -> r[F]``; both are ordinary within-signal steps, so no click is
  introduced and the original wrap jump is fully smoothed inside ``Z``.
"""

import logging

import numpy as np

from .errors import ParameterRangeError

logger = logging.getLogger("seamless_loop")


def equal_power_windows(fade_len):
    """Return ``(fade_out, fade_in)`` windows of ``fade_len`` samples.

    The windows use the normalised phase ``k / (fade_len - 1)`` so that, for
    ``fade_len >= 2``, they land on exact 0.0 and 1.0 values at both ends.
    They satisfy ``fade_out**2 + fade_in**2 == 1`` everywhere, i.e. constant
    power.

    Parameters
    ----------
    fade_len:
        Crossfade length in samples (must be >= 1).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
    ``(tail_weight, head_weight)`` float64 arrays of shape ``(fade_len,)``.
    ``tail_weight`` is a cosine window 1.0 -> 0.0 and ``head_weight`` is a
    sine window 0.0 -> 1.0; their squared values sum to 1 everywhere.
    """
    if fade_len < 1:
        raise ParameterRangeError(
            "crossfade length must be at least 1 sample, got %d" % fade_len
        )
    if fade_len == 1:
        # Degenerate single-sample crossfade: keep the head value.
        return (
            np.zeros(1, dtype=np.float64),
            np.ones(1, dtype=np.float64),
        )
    phase = np.arange(fade_len, dtype=np.float64) / (fade_len - 1)
    tail_weight = np.cos(0.5 * np.pi * phase)
    head_weight = np.sin(0.5 * np.pi * phase)
    # Pin the endpoints to exact values to remove floating point drift.
    tail_weight[0], tail_weight[-1] = 1.0, 0.0
    head_weight[0], head_weight[-1] = 0.0, 1.0
    return tail_weight, head_weight


def build_loop(region, fade_len, track_name="track"):
    """Build one seamless, periodic loop from a sliced loop region.

    Parameters
    ----------
    region:
        ``numpy.ndarray`` of shape ``(frames,)`` or ``(frames, channels)``
        holding the samples inside the configured loop region.
    fade_len:
        Requested crossfade length in samples.  When the region does not
        contain enough samples for a full crossfade (a region needs at least
        ``2 * fade_len`` samples), the length is dynamically shortened to
        ``frames // 2`` and the adjustment is logged.
    track_name:
        Track label used in the adjustment log message.

    Returns
    -------
        tuple[numpy.ndarray, int]
        The seamless loop (length ``frames - fade``) and the crossfade length
        actually used, in samples.
    """
    region = np.asarray(region, dtype=np.float64)
    frames = region.shape[0]
    if frames < 2:
        raise ParameterRangeError(
            "track %r loop region has only %d frame(s); at least 2 are needed"
            % (track_name, frames)
        )

    max_fade = frames // 2
    effective_fade = int(min(fade_len, max_fade))
    if effective_fade < 1:
        raise ParameterRangeError(
            "track %r cannot fit even a 1-sample crossfade in %d frames"
            % (track_name, frames)
        )
    if effective_fade != fade_len:
        logger.warning(
            "track %r: loop region (%d samples) cannot fit a %d-sample "
            "crossfade; shortening crossfade to %d samples",
            track_name, frames, fade_len, effective_fade,
        )

    tail_weight, head_weight = equal_power_windows(effective_fade)
    if region.ndim == 2:
        tail_weight = tail_weight[:, np.newaxis]
        head_weight = head_weight[:, np.newaxis]

    head = region[:effective_fade]
    tail = region[frames - effective_fade:]
    middle = region[effective_fade:frames - effective_fade]
    # Equal-power crossfade absorbs the raw tail->head wrap discontinuity.
    crossfade_section = tail * tail_weight + head * head_weight
    # Middle first, crossfade last.  Both junctions are source-adjacent steps.
    loop = np.concatenate([middle, crossfade_section], axis=0)
    return loop, effective_fade
