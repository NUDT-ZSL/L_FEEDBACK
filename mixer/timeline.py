"""Timeline ordering and overlap validation."""

from __future__ import annotations

from dataclasses import dataclass

from .config import ShotSpec
from .errors import ErrorCode, MixerError


@dataclass(frozen=True)
class Overlap:
    earlier: ShotSpec
    later: ShotSpec
    start_sample: int
    end_sample: int

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample


def order_shots(selected: list[ShotSpec]) -> list[ShotSpec]:
    indexed = list(enumerate(selected))
    indexed.sort(key=lambda item: (item[1].start_sample, item[0]))
    return [shot for _, shot in indexed]


def validate_and_order(selected: list[ShotSpec]) -> tuple[list[ShotSpec], list[Overlap]]:
    """Order selected shots and reject undeclared overlap.

    A later shot may overlap an earlier one when either side declares overlap
    (``allow_overlap`` or a positive crossfade duration). Validation is
    intentionally based on the chosen path only; unused branch content must not
    make another branch fail.
    """

    ordered = order_shots(selected)
    overlaps: list[Overlap] = []
    accepted: list[ShotSpec] = []

    for shot in ordered:
        for earlier in accepted:
            if shot.start_sample >= earlier.end_sample:
                continue
            if not shot.allow_overlap and not earlier.allow_overlap:
                raise MixerError(
                    ErrorCode.E_TIMELINE_CONFLICT,
                    (
                        f"{shot.shot_id} overlaps {earlier.shot_id} without an "
                        "overlap/crossfade declaration"
                    ),
                    context={
                        "partial_shot_ids": [item.shot_id for item in accepted],
                        "conflict_shots": [earlier.shot_id, shot.shot_id],
                    },
                )
            overlap_start = shot.start_sample
            overlap_end = min(shot.end_sample, earlier.end_sample)
            if overlap_end > overlap_start:
                overlaps.append(
                    Overlap(
                        earlier=earlier,
                        later=shot,
                        start_sample=overlap_start,
                        end_sample=overlap_end,
                    )
                )
        accepted.append(shot)

    return ordered, overlaps
