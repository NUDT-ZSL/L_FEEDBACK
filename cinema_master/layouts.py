"""Canonical channel layouts, WAV masks, and mapping helpers."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .errors import MasterError


CHANNEL_BITS: Dict[str, int] = {
    "FL": 0x00000001,
    "FR": 0x00000002,
    "FC": 0x00000004,
    "LFE": 0x00000008,
    "BL": 0x00000010,
    "BR": 0x00000020,
    "FLC": 0x00000040,
    "FRC": 0x00000080,
    "BC": 0x00000100,
    "SL": 0x00000200,
    "SR": 0x00000400,
    "TC": 0x00000800,
    "TFL": 0x00001000,
    "TFC": 0x00002000,
    "TFR": 0x00004000,
    "TBL": 0x00008000,
    "TBC": 0x00010000,
    "TBR": 0x00020000,
}

BIT_CHANNELS: Dict[int, str] = {bit: name for name, bit in CHANNEL_BITS.items()}

LAYOUTS: Dict[str, List[str]] = {
    "mono": ["FC"],
    "stereo": ["FL", "FR"],
    "2.0": ["FL", "FR"],
    "5.1": ["FL", "FR", "FC", "LFE", "BL", "BR"],
    "5.1(side)": ["FL", "FR", "FC", "LFE", "SL", "SR"],
}

LAYOUT_MASKS: Dict[str, int] = {
    "mono": CHANNEL_BITS["FC"],
    "stereo": CHANNEL_BITS["FL"] | CHANNEL_BITS["FR"],
    "2.0": CHANNEL_BITS["FL"] | CHANNEL_BITS["FR"],
    "5.1": CHANNEL_BITS["FL"]
    | CHANNEL_BITS["FR"]
    | CHANNEL_BITS["FC"]
    | CHANNEL_BITS["LFE"]
    | CHANNEL_BITS["BL"]
    | CHANNEL_BITS["BR"],
    "5.1(side)": CHANNEL_BITS["FL"]
    | CHANNEL_BITS["FR"]
    | CHANNEL_BITS["FC"]
    | CHANNEL_BITS["LFE"]
    | CHANNEL_BITS["SL"]
    | CHANNEL_BITS["SR"],
}

ROLE_ALIASES: Dict[str, str] = {
    "L": "FL",
    "R": "FR",
    "C": "FC",
    "M": "FC",
    "MONO": "FC",
    "LS": "SL",
    "RS": "SR",
    "LB": "BL",
    "RB": "BR",
    "SIDELEFT": "SL",
    "SIDERIGHT": "SR",
    "REARLEFT": "BL",
    "REARRIGHT": "BR",
    "LOWFREQUENCY": "LFE",
}


def normalize_role(value: Any, *, allow_none: bool = False) -> Optional[str]:
    if value is None or value == "":
        if allow_none:
            return None
        raise MasterError(
            "E_CONFIG_SCHEMA", "A channel role must not be empty"
        )
    if not isinstance(value, str):
        raise MasterError(
            "E_CONFIG_SCHEMA",
            "Channel roles must be strings",
            {"value": repr(value)},
        )
    role = value.strip().upper().replace(" ", "").replace("_", "")
    role = ROLE_ALIASES.get(role, role)
    if role not in CHANNEL_BITS:
        raise MasterError(
            "E_CONFIG_SCHEMA",
            "Unsupported channel role",
            {"role": value, "supported": sorted(CHANNEL_BITS)},
        )
    return role


def roles_from_mask(mask: int, expected_channels: Optional[int] = None) -> List[str]:
    roles = [BIT_CHANNELS[bit] for bit in sorted(BIT_CHANNELS) if mask & bit]
    if expected_channels is not None and len(roles) != expected_channels:
        raise MasterError(
            "E_CHANNEL_MISMATCH",
            "WAV channel mask does not match the number of channels",
            {
                "channel_mask": _format_mask(mask),
                "mask_channels": len(roles),
                "file_channels": expected_channels,
                "mask_roles": roles,
            },
        )
    return roles


def mask_for_roles(roles: Sequence[str]) -> int:
    mask = 0
    for role in roles:
        mask |= CHANNEL_BITS[role]
    return mask


def default_roles_for_count(channels: int) -> Tuple[List[str], str]:
    if channels == 1:
        return ["FC"], "implicit_mono"
    if channels == 2:
        return ["FL", "FR"], "implicit_stereo"
    if channels == 4:
        return ["FL", "FR", "BL", "BR"], "implicit_quad"
    if channels == 6:
        return ["FL", "FR", "FC", "LFE", "BL", "BR"], "implicit_5.1"
    if channels == 8:
        return ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"], "implicit_7.1"
    raise MasterError(
        "E_CHANNEL_MISMATCH",
        "A multichannel PCM WAV without a channel mask needs an explicit channel map",
        {"file_channels": channels},
    )


def resolve_output_layout(value: Any) -> Tuple[str, List[str], int]:
    if isinstance(value, str):
        key = value.strip()
        lookup = {k.lower(): k for k in LAYOUTS}
        normalized = lookup.get(key.lower())
        if normalized is None:
            raise MasterError(
                "E_CONFIG_SCHEMA",
                "Unsupported output layout",
                {"layout": value, "supported": ["5.1", "stereo"]},
            )
        if normalized not in ("5.1", "stereo", "2.0"):
            raise MasterError(
                "E_CONFIG_SCHEMA",
                "Only 5.1 and stereo output layouts are supported",
                {"layout": value},
            )
        name = "stereo" if normalized in ("stereo", "2.0") else "5.1"
        roles = list(LAYOUTS[normalized])
        return name, roles, LAYOUT_MASKS[normalized]

    if isinstance(value, dict):
        name = str(value.get("name", "custom"))
        raw_channels = value.get("channels")
        if not isinstance(raw_channels, list):
            raise MasterError(
                "E_CONFIG_SCHEMA",
                "output_layout.channels must be a list",
            )
        roles = [normalize_role(item) for item in raw_channels]
        assert all(role is not None for role in roles)
        return name, list(roles), mask_for_roles(list(roles))  # type: ignore[arg-type]

    if isinstance(value, list):
        roles = [normalize_role(item) for item in value]
        assert all(role is not None for role in roles)
        if roles == ["FL", "FR"]:
            name = "stereo"
        elif roles == LAYOUTS["5.1"]:
            name = "5.1"
        elif roles == LAYOUTS["5.1(side)"]:
            name = "5.1(side)"
        else:
            name = "custom"
        return name, list(roles), mask_for_roles(list(roles))  # type: ignore[arg-type]

    raise MasterError(
        "E_CONFIG_SCHEMA",
        "output_layout must be a string or a list of channel roles",
    )


def _downmix_coefficients(role: str) -> Dict[str, float]:
    c = 1.0 / math.sqrt(2.0)
    table: Dict[str, Dict[str, float]] = {
        "FL": {"FL": 1.0},
        "FR": {"FR": 1.0},
        "FC": {"FL": c, "FR": c},
        "LFE": {},
        "BL": {"FL": c},
        "BR": {"FR": c},
        "SL": {"FL": c},
        "SR": {"FR": c},
        "FLC": {"FL": 1.0},
        "FRC": {"FR": 1.0},
        "BC": {"FL": c, "FR": c},
    }
    return table.get(role, {})


def mapping_matrix(
    source_roles: Sequence[str],
    output_roles: Sequence[str],
    output_layout_name: str,
) -> Tuple[Any, Dict[str, Any]]:
    """Build float coefficients from source channel order to output order."""

    import numpy as np

    matrix = np.zeros((len(source_roles), len(output_roles)), dtype=np.float64)
    mapped: List[Dict[str, Any]] = []
    dropped: List[str] = []
    is_stereo = list(output_roles) == ["FL", "FR"]

    for src_index, role in enumerate(source_roles):
        contributions: Dict[str, float] = {}
        if is_stereo:
            contributions = _downmix_coefficients(role)
        else:
            target = role
            if target not in output_roles:
                if role == "SL" and "BL" in output_roles:
                    target = "BL"
                elif role == "SR" and "BR" in output_roles:
                    target = "BR"
                elif role == "BL" and "SL" in output_roles:
                    target = "SL"
                elif role == "BR" and "SR" in output_roles:
                    target = "SR"
            if target in output_roles:
                contributions[target] = 1.0

        if not contributions:
            dropped.append(role)
        for target, coefficient in contributions.items():
            matrix[src_index, output_roles.index(target)] += coefficient
        mapped.append(
            {
                "source_channel": src_index,
                "source_role": role,
                "outputs": {
                    target: coefficient
                    for target, coefficient in contributions.items()
                },
            }
        )

    details = {
        "layout": output_layout_name,
        "downmix_standard": "ITU-R BS.775"
        if is_stereo
        else None,
        "centre_and_surround_coefficient": 1.0 / math.sqrt(2.0)
        if is_stereo
        else None,
        "lfe_in_stereo_downmix": False if is_stereo else None,
        "mapped_channels": mapped,
        "dropped_roles": dropped,
    }
    return matrix, details


def channel_weights(roles: Sequence[str]) -> List[float]:
    weights = []
    for role in roles:
        if role == "LFE":
            weights.append(0.0)
        elif role in ("BL", "BR", "SL", "SR"):
            weights.append(1.41)
        else:
            weights.append(1.0)
    return weights


def _format_mask(mask: int) -> str:
    return "0x{:08x}".format(int(mask) & 0xFFFFFFFF)


def format_mask(mask: int) -> str:
    return _format_mask(mask)

