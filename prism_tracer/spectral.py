"""Fixed spectral-to-RGB weights and data-driven dispersion/absorption."""

from __future__ import annotations

import numpy as np

RGB_RESPONSE_LOBES = ((620.0, 90.0), (560.0, 100.0), (470.0, 90.0))


class SpectralModel:
    """Maps a sampled wavelength to normalized RGB and optical properties."""

    def __init__(
        self,
        min_nm: float,
        max_nm: float,
        materials: dict[str, dict[str, np.ndarray | list[float]]],
    ) -> None:
        self.min_nm = float(min_nm)
        self.max_nm = float(max_nm)
        self._grid = np.linspace(self.min_nm, self.max_nm, 3001)
        # Smooth, strictly positive RGB response lobes.  Unlike a clipped
        # XYZ->sRGB matrix they have no zeroed wavelength intervals, so a
        # white source has low per-sample chromatic variance.  Equal-energy
        # white is normalized to (1, 1, 1) independently for every channel.
        responses = []
        for center, width in RGB_RESPONSE_LOBES:
            t = (self._grid - center) / width
            responses.append(np.exp(-0.5 * t * t))
        weights = np.column_stack(responses)
        weights /= np.mean(weights, axis=0, keepdims=True)
        self._weights = weights

        self._materials: dict[str, dict[str, np.ndarray]] = {}
        for name, material in materials.items():
            ior = material["ior"]
            sigma = material["absorption"]
            self._materials[name] = {
                "ior_nm": np.asarray(ior["nm"], dtype=np.float64),
                "ior_values": np.asarray(ior["values"], dtype=np.float64),
                "sigma_nm": np.asarray(sigma["nm"], dtype=np.float64),
                "sigma_values": np.asarray(sigma["values"], dtype=np.float64),
            }

    def rgb_weights(self, wavelength_nm: np.ndarray) -> np.ndarray:
        wavelength_nm = np.asarray(wavelength_nm, dtype=np.float64)
        return np.column_stack(
            [
                np.interp(wavelength_nm, self._grid, self._weights[:, channel])
                for channel in range(3)
            ]
        )

    def ior(self, material: str, wavelength_nm: np.ndarray) -> np.ndarray:
        data = self._materials[material]
        return np.interp(wavelength_nm, data["ior_nm"], data["ior_values"])

    def sigma(self, material: str, wavelength_nm: np.ndarray) -> np.ndarray:
        data = self._materials[material]
        return np.interp(wavelength_nm, data["sigma_nm"], data["sigma_values"])

    def white_error(self) -> np.ndarray:
        """Mean of the fixed RGB weights over the configured spectrum."""

        return np.mean(
            self.rgb_weights(np.linspace(self.min_nm, self.max_nm, 20001)), axis=0
        )

