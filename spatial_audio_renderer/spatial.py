"""Spatial preparation: assets, distance, pan, occlusion, and reverb routing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .assets import asset_sha256, generate_asset
from .dsp import (
    automatic_pan,
    db_to_gain,
    distance_gain,
    distance_vectors,
    equal_power_pan,
    maybe_lowpass,
    occlusion_cutoff,
)
from .reverb import ReverbImpulse, build_reverb_impulse
from .scene import EmitterSpec, RegionSpec, Scene


@dataclass
class PreparedEmitter:
    spec: EmitterSpec
    dry_track: np.ndarray
    send_track: Optional[np.ndarray]
    pan: float
    pan_left: float
    pan_right: float
    distance_gain: float
    occlusion_cutoff_hz: Optional[float]
    region: Optional[RegionSpec]
    lane_index: Optional[int]
    energy_weight: float
    asset_hash: str


@dataclass
class ReverbLane:
    region: RegionSpec
    impulse: ReverbImpulse
    wet_gain: float
    dry_gain: float
    emitter_indices: List[int]
    send_energy: float = 0.0

    def state(self) -> Dict[str, object]:
        return {
            "region_id": self.region.region_id,
            "rt60_seconds": self.region.reverb.rt60_seconds,
            "pre_delay_seconds": self.region.reverb.pre_delay_seconds,
            "wet": self.region.reverb.wet,
            "dry": self.region.reverb.dry,
            "impulse_sha256": self.impulse.sha256,
            "impulse_length_samples": self.impulse.length_samples,
            "pre_delay_samples": self.impulse.pre_delay_samples,
        }


@dataclass
class SpatialPreparation:
    emitters: List[PreparedEmitter]
    lanes: List[ReverbLane]
    tracker_sha256: str

    @property
    def lane_count(self) -> int:
        return len(self.lanes)


def _prepare_one_emitter(scene: Scene, emitter: EmitterSpec,
                         asset_hash: str) -> Tuple[PreparedEmitter, Optional[np.ndarray], float]:
    asset = scene.asset_for_emitter(emitter)
    source = generate_asset(asset, scene)
    distance, delta = distance_vectors(emitter.position, scene.listener_position)
    attenuation = distance_gain(distance, scene.attenuation)
    emitter_gain = db_to_gain(emitter.gain_db)
    dry = source * (emitter_gain * attenuation)

    cutoff = None
    if emitter.occlusion_amount > 0.0:
        closed_cutoff = (
            emitter.occlusion_closed_cutoff_hz
            if emitter.occlusion_closed_cutoff_hz is not None
            else scene.occlusion.closed_cutoff_hz
        )
        cutoff = occlusion_cutoff(
            emitter.occlusion_amount,
            scene.occlusion.open_cutoff_hz,
            closed_cutoff,
            scene.sample_rate,
        )
        dry = maybe_lowpass(dry, cutoff, scene.sample_rate)

    if emitter.pan is None:
        pan = automatic_pan(delta, scene.listener_forward, scene.listener_up)
    else:
        pan = emitter.pan
    pan_left, pan_right = equal_power_pan(pan)

    region = scene.region_for_emitter(emitter)
    send = None
    if region is not None and region.reverb.wet > 0.0:
        send = maybe_lowpass(dry, region.lowpass_hz, scene.sample_rate)
    energy = float(np.mean(dry.astype(np.float64) ** 2)) if dry.size else 0.0
    prepared = PreparedEmitter(
        spec=emitter,
        dry_track=dry.astype(np.float64, copy=False),
        send_track=send,
        pan=pan,
        pan_left=pan_left,
        pan_right=pan_right,
        distance_gain=attenuation,
        occlusion_cutoff_hz=cutoff,
        region=region,
        lane_index=None,
        energy_weight=1.0,
        asset_hash=asset_hash,
    )
    return prepared, send, energy


def prepare_spatial(scene: Scene) -> SpatialPreparation:
    asset_hashes: Dict[str, str] = {}
    emitters: List[PreparedEmitter] = []
    send_energies: List[float] = []
    for emitter in scene.emitters:
        if emitter.asset_id not in asset_hashes:
            asset_hashes[emitter.asset_id] = asset_sha256(
                generate_asset(scene.assets[emitter.asset_id], scene)
            )
        prepared, _, energy = _prepare_one_emitter(
            scene, emitter, asset_hashes[emitter.asset_id]
        )
        emitters.append(prepared)
        send_energies.append(energy)

    lanes_by_region: Dict[str, ReverbLane] = {}
    for index, emitter in enumerate(emitters):
        if emitter.send_track is None or emitter.region is None:
            continue
        region = emitter.region
        lane = lanes_by_region.get(region.region_id)
        if lane is None:
            impulse = build_reverb_impulse(region.reverb, scene.sample_rate)
            lane = ReverbLane(
                region=region,
                impulse=impulse,
                wet_gain=region.reverb.wet,
                dry_gain=region.reverb.dry,
                emitter_indices=[],
            )
            lanes_by_region[region.region_id] = lane
        lane.emitter_indices.append(index)

    lanes = [lanes_by_region[region.region_id] for region in scene.regions
             if region.region_id in lanes_by_region]
    lane_index_by_region = {lane.region.region_id: index for index, lane in enumerate(lanes)}

    for index, emitter in enumerate(emitters):
        if emitter.region is None or emitter.send_track is None:
            continue
        emitter.lane_index = lane_index_by_region[emitter.region.region_id]
        lanes[emitter.lane_index].send_energy += send_energies[index]

    for lane in lanes:
        if not lane.emitter_indices:
            continue
        mean_energy = lane.send_energy / float(len(lane.emitter_indices))
        for emitter_index in lane.emitter_indices:
            emitter = emitters[emitter_index]
            if mean_energy > 0.0 and send_energies[emitter_index] > 0.0:
                emitter.energy_weight = float(
                    np.sqrt(send_energies[emitter_index] / mean_energy)
                )
            else:
                emitter.energy_weight = 1.0

    tracker_hasher = hashlib.sha256()
    for emitter in emitters:
        tracker_hasher.update(emitter.spec.emitter_id.encode("utf-8"))
        tracker_hasher.update(np.ascontiguousarray(emitter.dry_track).tobytes())
    tracker_sha256 = tracker_hasher.hexdigest()

    return SpatialPreparation(emitters, lanes, tracker_sha256)


def emitter_state(prepared: PreparedEmitter, next_sample_frame: int) -> Dict[str, object]:
    return {
        "id": prepared.spec.emitter_id,
        "asset_id": prepared.spec.asset_id,
        "asset_sha256": prepared.asset_hash,
        "next_sample_frame": next_sample_frame,
        "distance_gain": prepared.distance_gain,
        "pan": prepared.pan,
        "occlusion_cutoff_hz": prepared.occlusion_cutoff_hz,
        "region_id": prepared.region.region_id if prepared.region else None,
        "energy_weight": prepared.energy_weight,
    }
