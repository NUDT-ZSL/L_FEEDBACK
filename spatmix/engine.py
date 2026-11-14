"""Frame-based deterministic spatial mixing engine.

The engine advances in fixed-size frames. Each frame it samples the
listener trajectory, updates per-source geometry, applies doppler
resampling, early reflections, air absorption, occlusion filtering and
distance gain, pans into the output layout with crossfaded VBAP gains,
and only then appends the finished frame to the raw mix buffer.

All mutable state lives in plain float64 scalars/arrays and is fully
captured by snapshots, so resuming from a snapshot is bit-identical
to uninterrupted processing.
"""

import json
import math
import os

import numpy as np

from . import EVENT_LOG_VERSION
from .dsp import DelayLine, OnePoleLowpass, Resampler, cutoff_to_coef
from .geometry import (
    distance,
    listener_state_at,
    mirror_across_plane,
    point_in_aabb,
    segment_intersects_aabb,
)
from .panning import num_channels, vbap_gains
from .synth import DrySource

RAW_NAME = "mix.rawf32"
EVENTS_NAME = "events.jsonl"


class SourceState:
    """Runtime state for one source (fully snapshot-able)."""

    def __init__(self, spec, scene):
        self.spec = spec
        self.dry = DrySource(spec, scene["sample_rate"])
        self.resampler = Resampler()
        self.air_lp = OnePoleLowpass()
        self.occ_lp = OnePoleLowpass()
        self.reflections = []
        self.pan_prev = None
        self.d_prev = None
        self.doppler_factor = 1.0
        self.doppler_clamped = False
        self.audible = False
        self.occlusion = 0.0
        self.occlusion_degenerate = False

    def state_dict(self):
        return {
            "dry": self.dry.state_dict(),
            "resampler": self.resampler.state_dict(),
            "air_lp": self.air_lp.state_dict(),
            "occ_lp": self.occ_lp.state_dict(),
            "reflections": [r.state_dict() for r in self.reflections],
            "pan_prev": None if self.pan_prev is None else self.pan_prev.tolist(),
            "d_prev": self.d_prev,
            "doppler_factor": self.doppler_factor,
            "doppler_clamped": self.doppler_clamped,
            "audible": self.audible,
            "occlusion": self.occlusion,
            "occlusion_degenerate": self.occlusion_degenerate,
        }

    def load_state(self, state):
        self.dry.load_state(state["dry"])
        self.resampler.load_state(state["resampler"])
        self.air_lp.load_state(state["air_lp"])
        self.occ_lp.load_state(state["occ_lp"])
        self.reflections = [DelayLine.from_state(s) for s in state["reflections"]]
        pan = state["pan_prev"]
        self.pan_prev = None if pan is None else np.array(pan, dtype=np.float64)
        self.d_prev = state["d_prev"]
        self.doppler_factor = float(state["doppler_factor"])
        self.doppler_clamped = bool(state["doppler_clamped"])
        self.audible = bool(state["audible"])
        self.occlusion = float(state["occlusion"])
        self.occlusion_degenerate = bool(state["occlusion_degenerate"])


class Engine:
    def __init__(self, scene, out_dir, raw_mode="wb"):
        self.scene = scene
        self.out_dir = out_dir
        self.sr = scene["sample_rate"]
        self.frame_size = scene["frame_size"]
        self.layout = scene["layout"]
        self.channels = num_channels(self.layout)
        self.cfg = scene["config"]
        self.total_frames = int(
            math.ceil(scene["duration"] * self.sr / self.frame_size)
        )
        self.frames_completed = 0
        os.makedirs(out_dir, exist_ok=True)
        self.raw_path = os.path.join(out_dir, RAW_NAME)
        self.events_path = os.path.join(out_dir, EVENTS_NAME)
        self._raw = open(self.raw_path, raw_mode)
        self._events_file = open(self.events_path, "a", encoding="utf-8")
        self.sources = [SourceState(s, scene) for s in scene["sources"]]
        self._material_alpha = {}

    def emit(self, **event):
        event.setdefault("frame", self.frames_completed)
        event.setdefault("t", self.frames_completed * self.frame_size / self.sr)
        self._events_file.write(json.dumps(event, sort_keys=True) + "\n")
        self._events_file.flush()

    def initialize(self):
        """One-time setup for a fresh run (frame 0)."""
        self.emit(type="header", log_version=EVENT_LOG_VERSION,
                  layout=self.layout, sample_rate=self.sr,
                  frame_size=self.frame_size, channels=self.channels)
        listener_pos, _ = listener_state_at(self.scene["listener"]["keyframes"], 0.0)
        for src in self.sources:
            spec = src.spec
            label = spec["material"]
            if label == "default":
                alpha = self.cfg["air_absorption"]
            elif label in self.scene["materials"]:
                alpha = self.scene["materials"][label]["air_absorption"]
            else:
                alpha = self.cfg["air_absorption"]
                self.emit(type="unknown_material", source=spec["id"],
                          material=label, fallback="default")
            self._material_alpha[spec["id"]] = alpha
            self._setup_reflections(src, listener_pos)

    def _setup_reflections(self, src, listener_pos):
        spec = src.spec
        c = self.cfg["speed_of_sound"]
        ref = self.cfg["reference_distance"]
        for refl in self.scene["reflectors"]:
            img = mirror_across_plane(spec["pos"], refl["axis"], refl["value"])
            d_img = distance(img, listener_pos)
            delay_samples = int(round(d_img / c * self.sr))
            gain = (1.0 - refl["absorption"]) * (ref / max(d_img, ref)) ** 2
            if delay_samples < 1:
                self.emit(type="reflection_skipped", source=spec["id"],
                          reflector=refl["id"], reason="degenerate_image")
                continue
            if gain < self.cfg["min_reflection_gain"]:
                self.emit(type="reflection_skipped", source=spec["id"],
                          reflector=refl["id"], reason="too_weak")
                continue
            src.reflections.append(DelayLine(delay_samples, gain))
            self.emit(type="reflection_added", source=spec["id"],
                      reflector=refl["id"], delay_samples=delay_samples,
                      gain=gain)

    def restore_materials(self):
        """Rebuild the material map on resume (no events re-emitted)."""
        for src in self.sources:
            label = src.spec["material"]
            if label != "default" and label in self.scene["materials"]:
                alpha = self.scene["materials"][label]["air_absorption"]
            else:
                alpha = self.cfg["air_absorption"]
            self._material_alpha[src.spec["id"]] = alpha

    def load_snapshot_state(self, state):
        self.frames_completed = int(state["frames_completed"])
        by_id = {s.spec["id"]: s for s in self.sources}
        for sid, sstate in state["sources"].items():
            by_id[sid].load_state(sstate)

    def process_frame(self):
        frame_idx = self.frames_completed
        t = frame_idx * self.frame_size / self.sr
        dt = self.frame_size / self.sr
        listener_pos, heading = listener_state_at(
            self.scene["listener"]["keyframes"], t)
        mix = np.zeros((self.frame_size, self.channels), dtype=np.float64)
        for src in self.sources:
            mono = self._source_frame(src, listener_pos, dt)
            azimuth = self._relative_azimuth(src.spec["pos"], listener_pos, heading)
            gains = vbap_gains(azimuth, self.layout)
            if src.pan_prev is None:
                src.pan_prev = gains.copy()
            ramp = np.linspace(0.0, 1.0, self.frame_size)[:, None]
            frame_gains = src.pan_prev[None, :] * (1.0 - ramp) + gains[None, :] * ramp
            mix += mono[:, None] * frame_gains
            src.pan_prev = gains
        self._raw.write(mix.astype("<f4").tobytes())
        self.frames_completed += 1

    def _relative_azimuth(self, pos, listener_pos, heading):
        dx = pos[0] - listener_pos[0]
        dy = pos[1] - listener_pos[1]
        az_world = math.degrees(math.atan2(dx, dy))
        return az_world - heading

    def _source_frame(self, src, listener_pos, dt):
        spec = src.spec
        cfg = self.cfg
        pos = spec["pos"]
        d = distance(pos, listener_pos)

        audible = d <= cfg["max_audible_distance"]
        if audible and not src.audible:
            self.emit(type="source_audible", source=spec["id"], distance=d)
        elif not audible and src.audible:
            self.emit(type="source_inaudible", source=spec["id"], distance=d)
        src.audible = audible

        d_prev = src.d_prev if src.d_prev is not None else d
        v_radial = (d - d_prev) / dt
        src.d_prev = d
        c = cfg["speed_of_sound"]
        target = c / (c + v_radial)
        clamped = min(max(target, cfg["doppler_min"]), cfg["doppler_max"])
        is_clamped = clamped != target
        if is_clamped and not src.doppler_clamped:
            self.emit(type="doppler_clamp_start", source=spec["id"],
                      raw_factor=target, clamped_factor=clamped)
        elif not is_clamped and src.doppler_clamped:
            self.emit(type="doppler_clamp_end", source=spec["id"])
        src.doppler_clamped = is_clamped
        tau = cfg["doppler_smoothing"]
        beta = 1.0 - math.exp(-dt / tau) if tau > 0 else 1.0
        src.doppler_factor += beta * (clamped - src.doppler_factor)

        x = src.resampler.process(src.dry, self.frame_size, src.doppler_factor)
        keep_from = int(math.floor(src.resampler.phase)) - 1
        src.dry.discard_before(keep_from)

        mono = x
        for delay in src.reflections:
            mono = mono + delay.process(x)

        ref = cfg["reference_distance"]
        dist_gain = (ref / max(d, ref)) ** 2
        alpha = self._material_alpha[spec["id"]]
        fc_air = min(0.45 * self.sr * math.exp(-alpha * d), 0.45 * self.sr)
        fc_air = max(fc_air, 100.0)
        mono = src.air_lp.process(mono, cutoff_to_coef(fc_air, self.sr))

        occlusion, degenerate = self._occlusion(pos, listener_pos)
        if occlusion > 0.0 and src.occlusion == 0.0:
            self.emit(type="occlusion_start", source=spec["id"],
                      occlusion=occlusion, degenerate=degenerate)
        elif occlusion == 0.0 and src.occlusion > 0.0:
            self.emit(type="occlusion_end", source=spec["id"])
        if degenerate and not src.occlusion_degenerate:
            self.emit(type="occlusion_degenerate_start", source=spec["id"])
        elif not degenerate and src.occlusion_degenerate:
            self.emit(type="occlusion_degenerate_end", source=spec["id"])
        src.occlusion = occlusion
        src.occlusion_degenerate = degenerate
        if occlusion > 0.0:
            fc_occ = max(200.0, fc_air * (1.0 - 0.95 * occlusion))
            mono = src.occ_lp.process(mono, cutoff_to_coef(fc_occ, self.sr))
            mono = mono * (1.0 - 0.9 * occlusion)

        return mono * dist_gain

    def _occlusion(self, src_pos, listener_pos):
        """Combined occlusion factor in [0, 1] plus degenerate flag.

        Degenerate rule: if the listener or the source lies inside an
        obstacle box, that box contributes a fixed partial occlusion of
        0.5 instead of a geometric intersection test.
        """
        total = 0.0
        degenerate = False
        for obs in self.scene["obstacles"]:
            omin, omax = obs["min"], obs["max"]
            if (point_in_aabb(src_pos, omin, omax)
                    or point_in_aabb(listener_pos, omin, omax)):
                total += 0.5
                degenerate = True
            elif segment_intersects_aabb(listener_pos, src_pos, omin, omax):
                total += obs["occlusion"]
        return min(total, 1.0), degenerate

    def run(self, max_frames=None, snapshot_hook=None):
        limit = self.total_frames
        if max_frames is not None:
            limit = min(limit, max_frames)
        interval = self.cfg["snapshot_interval"]
        while self.frames_completed < limit:
            self.process_frame()
            done = self.frames_completed
            if done < self.total_frames and (
                done == limit or (interval and done % interval == 0)
            ):
                if snapshot_hook is not None:
                    snapshot_hook(self)
        return self.frames_completed >= self.total_frames

    def finalize(self):
        """Convert the raw mix buffer to WAV and wrap up the event log."""
        self._raw.close()
        import soundfile as sf

        data = np.fromfile(self.raw_path, dtype="<f4")
        frames = data.size // self.channels
        data = data[: frames * self.channels].reshape(frames, self.channels)
        wav_path = os.path.join(self.out_dir, "mix.wav")
        sf.write(wav_path, data, self.sr, subtype="FLOAT")

        events = []
        with open(self.events_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        header = next((e for e in events if e.get("type") == "header"), {})
        body = [e for e in events if e.get("type") != "header"]
        log = {
            "log_version": EVENT_LOG_VERSION,
            "format": "spatmix-event-log",
            "layout": header.get("layout", self.layout),
            "sample_rate": header.get("sample_rate", self.sr),
            "event_count": len(body),
            "events": body,
        }
        log_path = os.path.join(self.out_dir, "events.json")
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(log, fh, indent=2, sort_keys=True)
        return wav_path, log_path

    def close(self):
        if not self._raw.closed:
            self._raw.close()
        if not self._events_file.closed:
            self._events_file.close()
