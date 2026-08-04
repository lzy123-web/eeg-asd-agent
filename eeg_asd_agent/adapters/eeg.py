from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import io as scipy_io
from scipy import signal

from eeg_asd_agent.schemas import AssessmentCaseState, EEGProcessingConfig


@dataclass(frozen=True)
class EEGRecording:
    data_v: np.ndarray
    sfreq_hz: float
    channel_names: list[str]
    events: np.ndarray
    source_format: str
    synthetic: bool
    unit_assumption: str

    @property
    def duration_seconds(self) -> float:
        return float(self.data_v.shape[1] / self.sfreq_hz)


class EEGSignalAdapter:
    """Read-only EEG loader and deterministic SciPy processing pipeline.

    Algorithms are isolated adaptations of ``eegsys-bottom-main``: polyphase
    resampling, Welch PSD, and integrated band power. No service, HDFS, or MQ
    dependency is imported.
    """

    version = "eeg-v1.0"
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
        "beta": (12.0, 30.0),
        "gamma": (30.0, 40.0),
    }

    def __init__(self, config: EEGProcessingConfig | None = None) -> None:
        self.config = config or EEGProcessingConfig()

    def inspect(self, state: AssessmentCaseState) -> dict[str, Any]:
        recording = self.load_case(state)
        return {
            "pipeline_version": self.config.version,
            "source_format": recording.source_format,
            "synthetic": recording.synthetic,
            "channel_count": int(recording.data_v.shape[0]),
            "sample_count": int(recording.data_v.shape[1]),
            "sampling_rate_hz": round(recording.sfreq_hz, 6),
            "duration_seconds": round(recording.duration_seconds, 6),
            "event_count": int(len(recording.events)),
            "unit_assumption": recording.unit_assumption,
        }

    def quality(self, state: AssessmentCaseState) -> dict[str, Any]:
        recording = self.load_case(state)
        data = recording.data_v
        finite_ratio = float(np.isfinite(data).sum() / data.size)
        channel_std_uv = np.nanstd(data, axis=1) * 1e6
        max_abs_uv = float(np.nanmax(np.abs(data)) * 1e6)
        flat_channels = int(np.sum(channel_std_uv < 0.1))
        issues: list[str] = []
        if finite_ratio < 1.0:
            issues.append("non-finite signal samples detected")
        if flat_channels:
            issues.append(f"{flat_channels} flat or near-flat channels detected")
        if max_abs_uv > self.config.max_abs_amplitude_uv:
            issues.append("amplitude exceeds configured artifact threshold")
        if recording.duration_seconds < self.config.min_duration_seconds:
            issues.append("recording is shorter than configured minimum duration")
        if data.shape[0] < self.config.min_channels:
            issues.append("recording has fewer channels than configured minimum")
        return {
            "quality_status": "pass" if not issues else "blocked",
            "finite_sample_ratio": round(finite_ratio, 8),
            "flat_channel_count": flat_channels,
            "max_abs_amplitude_uv": round(max_abs_uv, 6),
            "issue_count": len(issues),
            "issues": issues,
            "synthetic": recording.synthetic,
            "pipeline_version": self.config.version,
        }

    def preprocess_summary(self, state: AssessmentCaseState) -> dict[str, Any]:
        recording = self.load_case(state)
        processed = self.preprocess(recording)
        return {
            "pipeline_version": self.config.version,
            "source_sampling_rate_hz": round(recording.sfreq_hz, 6),
            "target_sampling_rate_hz": round(processed.sfreq_hz, 6),
            "channel_count": int(processed.data_v.shape[0]),
            "sample_count": int(processed.data_v.shape[1]),
            "highpass_hz": self.config.highpass_hz,
            "lowpass_hz": self.config.lowpass_hz,
            "notch_hz": self.config.notch_hz,
            "normalization": "per-channel mean centering only; amplitudes remain volts",
            "synthetic": recording.synthetic,
        }

    def event_summary(self, state: AssessmentCaseState) -> dict[str, Any]:
        recording = self.preprocess(self.load_case(state))
        events, mode = self.extract_events(recording)
        return {
            "event_count": int(len(events)),
            "event_mode": mode,
            "event_codes": sorted({int(item[1]) for item in events}),
            "pipeline_version": self.config.version,
            "synthetic": recording.synthetic,
        }

    def epoch_summary(self, state: AssessmentCaseState) -> dict[str, Any]:
        recording = self.preprocess(self.load_case(state))
        epochs, mode = self.create_epochs(recording)
        return {
            "epoch_count": int(epochs.shape[0]),
            "channel_count": int(epochs.shape[1]),
            "samples_per_epoch": int(epochs.shape[2]),
            "epoch_seconds": self.config.epoch_seconds,
            "event_mode": mode,
            "pipeline_version": self.config.version,
            "synthetic": recording.synthetic,
        }

    def extract_features(self, state: AssessmentCaseState) -> dict[str, Any]:
        recording = self.preprocess(self.load_case(state))
        epochs, event_mode = self.create_epochs(recording)
        flattened = np.concatenate([epoch for epoch in epochs], axis=1)
        freqs, psd = signal.welch(
            flattened,
            fs=recording.sfreq_hz,
            nperseg=min(flattened.shape[1], int(recording.sfreq_hz * 2)),
            axis=1,
        )
        total = self._integrate(freqs, psd, 0.5, 40.0)
        relative: dict[str, np.ndarray] = {}
        features: dict[str, float] = {}
        for name, (low, high) in self.bands.items():
            absolute = self._integrate(freqs, psd, low, high)
            relative[name] = absolute / np.maximum(total, np.finfo(float).eps)
            features[f"{name}_relative_power"] = float(np.mean(relative[name]))
            features[f"{name}_relative_power_std"] = float(np.std(relative[name]))

        channel_lookup = {name.upper(): index for index, name in enumerate(recording.channel_names)}
        parietal_indices = [
            channel_lookup[name]
            for name in ("P3", "P4", "O1", "O2")
            if name in channel_lookup
        ]
        if not parietal_indices:
            parietal_indices = list(range(recording.data_v.shape[0]))
        features["parietal_alpha"] = float(np.mean(relative["alpha"][parietal_indices]))
        features["beta_gamma_mean"] = float(
            np.mean(relative["beta"] + relative["gamma"])
        )
        features["higuchi_fd"] = float(
            np.mean([self._higuchi_fd(channel) for channel in flattened])
        )
        if "F3" in channel_lookup and "F4" in channel_lookup:
            left = max(relative["alpha"][channel_lookup["F3"]], np.finfo(float).eps)
            right = max(relative["alpha"][channel_lookup["F4"]], np.finfo(float).eps)
            features["faa_global"] = float(math.log(right) - math.log(left))
        if "P3" in channel_lookup and "P4" in channel_lookup:
            f_coh, coherence = signal.coherence(
                flattened[channel_lookup["P3"]],
                flattened[channel_lookup["P4"]],
                fs=recording.sfreq_hz,
                nperseg=min(flattened.shape[1], int(recording.sfreq_hz * 2)),
            )
            theta_mask = (f_coh >= 4.0) & (f_coh < 8.0)
            features["theta_coh"] = float(np.mean(coherence[theta_mask]))

        clean_features = {
            key: round(value, 9)
            for key, value in features.items()
            if math.isfinite(value)
        }
        return {
            "features": clean_features,
            "feature_count": len(clean_features),
            "epoch_count": int(epochs.shape[0]),
            "event_mode": event_mode,
            "pipeline_version": self.config.version,
            "synthetic": recording.synthetic,
            "feature_semantics": "recording-level deterministic spectral summaries",
        }

    def load_case(self, state: AssessmentCaseState) -> EEGRecording:
        if not state.raw_files:
            raise ValueError("raw EEG input is required")
        source = state.raw_files[0]
        if source.startswith("synthetic://"):
            return self._synthetic(source, state)
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError("configured EEG input does not exist")
        suffix = path.suffix.lower()
        if suffix == ".npz":
            return self._load_npz(path)
        if suffix == ".mat":
            return self._load_mat(path, state)
        if suffix == ".edf":
            return self._load_edf(path)
        raise ValueError(f"unsupported EEG format: {suffix or 'unknown'}")

    def preprocess(self, recording: EEGRecording) -> EEGRecording:
        data = np.asarray(recording.data_v, dtype=np.float64)
        data = signal.detrend(data, axis=1, type="constant")
        nyquist = recording.sfreq_hz / 2
        high = min(self.config.lowpass_hz, nyquist * 0.95)
        if high <= self.config.highpass_hz:
            raise ValueError("sampling rate is too low for configured bandpass")
        sos = signal.butter(
            4,
            [self.config.highpass_hz, high],
            btype="bandpass",
            fs=recording.sfreq_hz,
            output="sos",
        )
        data = signal.sosfiltfilt(sos, data, axis=1)
        if self.config.notch_hz < nyquist * 0.95:
            b_notch, a_notch = signal.iirnotch(
                self.config.notch_hz, 30.0, fs=recording.sfreq_hz
            )
            data = signal.filtfilt(b_notch, a_notch, data, axis=1)
        target = self.config.target_sfreq_hz
        if not math.isclose(recording.sfreq_hz, target):
            source_int = int(round(recording.sfreq_hz))
            divisor = math.gcd(source_int, target)
            data = signal.resample_poly(data, target // divisor, source_int // divisor, axis=1)
        return EEGRecording(
            data_v=data,
            sfreq_hz=float(target),
            channel_names=recording.channel_names,
            events=self._rescale_events(recording.events, recording.sfreq_hz, target),
            source_format=recording.source_format,
            synthetic=recording.synthetic,
            unit_assumption=recording.unit_assumption,
        )

    def extract_events(self, recording: EEGRecording) -> tuple[np.ndarray, str]:
        if recording.events.size:
            return recording.events.astype(np.int64), "recorded"
        step = int(round(self.config.epoch_seconds * recording.sfreq_hz))
        onsets = np.arange(0, recording.data_v.shape[1] - step + 1, step, dtype=np.int64)
        return np.column_stack([onsets, np.ones(len(onsets), dtype=np.int64)]), "fixed_window_inferred"

    def create_epochs(self, recording: EEGRecording) -> tuple[np.ndarray, str]:
        events, mode = self.extract_events(recording)
        length = int(round(self.config.epoch_seconds * recording.sfreq_hz))
        epochs = [
            recording.data_v[:, int(onset) : int(onset) + length]
            for onset, _code in events
            if int(onset) >= 0 and int(onset) + length <= recording.data_v.shape[1]
        ]
        if not epochs:
            raise ValueError("no complete epochs can be created")
        return np.stack(epochs), mode

    def _synthetic(self, source: str, state: AssessmentCaseState) -> EEGRecording:
        seed = int.from_bytes(hashlib.sha256(source.encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        sfreq = float(state.metadata_summary.get("sampling_rate_hz", 128))
        duration = float(state.metadata_summary.get("duration_seconds", 10))
        requested_channels = int(state.metadata_summary.get("channel_count", 8))
        names = ["F3", "F4", "P3", "P4", "C3", "C4", "O1", "O2"][:requested_channels]
        if len(names) < 2:
            names = ["F3", "F4"]
        times = np.arange(int(sfreq * duration)) / sfreq
        channels = []
        for index, _name in enumerate(names):
            alpha = (18 + index) * 1e-6 * np.sin(2 * np.pi * (9.5 + index * 0.1) * times)
            theta = 8e-6 * np.sin(2 * np.pi * 6 * times + index * 0.2)
            noise = rng.normal(0, 4e-6, len(times))
            channels.append(alpha + theta + noise)
        event_step = int(round(self.config.epoch_seconds * sfreq))
        event_count = int(state.metadata_summary.get("event_count", 0))
        onsets = np.arange(0, len(times) - event_step + 1, event_step, dtype=np.int64)
        if event_count > 0:
            onsets = onsets[:event_count]
        events = np.column_stack([onsets, np.ones(len(onsets), dtype=np.int64)])
        return EEGRecording(
            data_v=np.asarray(channels),
            sfreq_hz=sfreq,
            channel_names=names,
            events=events,
            source_format="synthetic_npz_equivalent",
            synthetic=True,
            unit_assumption="generated_volts",
        )

    def _load_npz(self, path: Path) -> EEGRecording:
        with np.load(path, allow_pickle=False) as archive:
            if "data" not in archive or "sfreq" not in archive:
                raise ValueError("NPZ requires data and sfreq arrays")
            data = self._as_channels_first(np.asarray(archive["data"], dtype=float))
            sfreq = float(np.asarray(archive["sfreq"]).reshape(-1)[0])
            names = (
                [str(item) for item in np.asarray(archive["channel_names"]).reshape(-1)]
                if "channel_names" in archive
                else [f"EEG{index + 1}" for index in range(data.shape[0])]
            )
            events = (
                np.asarray(archive["events"], dtype=np.int64).reshape(-1, 2)
                if "events" in archive
                else np.empty((0, 2), dtype=np.int64)
            )
            unit = (
                str(np.asarray(archive["unit"]).reshape(-1)[0]).lower()
                if "unit" in archive
                else "v"
            )
            synthetic = bool(np.asarray(archive.get("synthetic", False)).reshape(-1)[0])
        factor = 1e-6 if unit in {"uv", "µv", "microvolt", "microvolts"} else 1.0
        return EEGRecording(
            data_v=data * factor,
            sfreq_hz=sfreq,
            channel_names=names,
            events=events,
            source_format="npz",
            synthetic=synthetic,
            unit_assumption=f"declared_{unit}",
        )

    def _load_mat(self, path: Path, state: AssessmentCaseState) -> EEGRecording:
        raw = {key: value for key, value in scipy_io.loadmat(path).items() if not key.startswith("__")}
        data_value = next(
            (value for key, value in raw.items() if key.lower().startswith("data") and np.asarray(value).ndim == 2),
            None,
        )
        if data_value is None:
            data_value = next((value for value in raw.values() if np.asarray(value).ndim == 2), None)
        if data_value is None:
            raise ValueError("MAT contains no two-dimensional signal array")
        data = self._as_channels_first(np.asarray(data_value, dtype=float))
        sfreq_value = next(
            (value for key, value in raw.items() if key.lower() in {"sfreq", "srate", "fs"}),
            state.metadata_summary.get("sampling_rate_hz"),
        )
        if sfreq_value is None:
            raise ValueError("MAT sampling rate is missing")
        sfreq = float(np.asarray(sfreq_value).reshape(-1)[0])
        scale = 1e-6 if float(np.nanmedian(np.abs(data))) > 0.01 else 1.0
        return EEGRecording(
            data_v=data * scale,
            sfreq_hz=sfreq,
            channel_names=[f"EEG{index + 1}" for index in range(data.shape[0])],
            events=np.empty((0, 2), dtype=np.int64),
            source_format="mat",
            synthetic=False,
            unit_assumption="magnitude_inferred_microvolts" if scale == 1e-6 else "magnitude_inferred_volts",
        )

    def _load_edf(self, path: Path) -> EEGRecording:
        with path.open("rb") as handle:
            fixed = handle.read(256).decode("latin-1")
            header_bytes = int(fixed[184:192].strip())
            record_count = int(fixed[236:244].strip())
            record_duration = float(fixed[244:252].strip())
            signal_count = int(fixed[252:256].strip())
            fields = [
                ("labels", 16), ("transducers", 80), ("dimensions", 8),
                ("physical_min", 8), ("physical_max", 8), ("digital_min", 8),
                ("digital_max", 8), ("prefilter", 80), ("samples", 8), ("reserved", 32),
            ]
            parsed: dict[str, list[str]] = {}
            for key, width in fields:
                parsed[key] = [handle.read(width).decode("latin-1").strip() for _ in range(signal_count)]
            samples = [int(value) for value in parsed["samples"]]
            rates = [value / record_duration for value in samples]
            data_indices = [i for i, label in enumerate(parsed["labels"]) if label != "EDF Annotations"]
            unique_rates = {rates[index] for index in data_indices}
            if len(unique_rates) != 1:
                raise ValueError("EDF channels with mixed sampling rates are not supported in V1")
            buffers = [np.empty(record_count * samples[index], dtype=float) for index in data_indices]
            handle.seek(header_bytes)
            output_lookup = {source: target for target, source in enumerate(data_indices)}
            for record in range(record_count):
                for source_index, count in enumerate(samples):
                    values = np.frombuffer(handle.read(count * 2), dtype="<i2")
                    if source_index not in output_lookup:
                        continue
                    target = output_lookup[source_index]
                    pmin = float(parsed["physical_min"][source_index])
                    pmax = float(parsed["physical_max"][source_index])
                    dmin = float(parsed["digital_min"][source_index])
                    dmax = float(parsed["digital_max"][source_index])
                    physical = values * ((pmax - pmin) / (dmax - dmin)) + pmax - ((pmax - pmin) / (dmax - dmin)) * dmax
                    start = record * count
                    buffers[target][start : start + count] = physical
        dimensions = [parsed["dimensions"][index].lower() for index in data_indices]
        factors = [1e-6 if unit in {"uv", "µv"} else 1.0 for unit in dimensions]
        data = np.stack([buffer * factor for buffer, factor in zip(buffers, factors, strict=True)])
        return EEGRecording(
            data_v=data,
            sfreq_hz=float(unique_rates.pop()),
            channel_names=[parsed["labels"][index] for index in data_indices],
            events=np.empty((0, 2), dtype=np.int64),
            source_format="edf",
            synthetic=False,
            unit_assumption="edf_physical_dimension",
        )

    @staticmethod
    def _as_channels_first(data: np.ndarray) -> np.ndarray:
        if data.ndim != 2:
            raise ValueError("EEG data must have shape channels x samples")
        if data.shape[0] > data.shape[1]:
            data = data.T
        if data.size == 0:
            raise ValueError("EEG data is empty")
        return data

    @staticmethod
    def _rescale_events(events: np.ndarray, source: float, target: float) -> np.ndarray:
        if not events.size or math.isclose(source, target):
            return events.copy()
        scaled = events.copy().astype(np.int64)
        scaled[:, 0] = np.rint(scaled[:, 0] * target / source).astype(np.int64)
        return scaled

    @staticmethod
    def _integrate(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> np.ndarray:
        mask = (freqs >= low) & (freqs < high)
        return np.asarray(np.trapezoid(psd[:, mask], freqs[mask], axis=1))

    @staticmethod
    def _higuchi_fd(values: np.ndarray, k_max: int = 8) -> float:
        series = np.asarray(values, dtype=float)
        lengths: list[float] = []
        scales: list[float] = []
        for k in range(1, min(k_max, len(series) // 4) + 1):
            per_offset = []
            for offset in range(k):
                count = (len(series) - offset - 1) // k
                if count < 2:
                    continue
                indices = offset + np.arange(count + 1) * k
                distance = np.abs(np.diff(series[indices])).sum()
                per_offset.append(distance * (len(series) - 1) / (count * k * k))
            if per_offset and np.mean(per_offset) > 0:
                lengths.append(float(np.mean(per_offset)))
                scales.append(1.0 / k)
        if len(lengths) < 2:
            return 1.0
        slope, _ = np.polyfit(np.log(scales), np.log(lengths), 1)
        return float(slope)
