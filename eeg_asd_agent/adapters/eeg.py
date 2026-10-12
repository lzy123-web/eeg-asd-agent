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
    auxiliary_channel_names: tuple[str, ...] = ()
    reference_channel_names: tuple[str, ...] = ()

    @property
    def duration_seconds(self) -> float:
        return float(self.data_v.shape[1] / self.sfreq_hz)


class EEGSignalAdapter:
    """Read-only EEG loader and deterministic SciPy processing pipeline.

    Algorithms are isolated adaptations of ``eegsys-bottom-main``: polyphase
    resampling, Welch PSD, and integrated band power. No service, HDFS, or MQ
    dependency is imported.
    """

    version = "eeg-v2.0"
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
        "beta": (12.0, 30.0),
        "gamma": (30.0, 40.0),
    }
    regions = {
        "prefrontal": ("FP1", "FP2"),
        "frontal": ("F3", "F4", "F7", "F8", "FZ"),
        "central": ("C3", "C4", "CZ"),
        "parietal": ("P3", "P4", "PZ"),
        "temporal": ("T3", "T4", "T5", "T6", "T7", "T8", "P7", "P8"),
        "occipital": ("O1", "O2"),
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
            "channel_names": recording.channel_names,
            "auxiliary_channel_names": list(recording.auxiliary_channel_names),
            "reference_channel_names": list(recording.reference_channel_names),
            "montage_family": "standard_10_20_or_compatible",
        }

    def quality(self, state: AssessmentCaseState) -> dict[str, Any]:
        recording = self.load_case(state)
        data = recording.data_v
        finite_ratio = float(np.isfinite(data).sum() / data.size)
        clean = np.nan_to_num(data - np.nanmean(data, axis=1, keepdims=True))
        if self.config.reference == "average" and clean.shape[0] > 1:
            clean = clean - np.mean(clean, axis=0, keepdims=True)
        channel_std_uv = np.nanstd(clean, axis=1) * 1e6
        channel_p99_abs_uv = np.nanpercentile(np.abs(clean), 99.0, axis=1) * 1e6
        channel_artifact_fraction = np.mean(
            np.abs(clean) * 1e6 > self.config.max_abs_amplitude_uv,
            axis=1,
        )
        max_abs_uv = float(np.nanmax(np.abs(clean)) * 1e6)
        flat_mask = channel_std_uv < 0.1
        flat_names = [
            name
            for name, is_flat in zip(recording.channel_names, flat_mask, strict=True)
            if is_flat
        ]
        excessive_names = [
            name
            for name, value in zip(
                recording.channel_names, channel_p99_abs_uv, strict=True
            )
            if value > self.config.max_abs_amplitude_uv
        ]
        freqs, psd = signal.welch(
            clean,
            fs=recording.sfreq_hz,
            nperseg=min(clean.shape[1], max(32, int(recording.sfreq_hz * 2))),
            axis=1,
        )
        upper = min(40.0, recording.sfreq_hz / 2 * 0.95)
        total_power = self._integrate(freqs, psd, 1.0, upper)
        line_low = max(1.0, self.config.notch_hz - 1.0)
        line_high = min(upper, self.config.notch_hz + 1.0)
        line_ratio = (
            self._integrate(freqs, psd, line_low, line_high)
            / np.maximum(total_power, np.finfo(float).eps)
            if line_high > line_low
            else np.zeros(data.shape[0])
        )
        high_low = min(30.0, upper)
        high_ratio = (
            self._integrate(freqs, psd, high_low, upper)
            / np.maximum(total_power, np.finfo(float).eps)
            if upper > high_low
            else np.zeros(data.shape[0])
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(clean) if data.shape[0] > 1 else np.ones((1, 1))
        off_diagonal = corr[np.triu_indices_from(corr, k=1)]
        finite_correlations = np.abs(off_diagonal[np.isfinite(off_diagonal)])
        median_abs_corr = (
            float(np.median(finite_correlations)) if finite_correlations.size else 0.0
        )
        issues: list[str] = []
        warnings: list[str] = []
        if finite_ratio < 1.0:
            issues.append("non-finite signal samples detected")
        if flat_names:
            warnings.append(f"{len(flat_names)} flat or near-flat channels detected")
        if excessive_names:
            warnings.append(
                f"{len(excessive_names)} channels exceed the robust amplitude threshold"
            )
        if recording.duration_seconds < self.config.min_duration_seconds:
            issues.append("recording is shorter than configured minimum duration")
        if data.shape[0] < self.config.min_channels:
            issues.append("recording has fewer channels than configured minimum")
        usable_channels = data.shape[0] - len(flat_names)
        if usable_channels < self.config.min_channels:
            issues.append("too few usable EEG channels remain after flat-channel screening")
        quality_status = "blocked" if issues else ("review" if warnings else "pass")
        return {
            "quality_status": quality_status,
            "finite_sample_ratio": round(finite_ratio, 8),
            "flat_channel_count": len(flat_names),
            "flat_channels": flat_names,
            "excessive_amplitude_channels": excessive_names,
            "max_abs_amplitude_uv": round(max_abs_uv, 6),
            "median_p99_abs_amplitude_uv": round(
                float(np.nanmedian(channel_p99_abs_uv)), 6
            ),
            "max_artifact_sample_fraction": round(
                float(np.nanmax(channel_artifact_fraction)), 8
            ),
            "median_line_noise_ratio": round(float(np.nanmedian(line_ratio)), 8),
            "median_high_frequency_ratio": round(float(np.nanmedian(high_ratio)), 8),
            "median_absolute_channel_correlation": round(median_abs_corr, 8),
            "usable_channel_count": usable_channels,
            "issue_count": len(issues),
            "issues": issues,
            "warning_count": len(warnings),
            "warnings": warnings,
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
            "normalization": "per-channel mean centering and configured rereferencing; amplitudes remain volts",
            "reference": self.config.reference,
            "excluded_auxiliary_channels": list(recording.auxiliary_channel_names),
            "excluded_reference_channels": list(recording.reference_channel_names),
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

        channel_lookup = {
            self._normalize_channel_name(name).upper(): index
            for index, name in enumerate(recording.channel_names)
        }
        global_band_power = {
            name: round(float(np.mean(values)), 9) for name, values in relative.items()
        }
        regional_band_power: dict[str, dict[str, float]] = {}
        for region, candidates in self.regions.items():
            indices = [channel_lookup[name] for name in candidates if name in channel_lookup]
            if indices:
                regional_band_power[region] = {
                    band: round(float(np.mean(values[indices])), 9)
                    for band, values in relative.items()
                }
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

        rms_uv = np.sqrt(np.mean(np.square(flattened), axis=1)) * 1e6
        peak_to_peak_uv = np.ptp(flattened, axis=1) * 1e6
        variance = np.var(flattened, axis=1)
        diff1 = np.diff(flattened, axis=1)
        diff2 = np.diff(diff1, axis=1)
        mobility = np.sqrt(np.var(diff1, axis=1) / np.maximum(variance, np.finfo(float).eps))
        diff_mobility = np.sqrt(
            np.var(diff2, axis=1)
            / np.maximum(np.var(diff1, axis=1), np.finfo(float).eps)
        )
        complexity = diff_mobility / np.maximum(mobility, np.finfo(float).eps)
        temporal_summary = {
            "rms_uv_median": round(float(np.median(rms_uv)), 6),
            "rms_uv_iqr": round(
                float(np.percentile(rms_uv, 75) - np.percentile(rms_uv, 25)), 6
            ),
            "peak_to_peak_uv_median": round(float(np.median(peak_to_peak_uv)), 6),
            "hjorth_activity_median": round(float(np.median(variance)), 12),
            "hjorth_mobility_median": round(float(np.median(mobility)), 9),
            "hjorth_complexity_median": round(float(np.median(complexity)), 9),
        }

        asymmetry: dict[str, dict[str, float]] = {}
        for left_name, right_name in (
            ("FP1", "FP2"),
            ("F3", "F4"),
            ("C3", "C4"),
            ("P3", "P4"),
            ("O1", "O2"),
        ):
            if left_name not in channel_lookup or right_name not in channel_lookup:
                continue
            left_index = channel_lookup[left_name]
            right_index = channel_lookup[right_name]
            asymmetry[f"{left_name}-{right_name}"] = {
                band: round(
                    float(
                        math.log(max(values[right_index], np.finfo(float).eps))
                        - math.log(max(values[left_index], np.finfo(float).eps))
                    ),
                    9,
                )
                for band, values in relative.items()
                if band in {"theta", "alpha", "beta"}
            }

        correlation = np.corrcoef(flattened)
        correlation_values = correlation[np.triu_indices_from(correlation, k=1)]
        connectivity_summary = {
            "mean_absolute_correlation": round(
                float(np.nanmean(np.abs(correlation_values))), 9
            )
            if correlation_values.size
            else 1.0,
            "median_absolute_correlation": round(
                float(np.nanmedian(np.abs(correlation_values))), 9
            )
            if correlation_values.size
            else 1.0,
            "theta_p3_p4_coherence": round(float(features["theta_coh"]), 9)
            if "theta_coh" in features
            else None,
        }

        epoch_band_values: dict[str, list[float]] = {name: [] for name in self.bands}
        for epoch in epochs:
            epoch_freqs, epoch_psd = signal.welch(
                epoch,
                fs=recording.sfreq_hz,
                nperseg=min(epoch.shape[1], int(recording.sfreq_hz * 2)),
                axis=1,
            )
            epoch_total = self._integrate(epoch_freqs, epoch_psd, 0.5, 40.0)
            for band, (low, high) in self.bands.items():
                epoch_absolute = self._integrate(epoch_freqs, epoch_psd, low, high)
                epoch_relative = epoch_absolute / np.maximum(
                    epoch_total, np.finfo(float).eps
                )
                epoch_band_values[band].append(float(np.mean(epoch_relative)))
        stability = {
            band: {
                "median": round(float(np.median(values)), 9),
                "iqr": round(
                    float(np.percentile(values, 75) - np.percentile(values, 25)), 9
                ),
            }
            for band, values in epoch_band_values.items()
            if values
        }

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
            "assessment_summary": {
                "global_relative_band_power": global_band_power,
                "regional_relative_band_power": regional_band_power,
                "temporal": temporal_summary,
                "asymmetry_log_power_ratio": asymmetry,
                "connectivity": connectivity_summary,
                "window_stability": stability,
                "channel_count": len(recording.channel_names),
                "channel_names": recording.channel_names,
                "duration_seconds": round(recording.duration_seconds, 6),
                "sampling_rate_hz": round(recording.sfreq_hz, 6),
            },
        }

    def build_asd_profile(self, state: AssessmentCaseState) -> dict[str, Any]:
        """Build a typed, non-diagnostic profile from deterministic EEG summaries."""
        quality = self.quality(state)
        extracted = self.extract_features(state)
        summary = extracted["assessment_summary"]
        assert isinstance(summary, dict)
        subject_context = {
            key: state.metadata_summary[key]
            for key in ("age", "sex", "recording_state")
            if key in state.metadata_summary
        }
        missing_context = [
            key for key in ("age", "recording_state") if key not in subject_context
        ]
        band_power = summary["global_relative_band_power"]
        dominant_band = max(band_power, key=band_power.get)
        asymmetry = summary["asymmetry_log_power_ratio"]
        asymmetry_values = [
            (pair, band, float(value))
            for pair, values in asymmetry.items()
            for band, value in values.items()
        ]
        largest_asymmetry = (
            max(asymmetry_values, key=lambda item: abs(item[2]))
            if asymmetry_values
            else None
        )
        research_highlights = [
            (
                f"有 {len(quality['excessive_amplitude_channels'])} 个通道需要质量复核，"
                "复核前不宜进行个体层面的结果解释。"
                if quality["quality_status"] == "review"
                else f"信号质量门控结果为 {quality['quality_status']}。"
            ),
            (
                f"全程记录中占比最高的相对频带功率为 {dominant_band} "
                f"（{float(band_power[dominant_band]):.3f}）。"
            ),
        ]
        if largest_asymmetry is not None:
            pair, band, value = largest_asymmetry
            research_highlights.append(
                f"左右同源通道中最大的对数功率差异为 {pair} 的 {band} 频带"
                f"（{value:.3f}）；该结果仅为描述性指标，不是诊断阈值。"
            )
        return {
            "profile_version": "asd-eeg-profile-v1.0",
            "profile_status": "limited" if quality["quality_status"] == "blocked" else "available",
            "quality_status": quality["quality_status"],
            "quality_summary": {
                "usable_channel_count": quality["usable_channel_count"],
                "flat_channels": quality["flat_channels"],
                "excessive_amplitude_channels": quality["excessive_amplitude_channels"],
                "median_line_noise_ratio": quality["median_line_noise_ratio"],
                "warnings": quality["warnings"],
                "issues": quality["issues"],
            },
            "recording_summary": {
                "channel_count": summary["channel_count"],
                "channel_names": summary["channel_names"],
                "duration_seconds": summary["duration_seconds"],
                "analysis_sampling_rate_hz": summary["sampling_rate_hz"],
                "epoch_count": extracted["epoch_count"],
                "epoch_seconds": self.config.epoch_seconds,
            },
            "spectral_summary": {
                "global_relative_band_power": summary["global_relative_band_power"],
                "regional_relative_band_power": summary["regional_relative_band_power"],
                "window_stability": summary["window_stability"],
            },
            "temporal_summary": summary["temporal"],
            "asymmetry_summary": summary["asymmetry_log_power_ratio"],
            "connectivity_summary": summary["connectivity"],
            "research_highlights": research_highlights,
            "subject_context": subject_context,
            "missing_context": missing_context,
            "interpretation_scope": (
                "Descriptive EEG research profile only; ASD-related meaning requires external "
                "evidence and cannot establish diagnosis, severity, or functional ability."
            ),
            "synthetic": extracted["synthetic"],
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
        if self.config.reference == "average" and data.shape[0] > 1:
            data = data - np.mean(data, axis=0, keepdims=True)
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
            auxiliary_channel_names=recording.auxiliary_channel_names,
            reference_channel_names=recording.reference_channel_names,
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
            data_indices = [
                index
                for index, label in enumerate(parsed["labels"])
                if not self._is_auxiliary_channel(label)
            ]
            auxiliary_indices = [
                index for index in range(signal_count) if index not in data_indices
            ]
            reference_indices = [
                index
                for index in data_indices
                if self._is_reference_channel(parsed["labels"][index])
            ]
            data_indices = [index for index in data_indices if index not in reference_indices]
            if not data_indices:
                raise ValueError("EDF contains no EEG signal channels")
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
            channel_names=[
                self._normalize_channel_name(parsed["labels"][index])
                for index in data_indices
            ],
            events=np.empty((0, 2), dtype=np.int64),
            source_format="edf",
            synthetic=False,
            unit_assumption="edf_physical_dimension",
            auxiliary_channel_names=tuple(
                parsed["labels"][index]
                for index in auxiliary_indices
                if index not in reference_indices
            ),
            reference_channel_names=tuple(
                parsed["labels"][index] for index in reference_indices
            ),
        )

    @staticmethod
    def _normalize_channel_name(name: str) -> str:
        normalized = name.strip()
        if normalized.upper().startswith("EEG "):
            normalized = normalized[4:]
        for suffix in ("-REF", "-LE"):
            if normalized.upper().endswith(suffix):
                normalized = normalized[: -len(suffix)]
        return normalized.replace(" ", "")

    @classmethod
    def _is_auxiliary_channel(cls, name: str) -> bool:
        normalized = cls._normalize_channel_name(name).upper()
        compact = normalized.replace("_", "").replace("-", "")
        return compact in {
            "STATUS",
            "EDFANNOTATIONS",
            "ANNOTATIONS",
            "EVENT",
            "EVENTMARKER",
            "TRIGGER",
            "STIM",
        } or compact.startswith(("EOG", "ECG", "EKG", "EMG", "RESP", "TEMP"))

    @classmethod
    def _is_reference_channel(cls, name: str) -> bool:
        normalized = cls._normalize_channel_name(name).upper()
        return normalized in {"A1", "A2", "M1", "M2"}

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
        if not np.any(mask):
            return np.zeros(psd.shape[0])
        integrate = getattr(np, "trapezoid", np.trapz)
        return np.asarray(integrate(psd[:, mask], freqs[mask], axis=1))

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
