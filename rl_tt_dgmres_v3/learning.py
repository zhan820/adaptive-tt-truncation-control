"""Small Monte Carlo value ensembles for baseline-guided policy improvement.

Rows must come from real solver rollouts: execute the labelled first action and
then the declared continuation policy until termination.  Fitting another batch
collected under an improved continuation policy implements approximate policy
iteration.  This module neither invents transitions nor bootstraps targets from
its own predictions.

The gate uses ensemble disagreement as a heuristic, not a confidence certificate.
Its failure-penalized cost is distinct from successful-solve wall time.  A cost
gate alone does not prove convergence, preservation of success probability, or
wall-clock dominance over a fixed solver; all decision overhead must be measured.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch
from torch import nn


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite(value: Any, name: str, minimum: Optional[float] = None) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


@dataclass(frozen=True)
class LearningConfig:
    n_actions: int = 9
    ensemble_size: int = 3
    hidden_dim: int = 32
    epochs: int = 120
    batch_size: int = 64
    learning_rate: float = 0.002
    weight_decay: float = 0.0001
    failure_loss_weight: float = 0.5
    pairwise_loss_weight: float = 0.25
    failure_penalty: float = 5.0
    seed: int = 20261001
    feature_clip: float = 12.0
    gradient_clip: float = 5.0

    def __post_init__(self) -> None:
        for name in ("n_actions", "ensemble_size", "hidden_dim", "epochs", "batch_size"):
            _integer(getattr(self, name), name)
        _integer(self.seed, "seed", 0)
        for name in ("learning_rate", "feature_clip", "gradient_clip"):
            if _finite(getattr(self, name), name, 0.0) == 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("weight_decay", "failure_loss_weight", "pairwise_loss_weight", "failure_penalty"):
            _finite(getattr(self, name), name, 0.0)


@dataclass(frozen=True)
class GateConfig:
    """Validation-selected heuristic controls, in normalized total-cost units."""

    min_gain: float = 0.02
    uncertainty_scale: float = 1.0
    max_failure_increase: float = 0.02
    max_failure_probability: float = 0.25
    min_action_samples: int = 3
    max_standardized_feature: float = 8.0
    decision_overhead_fraction: float = 0.0

    def __post_init__(self) -> None:
        for name in ("min_gain", "uncertainty_scale", "decision_overhead_fraction"):
            _finite(getattr(self, name), name, 0.0)
        for name in ("max_failure_increase", "max_failure_probability"):
            value = _finite(getattr(self, name), name, 0.0)
            if value > 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        _integer(self.min_action_samples, "min_action_samples", 0)
        if _finite(self.max_standardized_feature, "max_standardized_feature", 0.0) == 0:
            raise ValueError("max_standardized_feature must be positive")


class _ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, config: LearningConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim, dtype=torch.float32, device="cpu"),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim, dtype=torch.float32, device="cpu"),
            nn.Tanh(),
        )
        self.time = nn.Linear(config.hidden_dim, config.n_actions, dtype=torch.float32, device="cpu")
        self.failure = nn.Linear(config.hidden_dim, config.n_actions, dtype=torch.float32, device="cpu")

    def forward(self, x: torch.Tensor):
        hidden = self.encoder(x)
        return self.time(hidden), self.failure(hidden)


class RolloutValueLearner:
    """CPU value ensemble; only ``fit`` updates weights or normalization.

    Required row fields:
      ``features``, ``action``, ``data_origin='real_rollout'`` and
      ``continuation_policy`` matching ``fit(target_policy=...)``.
    Targets are either ``remaining_wall/time_budget`` plus Boolean ``success``,
    or explicit ``cost_target`` and binary ``failure_target``.  The explicit cost
    must use the same normalized-time-plus-failure-penalty definition.  Optional
    ``state_id`` / ``snapshot_id`` and ``problem_id`` define bootstrap groups;
    otherwise identical feature vectors are grouped, with that limitation logged.

    Call ``fit`` only on the training split.  Threshold/model selection on a
    separate validation split belongs to the experiment driver.  Warm starts
    retain the original train-only scaler and start a fresh optimizer; they are
    not exact optimizer-state resumes.
    """

    FORMAT_VERSION = 1

    def __init__(self, input_dim: Optional[int] = None, config: Optional[LearningConfig] = None):
        self.config = config if config is not None else LearningConfig()
        if not isinstance(self.config, LearningConfig):
            raise TypeError("config must be LearningConfig")
        self.input_dim = None if input_dim is None else _integer(input_dim, "input_dim")
        self.models: list[_ValueNetwork] = []
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_scale: Optional[np.ndarray] = None
        self.feature_min: Optional[np.ndarray] = None
        self.feature_max: Optional[np.ndarray] = None
        self.time_mean = 0.0
        self.time_scale = 1.0
        self.action_counts = np.zeros(self.config.n_actions, dtype=np.int64)
        self.fit_calls = 0
        self.metadata: dict[str, Any] = {
            "method": "Monte Carlo approximate policy iteration with baseline-guided action gating",
            "target_definition": "remaining_wall / time_budget + failure_penalty * failure",
            "heads": "normalized remaining time and failure probability fitted separately; combined only for decisions",
            "pairwise_target": "within-state successful-action remaining-time differences, scaled by state median time",
            "uncertainty_kind": "ensemble disagreement; not a calibrated or certified error bound",
            "normalization": "training rows only; frozen during predict and validation",
            "observations_are_proven_markov": False,
            "fit_history": [],
        }

    @property
    def fitted(self) -> bool:
        return bool(self.models) and self.feature_mean is not None and self.fit_calls > 0

    def _features(self, features: Any) -> np.ndarray:
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                result = np.array(features, dtype=np.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("features must be a finite one-dimensional numeric sequence") from exc
        if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
            raise ValueError("features must be a finite one-dimensional numeric sequence")
        if self.input_dim is not None and result.shape != (self.input_dim,):
            raise ValueError(f"features must have shape ({self.input_dim},)")
        return result

    def _rows(self, rows: Iterable[Mapping[str, Any]], target_policy: str):
        materialized = list(rows)
        if not materialized:
            raise ValueError("fit needs nonempty real-rollout training rows")
        xs, actions, costs, failures, groups, records = [], [], [], [], [], []
        width = self.input_dim
        for index, row in enumerate(materialized):
            if not isinstance(row, Mapping):
                raise ValueError("each training row must be a mapping")
            if row.get("data_origin") != "real_rollout":
                raise ValueError("training rows must declare data_origin='real_rollout'")
            if row.get("continuation_policy") != target_policy:
                raise ValueError("cannot mix continuation policies in one Monte Carlo value fit")
            if row.get("split", "train") not in ("train", "training"):
                raise ValueError("fit accepts training rows only, never validation/test rows")
            features = self._features(row.get("features"))
            width = features.size if width is None else width
            if features.shape != (width,):
                raise ValueError("training feature dimensions do not match")
            action = _integer(row.get("action"), "action", 0)
            if action >= self.config.n_actions:
                raise ValueError("action index is outside configured action space")
            if "cost_target" in row or "failure_target" in row:
                cost = _finite(row.get("cost_target"), "cost_target", 0.0)
                failure = _finite(row.get("failure_target"), "failure_target", 0.0)
                if failure not in (0.0, 1.0):
                    raise ValueError("failure_target must be binary for a real trajectory")
                if cost + 1e-12 < self.config.failure_penalty * failure:
                    raise ValueError("cost_target omits the configured failure penalty")
            else:
                wall = _finite(row.get("remaining_wall"), "remaining_wall", 0.0)
                budget = _finite(row.get("time_budget"), "time_budget", 0.0)
                if budget == 0.0:
                    raise ValueError("time_budget must be positive")
                if not isinstance(row.get("success"), (bool, np.bool_)):
                    raise ValueError("success must be Boolean")
                failure = float(not row["success"])
                cost = wall / budget + self.config.failure_penalty * failure
                if not math.isfinite(cost):
                    raise ValueError("normalized cost is nonfinite")
            state_id = row.get("state_id", row.get("snapshot_id"))
            if state_id is None:
                group = "features:" + hashlib.sha256(features.tobytes()).hexdigest()
            else:
                group = json.dumps([str(row.get("problem_id", "")), str(state_id)], ensure_ascii=False)
            xs.append(features)
            actions.append(action)
            costs.append(cost)
            failures.append(failure)
            groups.append(group)
            records.append({
                "features": features.tolist(), "action": action, "cost_target": cost,
                "failure_target": failure, "group": group,
                "source": str(row.get("source_path", row.get("source", ""))),
            })
        return (
            np.asarray(xs, dtype=np.float64), np.asarray(actions, dtype=np.int64),
            np.asarray(costs, dtype=np.float64), np.asarray(failures, dtype=np.float32),
            groups, records,
        )

    def fit(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        target_policy: str = "baseline",
        source_data: Optional[Iterable[str]] = None,
        warm_start: bool = False,
        epochs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Fit observed Monte Carlo Q targets for one declared continuation policy."""
        if torch.is_inference_mode_enabled():
            raise RuntimeError("fit must run outside solver torch.inference_mode()")
        if not isinstance(target_policy, str) or not target_policy.strip():
            raise ValueError("target_policy must be a nonempty policy identifier")
        if not isinstance(warm_start, bool):
            raise ValueError("warm_start must be Boolean")
        epoch_count = self.config.epochs if epochs is None else _integer(epochs, "epochs")
        x, actions, costs, failures, groups, records = self._rows(rows, target_policy)
        times = np.maximum(costs - self.config.failure_penalty * failures, 0.0)
        if source_data is None:
            sources = sorted({record["source"] for record in records if record["source"]})
        elif isinstance(source_data, (str, bytes)):
            raise ValueError("source_data must be a sequence of source identifiers, not one string")
        else:
            sources = [str(source) for source in source_data]
        restart = not (warm_start and self.fitted)
        if restart:
            self.input_dim = int(x.shape[1])
            self.feature_mean = x.mean(axis=0)
            feature_std = x.std(axis=0)
            self.feature_scale = np.where(feature_std >= 1e-8, feature_std, 1.0)
            self.feature_min = x.min(axis=0)
            self.feature_max = x.max(axis=0)
            self.time_mean = float(times.mean())
            self.time_scale = max(float(times.std()), self.time_mean * 0.1, 1e-4)
            if not all(np.isfinite(values).all() for values in (
                self.feature_mean, self.feature_scale, [self.time_mean, self.time_scale],
            )):
                raise ValueError("training scales overflowed")
            self.models = []
        scaled = self._scale(x)
        y_time = ((times - self.time_mean) / self.time_scale).astype(np.float32)
        if not np.isfinite(y_time).all():
            raise ValueError("normalized training targets overflowed")
        self.action_counts = np.bincount(actions, minlength=self.config.n_actions)
        unique_groups = sorted(set(groups))
        group_rows = {key: np.flatnonzero(np.asarray(groups) == key) for key in unique_groups}
        group_pairs = {}
        for key, indices in group_rows.items():
            successful = [int(i) for i in indices if failures[i] == 0.0]
            scale = max(float(np.median(times[successful])), 1e-4) if successful else 1.0
            pairs = [(i, j, scale) for position, i in enumerate(successful)
                     for j in successful[position + 1:] if actions[i] != actions[j]]
            group_pairs[key] = np.asarray(pairs, dtype=np.float64).reshape(-1, 3)
        member_stats = []
        train_seed = self.config.seed + 104729 * self.fit_calls
        with torch.enable_grad():
            for member in range(self.config.ensemble_size):
                rng = np.random.default_rng(train_seed + 997 * member)
                if restart:
                    with torch.random.fork_rng(devices=[]):
                        torch.random.default_generator.manual_seed(train_seed + 997 * member)
                        model = _ValueNetwork(self.input_dim, self.config)
                    with torch.no_grad():
                        for action in range(self.config.n_actions):
                            mask = actions == action
                            if mask.any():
                                model.time.bias[action] = float(y_time[mask].mean())
                                prior = (float(failures[mask].sum()) + 1.0) / (int(mask.sum()) + 2.0)
                                model.failure.bias[action] = math.log(prior / (1.0 - prior))
                    self.models.append(model)
                else:
                    model = self.models[member]
                sampled_groups = rng.integers(len(unique_groups), size=len(unique_groups))
                indices = np.concatenate([group_rows[unique_groups[i]] for i in sampled_groups])
                pairs = np.concatenate([group_pairs[unique_groups[i]] for i in sampled_groups], axis=0)
                model.train()
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay,
                )
                last_loss = float("nan")
                updates = 0
                try:
                    for _ in range(epoch_count):
                        shuffled = rng.permutation(indices)
                        for start in range(0, shuffled.size, self.config.batch_size):
                            take = shuffled[start:start + self.config.batch_size]
                            bx = torch.from_numpy(scaled[take])
                            ba = torch.from_numpy(actions[take]).unsqueeze(1)
                            bt = torch.from_numpy(y_time[take])
                            bf = torch.from_numpy(failures[take])
                            pred_time, pred_failure = model(bx)
                            time_loss = nn.functional.smooth_l1_loss(pred_time.gather(1, ba).squeeze(1), bt)
                            failure_loss = nn.functional.binary_cross_entropy_with_logits(
                                pred_failure.gather(1, ba).squeeze(1), bf,
                            )
                            loss = time_loss + self.config.failure_loss_weight * failure_loss
                            if pairs.size and self.config.pairwise_loss_weight > 0.0:
                                chosen_pairs = pairs[rng.integers(len(pairs), size=min(self.config.batch_size, len(pairs)))]
                                left = chosen_pairs[:, 0].astype(np.int64)
                                right = chosen_pairs[:, 1].astype(np.int64)
                                pair_scale = torch.from_numpy(chosen_pairs[:, 2].astype(np.float32))
                                left_time, _ = model(torch.from_numpy(scaled[left]))
                                right_time, _ = model(torch.from_numpy(scaled[right]))
                                left_time = left_time.gather(1, torch.from_numpy(actions[left]).unsqueeze(1)).squeeze(1)
                                right_time = right_time.gather(1, torch.from_numpy(actions[right]).unsqueeze(1)).squeeze(1)
                                predicted_difference = (left_time - right_time) * self.time_scale / pair_scale
                                true_difference = torch.from_numpy((times[left] - times[right]).astype(np.float32)) / pair_scale
                                pair_loss = nn.functional.smooth_l1_loss(predicted_difference, true_difference)
                                loss = loss + self.config.pairwise_loss_weight * pair_loss
                            if not torch.isfinite(loss):
                                raise FloatingPointError("nonfinite value-ensemble training loss")
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            nn.utils.clip_grad_norm_(
                                model.parameters(), self.config.gradient_clip, error_if_nonfinite=True,
                            )
                            optimizer.step()
                            last_loss = float(loss.detach())
                            updates += 1
                finally:
                    model.eval()
                member_stats.append({"member": member, "updates": updates, "loss_last": last_loss,
                                     "bootstrap_rows": int(indices.size)})
        self.fit_calls += 1
        digest = hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False,
                                           allow_nan=False).encode("utf-8")).hexdigest()
        summary = {
            "fit_call": self.fit_calls, "target_policy": target_policy,
            "rows": int(x.shape[0]), "input_dim": self.input_dim,
            "bootstrap_groups": len(unique_groups), "bootstrap_unit": "state/snapshot when supplied, otherwise features",
            "epochs": epoch_count, "warm_start": not restart,
            "scaler_refitted": restart, "source_data": sources, "training_rows_sha256": digest,
            "action_counts": self.action_counts.tolist(), "members": member_stats,
            "failure_count": int(failures.sum()),
            "successful_action_pairs": sum(len(pairs) for pairs in group_pairs.values()),
        }
        self.metadata["target_policy"] = target_policy
        self.metadata["source_data"] = sources
        self.metadata["fit_history"].append(copy.deepcopy(summary))
        return summary

    def _scale(self, x: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("learner has no fitted training scaler")
        with np.errstate(over="ignore", invalid="ignore"):
            z = (x - self.feature_mean) / self.feature_scale
        if not np.isfinite(z).all():
            raise ValueError("standardized features overflowed")
        return np.clip(z, -self.config.feature_clip, self.config.feature_clip).astype(np.float32)

    def predict_costs(self, features: Any) -> dict[str, Any]:
        """Predict nine action values without training, scaling updates, or RNG use."""
        if not self.fitted:
            raise RuntimeError("learner must be fitted before prediction")
        x = self._features(features)
        scaled = self._scale(x)
        time_values, failure_values = [], []
        with torch.inference_mode():
            tensor = torch.from_numpy(scaled).unsqueeze(0)
            for model in self.models:
                model.eval()
                time, failure = model(tensor)
                time_values.append(time.squeeze(0).cpu().numpy().astype(np.float64) * self.time_scale + self.time_mean)
                failure_values.append(torch.sigmoid(failure).squeeze(0).cpu().numpy().astype(np.float64))
        time_array = np.maximum(np.asarray(time_values), 0.0)
        failure_array = np.asarray(failure_values)
        cost_array = time_array + self.config.failure_penalty * failure_array
        if not np.isfinite(cost_array).all() or not np.isfinite(failure_array).all():
            raise FloatingPointError("nonfinite value-ensemble prediction")
        zmax = float(np.max(np.abs((x - self.feature_mean) / self.feature_scale)))
        return {
            "cost_mean": cost_array.mean(axis=0).tolist(),
            "cost_std": cost_array.std(axis=0).tolist(),
            "time_mean": time_array.mean(axis=0).tolist(),
            "time_std": time_array.std(axis=0).tolist(),
            "failure_mean": failure_array.mean(axis=0).tolist(),
            "failure_std": failure_array.std(axis=0).tolist(),
            "action_counts": self.action_counts.tolist(),
            "max_standardized_feature": zmax,
            "target_policy": self.metadata["target_policy"],
            "uncertainty_is_certified": False,
        }

    def policy(
        self, features: Any, baseline: int, gate: Optional[GateConfig] = None,
    ) -> tuple[int, dict[str, Any]]:
        """Return the proposed action and a JSON-safe, auditable gate decision.

        Costs and risks are separate checks.  Ensemble intervals are heuristic.
        ``decision_overhead_fraction`` accounts for a candidate's estimated
        current query cost only; it is NOT a guarantee covering earlier or later
        queries that abstain and still consume time.
        """
        baseline = _integer(baseline, "baseline", 0)
        if baseline >= self.config.n_actions:
            raise ValueError("baseline action is outside configured action space")
        gate = GateConfig() if gate is None else gate
        if not isinstance(gate, GateConfig):
            raise TypeError("gate must be GateConfig")
        prediction = self.predict_costs(features)
        means = np.asarray(prediction["cost_mean"])
        spreads = np.asarray(prediction["cost_std"])
        risks = np.asarray(prediction["failure_mean"])
        risk_spreads = np.asarray(prediction["failure_std"])
        lower_base = means[baseline] - gate.uncertainty_scale * spreads[baseline]
        risk_lower_base = max(0.0, risks[baseline] - gate.uncertainty_scale * risk_spreads[baseline])
        upper = means + gate.uncertainty_scale * spreads
        risk_upper = np.minimum(1.0, risks + gate.uncertainty_scale * risk_spreads)
        required = gate.min_gain + gate.decision_overhead_fraction
        reasons: dict[str, list[str]] = {}
        eligible = []
        feature_reject = prediction["max_standardized_feature"] > gate.max_standardized_feature
        for action in range(self.config.n_actions):
            rejected = []
            if action == baseline:
                continue
            if feature_reject:
                rejected.append("feature_out_of_training_scale")
            if self.action_counts[action] < gate.min_action_samples:
                rejected.append("insufficient_action_samples")
            if self.action_counts[baseline] < gate.min_action_samples:
                rejected.append("insufficient_baseline_samples")
            if lower_base - upper[action] <= required:
                rejected.append("insufficient_conservative_gain")
            if risk_upper[action] > gate.max_failure_probability:
                rejected.append("absolute_failure_risk")
            if risk_upper[action] > risk_lower_base + gate.max_failure_increase:
                rejected.append("failure_risk_increase")
            reasons[str(action)] = rejected
            if not rejected:
                eligible.append(action)
        chosen = min(eligible, key=lambda a: (upper[a], a)) if eligible else baseline
        return chosen, {
            "action": chosen, "baseline_action": baseline,
            "deviated": chosen != baseline, "reason": "accepted" if eligible else "baseline_fallback",
            "eligible_actions": eligible, "rejections": reasons,
            "predicted_gain": float(means[baseline] - means[chosen]),
            "conservative_gain": float(lower_base - upper[chosen]) if eligible else 0.0,
            "required_gain": required, "candidate_failure_upper": float(risk_upper[chosen]),
            "baseline_failure_lower": float(risk_lower_base),
            "gate": asdict(gate), "prediction": prediction,
            "guarantees_wallclock_improvement": False,
        }

    def save(self, path: Any, metadata: Optional[Mapping[str, Any]] = None) -> None:
        if not self.fitted:
            raise RuntimeError("cannot save an unfitted value learner")
        saved_metadata = copy.deepcopy(self.metadata)
        if metadata is not None:
            for key, value in metadata.items():
                if key in self.metadata and value != self.metadata[key]:
                    raise ValueError(f"cannot overwrite protected learner metadata: {key}")
                saved_metadata[key] = copy.deepcopy(value)
        json.dumps(saved_metadata, allow_nan=False)
        checkpoint = {
            "format_version": self.FORMAT_VERSION, "config": asdict(self.config),
            "input_dim": self.input_dim, "fit_calls": self.fit_calls,
            "feature_mean": torch.from_numpy(self.feature_mean.copy()),
            "feature_scale": torch.from_numpy(self.feature_scale.copy()),
            "feature_min": torch.from_numpy(self.feature_min.copy()),
            "feature_max": torch.from_numpy(self.feature_max.copy()),
            "time_mean": self.time_mean, "time_scale": self.time_scale,
            "action_counts": torch.from_numpy(self.action_counts.copy()),
            "models": [model.state_dict() for model in self.models], "metadata": saved_metadata,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, destination)

    @classmethod
    def load(cls, path: Any) -> "RolloutValueLearner":
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported value-learner checkpoint format")
        learner = cls(checkpoint["input_dim"], LearningConfig(**checkpoint["config"]))
        learner.fit_calls = _integer(checkpoint["fit_calls"], "fit_calls")
        for name in ("feature_mean", "feature_scale", "feature_min", "feature_max"):
            value = checkpoint[name].detach().cpu().numpy().astype(np.float64, copy=True)
            if value.shape != (learner.input_dim,) or not np.isfinite(value).all():
                raise ValueError(f"invalid checkpoint {name}")
            setattr(learner, name, value)
        if (learner.feature_scale <= 0.0).any() or (learner.feature_min > learner.feature_max).any():
            raise ValueError("invalid checkpoint feature scale/range")
        learner.time_mean = _finite(checkpoint["time_mean"], "time_mean", 0.0)
        learner.time_scale = _finite(checkpoint["time_scale"], "time_scale", 0.0)
        if learner.time_scale == 0.0:
            raise ValueError("invalid checkpoint time scale")
        counts = checkpoint["action_counts"].detach().cpu().numpy()
        if counts.shape != (learner.config.n_actions,) or not np.issubdtype(counts.dtype, np.integer) or (counts < 0).any():
            raise ValueError("invalid checkpoint action counts")
        learner.action_counts = counts.astype(np.int64, copy=True)
        states = checkpoint["models"]
        if len(states) != learner.config.ensemble_size:
            raise ValueError("checkpoint ensemble size mismatch")
        with torch.inference_mode(False), torch.random.fork_rng(devices=[]):
            for state in states:
                if not all(torch.isfinite(value).all() for value in state.values()):
                    raise ValueError("nonfinite checkpoint model weights")
                model = _ValueNetwork(learner.input_dim, learner.config)
                model.load_state_dict(state)
                model.eval()
                learner.models.append(model)
        learner.metadata = copy.deepcopy(checkpoint["metadata"])
        if not isinstance(learner.metadata.get("target_policy"), str) or not learner.metadata["target_policy"]:
            raise ValueError("checkpoint lacks continuation-policy metadata")
        if len(learner.metadata.get("fit_history", [])) != learner.fit_calls:
            raise ValueError("checkpoint fit history mismatch")
        json.dumps(learner.metadata, allow_nan=False)
        return learner
