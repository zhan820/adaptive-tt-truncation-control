"""Small joint restart/error-budget controller for adaptive TT-DGMRES.

The model consumes numerical summaries, never a reference solution or raw TT
cores. Actions are *features*: ``(normalized_restart, log10_budget_fraction)``.
The caller owns their normalization and must keep it fixed across training and
inference. All four predictions share one action-conditioned trunk.
"""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


CHECKPOINT_VERSION = 1


class JointControllerNet(nn.Module):
    """Encode variable TT interfaces and cycle history, then score joint actions.

    ``log_cost`` (also exposed as ``q``) predicts log remaining total runtime;
    lower is better. Auxiliary outputs predict log cycle seconds, log residual
    reduction factor, and log rank, respectively. Their interpretation and loss
    targets belong to the training/data layer, not this network.

    Forward inputs are moved to the model device/dtype. Masked interface and
    time padding is ignored, including nonfinite values in padded positions.
    Each actual time step must contain at least one valid interface token.
    """

    def __init__(self, token_dim: int = 10, global_dim: int = 18) -> None:
        super().__init__()
        if token_dim < 1 or global_dim < 1:
            raise ValueError("token_dim and global_dim must be positive")
        self.token_dim = int(token_dim)
        self.global_dim = int(global_dim)
        self.token_encoder = nn.Sequential(
            nn.Linear(self.token_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )
        self.history_encoder = nn.GRU(96, 64, batch_first=True)
        self.action_trunk = nn.Sequential(
            nn.Linear(66, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
        )
        self.output_heads = nn.Linear(32, 4)

    def architecture_config(self) -> Dict[str, int]:
        return {"token_dim": self.token_dim, "global_dim": self.global_dim}

    def forward(
        self,
        tokens: Tensor,
        globals: Tensor,
        token_mask: Tensor,
        lengths: Tensor,
        actions: Tensor,
    ) -> Dict[str, Tensor]:
        """Return four ``[B,K]`` predictions for each state/action combination.

        Shapes: tokens ``[B,L,N,token_dim]``, globals ``[B,L,global_dim]``,
        token_mask ``[B,L,N]``, lengths ``[B]``, actions ``[K,2]`` or ``[B,K,2]``.
        Histories are right-padded. No optimizer or action-selection state is
        retained between calls.
        """
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        tokens = torch.as_tensor(tokens, device=device, dtype=dtype)
        globals = torch.as_tensor(globals, device=device, dtype=dtype)
        token_mask = torch.as_tensor(token_mask, device=device, dtype=torch.bool)
        actions = torch.as_tensor(actions, device=device, dtype=dtype)
        raw_lengths = torch.as_tensor(lengths, device="cpu")
        if raw_lengths.is_floating_point() and (
            not torch.isfinite(raw_lengths).all()
            or not torch.equal(raw_lengths, raw_lengths.floor())
        ):
            raise ValueError("lengths must contain finite integers")
        lengths = raw_lengths.to(dtype=torch.long)
        if tokens.ndim != 4 or tokens.shape[-1] != self.token_dim:
            raise ValueError("tokens must have shape [B,L,N,token_dim]")
        batch, steps, interfaces, _ = tokens.shape
        if min(batch, steps, interfaces) < 1:
            raise ValueError("batch, history, and token dimensions must be nonempty")
        if globals.shape != (batch, steps, self.global_dim):
            raise ValueError("globals must have shape [B,L,global_dim]")
        if token_mask.shape != (batch, steps, interfaces):
            raise ValueError("token_mask must have shape [B,L,N]")
        if lengths.shape != (batch,) or ((lengths < 1) | (lengths > steps)).any():
            raise ValueError("each history length must lie in [1,L]")
        if actions.ndim == 2 and actions.shape[-1] == 2:
            actions = actions.unsqueeze(0).expand(batch, -1, -1)
        if actions.ndim != 3 or actions.shape[0] != batch or actions.shape[-1] != 2:
            raise ValueError("actions must have shape [K,2] or [B,K,2]")
        if actions.shape[1] < 1 or not torch.isfinite(actions).all():
            raise ValueError("at least one finite action is required")

        valid_time = torch.arange(steps, device=device)[None, :] < lengths.to(device)[:, None]
        mask = token_mask & valid_time[:, :, None]
        if (valid_time & ~mask.any(dim=2)).any():
            raise ValueError("each actual time step needs at least one valid token")
        if not torch.isfinite(tokens[mask]).all() or not torch.isfinite(globals[valid_time]).all():
            raise ValueError("active state features must be finite")

        # Mask before encoding, so arbitrary padding cannot contaminate gradients.
        clean_tokens = tokens.masked_fill(~mask.unsqueeze(-1), 0.0)
        token_features = self.token_encoder(clean_tokens)
        count = mask.sum(dim=2, keepdim=True).clamp_min(1).to(dtype)
        mean = token_features.masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=2) / count
        maximum = token_features.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(dim=2)
        maximum = maximum.masked_fill(~valid_time.unsqueeze(-1), 0.0)
        global_features = self.global_encoder(globals.masked_fill(~valid_time.unsqueeze(-1), 0.0))
        sequence = torch.cat((mean, maximum, global_features), dim=-1)
        packed = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.history_encoder(packed)
        state = hidden[-1].unsqueeze(1).expand(-1, actions.shape[1], -1)
        predictions = self.output_heads(self.action_trunk(torch.cat((state, actions), dim=-1)))
        log_cost, cycle_log_seconds, log_reduction, log_rank = predictions.unbind(dim=-1)
        return {
            "q": log_cost,
            "log_cost": log_cost,
            "cycle_log_seconds": cycle_log_seconds,
            "log_reduction": log_reduction,
            "log_rank": log_rank,
        }


def tensorize_histories(
    histories: Sequence[Sequence[Mapping[str, Any]]],
    actions: Any,
    history_limit: int = 5,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Tensor]:
    """Pad histories of ``{'tokens': rows, 'global': features}`` for forward.

    The last ``history_limit`` states are retained, in chronological order.
    Interface counts and history lengths may differ between examples. Feature
    widths are inferred and validated. Action values are already normalized;
    both shared ``[K,2]`` and per-example ``[B,K,2]`` arrays are accepted.
    """
    if history_limit < 1 or not histories:
        raise ValueError("history_limit and number of histories must be positive")
    kept = [list(history)[-history_limit:] for history in histories]
    if any(not history for history in kept):
        raise ValueError("empty histories are not valid controller states")
    prepared = []
    token_dim = global_dim = None
    max_tokens = 0
    for history in kept:
        records = []
        for state in history:
            tok = torch.as_tensor(state["tokens"], dtype=dtype, device=device)
            glob = torch.as_tensor(state["global"], dtype=dtype, device=device)
            if tok.ndim != 2 or tok.shape[0] < 1 or tok.shape[1] < 1:
                raise ValueError("state tokens must be a nonempty two-dimensional array")
            if glob.ndim != 1 or glob.shape[0] < 1:
                raise ValueError("state global features must be a nonempty vector")
            if token_dim is None:
                token_dim, global_dim = tok.shape[1], glob.shape[0]
            if tok.shape[1] != token_dim or glob.shape[0] != global_dim:
                raise ValueError("all states must use consistent feature widths")
            if not torch.isfinite(tok).all() or not torch.isfinite(glob).all():
                raise ValueError("state features must be finite")
            max_tokens = max(max_tokens, tok.shape[0])
            records.append((tok, glob))
        prepared.append(records)
    batch, steps = len(kept), max(map(len, kept))
    tokens = torch.zeros((batch, steps, max_tokens, token_dim), dtype=dtype, device=device)
    globals_tensor = torch.zeros((batch, steps, global_dim), dtype=dtype, device=device)
    token_mask = torch.zeros((batch, steps, max_tokens), dtype=torch.bool, device=device)
    for b, history in enumerate(prepared):
        for t, (tok, glob) in enumerate(history):
            tokens[b, t, :tok.shape[0]] = tok
            globals_tensor[b, t] = glob
            token_mask[b, t, :tok.shape[0]] = True
    return {
        "tokens": tokens,
        "globals": globals_tensor,
        "token_mask": token_mask,
        "lengths": torch.tensor([len(h) for h in kept], dtype=torch.long),
        "actions": torch.as_tensor(actions, dtype=dtype, device=device),
    }


def save_controller_checkpoint(
    path: Union[str, Path],
    model: JointControllerNet,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Save weights/config plus caller metadata (use only simple data values)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": "shared_tt_mlp_gru_joint_v1",
        "model_config": model.architecture_config(),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": dict(metadata or {}),
    }, path)


def load_controller_checkpoint(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Tuple[JointControllerNet, Dict[str, Any]]:
    """Load a compatible checkpoint, preserve weight dtype, return eval model."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION
            or checkpoint.get("architecture") != "shared_tt_mlp_gru_joint_v1"):
        raise ValueError("unsupported controller checkpoint architecture/version")
    model = JointControllerNet(**checkpoint["model_config"])
    first_weight = next(iter(checkpoint["state_dict"].values()))
    model = model.to(device=device, dtype=first_weight.dtype)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, dict(checkpoint.get("metadata", {}))
