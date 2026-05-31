from dataclasses import dataclass
from functools import lru_cache
from math import prod
from typing import Dict, Iterable, Optional

from config import training as training_config
from utils.time.load_calculate import calculate_model_loads_by_layer


@dataclass(frozen=True)
class SplitPointProfile:
    split_point: int
    lower_model_bits: int
    upper_model_bits: int
    full_model_bits: int
    activation_shape_per_sample: tuple
    activation_bits_per_sample: int
    qbar_bits_per_minibatch: int


def _normalize_method(method: str) -> str:
    key = method.strip().lower().replace("-", "").replace("_", "")
    alias = {
        "fl": "fl",
        "splitfed": "splitfed",
        "ringsfl": "ringsfl",
        "parallelsfl": "parallelsfl",
        "sflmac": "sflmac",
    }
    if key not in alias:
        raise ValueError(f"Unsupported method: {method}")
    return alias[key]


def _validate_split_points(split_points: Iterable[int]) -> None:
    invalid = [sp for sp in split_points if sp not in training_config.RESNET18_VALID_SPLIT_POINTS]
    if invalid:
        raise ValueError(
            f"Invalid split points for ResNet18: {invalid}. "
            f"Valid: {training_config.RESNET18_VALID_SPLIT_POINTS}"
        )


@lru_cache(maxsize=None)
def _cached_resnet18_layer_loads(benchmark: str):
    return calculate_model_loads_by_layer(benchmark=benchmark, batch_size=1)


def get_resnet18_split_profiles(
    benchmark: str = "resnet18_cifar100",
    batch_size: int = 64,
    split_points: Optional[Iterable[int]] = None,
) -> Dict[int, SplitPointProfile]:
    """Return split-point communication profiles for ResNet18.

    Args:
        benchmark: e.g. resnet18_cifar10 / resnet18_cifar100.
        batch_size: mini-batch size used in one activation transfer.
        split_points: selected split points. Default uses valid ResNet18 points.
    """
    if split_points is None:
        split_points = training_config.RESNET18_VALID_SPLIT_POINTS
    split_points = list(split_points)
    _validate_split_points(split_points)

    _, _, parameter_bits_by_layer, activate_shape_by_layer = _cached_resnet18_layer_loads(benchmark)
    full_model_bits = sum(parameter_bits_by_layer[1:])

    profiles: Dict[int, SplitPointProfile] = {}
    for sp in split_points:
        lower_model_bits = int(sum(parameter_bits_by_layer[1:sp + 1]))
        upper_model_bits = int(sum(parameter_bits_by_layer[sp + 1:]))
        activation_shape_per_sample = tuple(activate_shape_by_layer[sp])
        activation_bits_per_sample = int(prod(activation_shape_per_sample) * 32)
        qbar_bits_per_minibatch = int(activation_bits_per_sample * batch_size)

        profiles[sp] = SplitPointProfile(
            split_point=sp,
            lower_model_bits=lower_model_bits,
            upper_model_bits=upper_model_bits,
            full_model_bits=int(full_model_bits),
            activation_shape_per_sample=activation_shape_per_sample,
            activation_bits_per_sample=activation_bits_per_sample,
            qbar_bits_per_minibatch=qbar_bits_per_minibatch,
        )

    return profiles


def get_resnet18_average_qbar_bits_per_minibatch(
    benchmark: str = "resnet18_cifar100",
    batch_size: int = 64,
    split_points: Optional[Iterable[int]] = None,
) -> float:
    """Return average minibatch activation communication bits over selected layers.

    By default, uses valid ResNet18 split layers defined in training config.
    """
    profiles = get_resnet18_split_profiles(
        benchmark=benchmark,
        batch_size=batch_size,
        split_points=split_points,
    )
    if len(profiles) == 0:
        raise ValueError("No split profiles available to compute average Qbar")

    total = sum(p.qbar_bits_per_minibatch for p in profiles.values())
    return float(total) / float(len(profiles))


def communication_cost_per_round_bits(
    method: str,
    num_clients: int,
    local_iterations: int,
    profile: SplitPointProfile,
    num_upper_clients: Optional[int] = None,
) -> Dict[str, float]:
    """Compute per-global-round communication cost (bits).

    Returns:
        {
            "per_client_bits": ...,
            "total_system_bits": ...
        }
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")
    if local_iterations <= 0:
        raise ValueError("local_iterations must be > 0")

    m = _normalize_method(method)
    n = float(num_clients)
    k = float(local_iterations)
    wl = float(profile.lower_model_bits)
    wu = float(profile.upper_model_bits)
    w = float(profile.full_model_bits)
    qbar = float(profile.qbar_bits_per_minibatch)

    if m == "fl":
        per_client = 2 * w
    elif m == "splitfed":
        per_client = 2 * k * qbar + 2 * wl
    elif m == "ringsfl":
        per_client = 2 * n * k * qbar + 2 * w
    elif m == "parallelsfl":
        if num_upper_clients is None:
            raise ValueError("num_upper_clients (C) is required for ParallelSFL")
        c = float(num_upper_clients)
        if c < 0 or c > n:
            raise ValueError("num_upper_clients (C) must satisfy 0 <= C <= N")
        per_client = (2 * (n - c) / n) * k * qbar + 2 * wl + (2 * c / n) * wu
    elif m == "sflmac":
        per_client = k * qbar + 2 * wl + wu
    else:
        raise ValueError(f"Unsupported method: {method}")

    return {
        "per_client_bits": per_client,
        "total_system_bits": n * per_client,
    }


def bits_to_megabits(bits: float) -> float:
    return bits / 1e6


def bits_to_megabytes(bits: float) -> float:
    return bits / 8e6
