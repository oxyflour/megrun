"""Content-addressed caches; separation never enters a module cache key."""
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .interaction import volume_operator, KERNEL_VERSION


def fingerprint(*parts):
    h = hashlib.sha256()
    for part in parts:
        h.update(part if isinstance(part, bytes) else json.dumps(part, sort_keys=True).encode())
    return h.hexdigest()[:24]


def self_operator(camera, directory, dtype="float64"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = fingerprint(camera.centers.tobytes(), camera.volumes.tobytes(), dtype, KERNEL_VERSION)
    path = directory / f"self_{key}.npy"
    start = perf_counter()
    hit = path.exists()
    if hit:
        operator = np.load(path, mmap_mode="r")
    else:
        operator = volume_operator(camera.centers, camera.centers, camera.volumes,
                                   target_volumes=camera.volumes, dtype=dtype)
        temporary = path.with_suffix(".tmp.npy")
        np.save(temporary, operator)
        temporary.replace(path)
    return operator, {"cache_hit": hit, "seconds": perf_counter() - start, "key": key, "path": str(path)}
