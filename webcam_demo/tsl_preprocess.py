import csv

import numpy as np


ARM_POINTS = ("l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist")
FACE_POINTS = ("lbrow_outer", "lbrow_inner", "rbrow_inner", "rbrow_outer", "mouth_right", "mouth_left")
COORDINATE_COLUMNS = (
    [f"{side}_{axis}{point}" for side in ("lh", "rh") for point in range(21) for axis in "xyz"]
    + [f"{point}_{axis}" for point in ARM_POINTS + FACE_POINTS for axis in "xyz"]
)
N_POINTS = len(COORDINATE_COLUMNS) // 3
INPUT_FEATURES = 4 * N_POINTS


def landmarks_to_tensor(coordinates, frames, time_ms=None):
    """Signed interval -> float32 array [frames, 216] in the trained feature order."""
    points = np.asarray(coordinates, dtype=np.float32).reshape(-1, N_POINTS, 3)
    if len(points) < 2:
        raise ValueError("Need at least two frames")
    if frames < 2:
        raise ValueError("frames must be at least two")

    valid = np.isfinite(points).all(axis=-1)
    if not valid.any():
        raise ValueError("No detected landmarks")
    points = np.where(valid[..., None], points, 0.0)

    left, right = points[:, 42, :2], points[:, 43, :2]
    widths = np.linalg.norm(left - right, axis=-1)
    anchors = valid[:, 42] & valid[:, 43] & (widths > 0.025)
    if anchors.any():
        indexes = np.arange(len(points))
        center = (left + right) / 2.0
        center = np.column_stack([
            np.interp(indexes, indexes[anchors], center[anchors, axis]) for axis in range(2)
        ])
        points[:, :, :2] -= center[:, None, :]
        points /= np.median(widths[anchors])
    else:
        points[:, :, :2] -= 0.5

    clock = np.arange(len(points), dtype=np.float64)
    if time_ms is not None:
        stamps = np.asarray(time_ms, dtype=np.float64)
        if stamps.shape == clock.shape and np.isfinite(stamps).all() and (np.diff(stamps) > 0).all():
            clock = stamps

    target = np.linspace(clock[0], clock[-1], frames)
    hi = np.searchsorted(clock, target, side="right").clip(1, len(points) - 1)
    lo = hi - 1
    alpha = ((target - clock[lo]) / (clock[hi] - clock[lo]))[:, None, None]
    nearest = np.where(alpha[:, 0, 0] < 0.5, lo, hi)
    both = valid[lo] & valid[hi]
    interpolated = points[lo] + alpha * (points[hi] - points[lo])
    selected = np.where(both[..., None], interpolated, points[nearest])
    selected = np.where(valid[nearest, :, None], selected, 0.0)
    return np.concatenate(
        [selected.reshape(frames, -1), valid[nearest]], axis=-1,
    ).astype(np.float32)


def read_landmark_csv(file, frames):
    """Read an open UTF-8/UTF-8-SIG text stream in the TSL-51 CSV format."""
    reader = csv.reader(file)
    header = next(reader, [])
    index = {name.strip(): i for i, name in enumerate(header)}
    missing = [column for column in COORDINATE_COLUMNS if column not in index]
    if missing:
        raise ValueError(f"Missing landmark columns: {missing[:8]} ({len(missing)} total)")
    columns = [index[name] for name in COORDINATE_COLUMNS]
    time_column = index.get("t_ms")
    values, stamps = [], []
    for row in reader:
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError(f"Wrong column count at CSV line {reader.line_num}")
        values.append([row[i].strip() or "nan" for i in columns])
        if time_column is not None:
            stamps.append(row[time_column].strip() or "nan")
    return landmarks_to_tensor(values, frames, stamps if time_column is not None else None)
