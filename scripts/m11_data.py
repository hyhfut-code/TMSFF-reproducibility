"""CitySim preparation for the M11 TMSFF versus AV-IDM audit.

The cache contains only deterministic transformations of the source files.  The
route map is reconstructed from vehicles that are not target followers, so the
30% follower holdout block is never used to define the AV-IDM road geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "scripts" / "m11_protocol.json"
DEFAULT_CACHE = PROJECT_ROOT / "reproduced_outputs" / "cache" / "m11_citysim_cache_k16.npz"
DEFAULT_METADATA = PROJECT_ROOT / "reproduced_outputs" / "cache" / "m11_citysim_cache_metadata.json"
ROUTE_POINTS = 201


def sha256_file(path: Path, block_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    pairs = Path(protocol["data"]["selected_pairs"])
    if not pairs.is_absolute():
        protocol["data"]["selected_pairs"] = str(PROJECT_ROOT / pairs)
    return protocol


def cumulative_station(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ds = np.hypot(np.diff(x), np.diff(y))
    return np.r_[0.0, np.cumsum(ds)]


def resample_track(track: pd.DataFrame, progress: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = track["carCenterXm"].to_numpy(float)
    y = track["carCenterYm"].to_numpy(float)
    station = cumulative_station(x, y)
    keep = np.r_[True, np.diff(station) > 1e-6]
    x = x[keep]
    y = y[keep]
    station = station[keep]
    if len(station) < 2 or station[-1] < 1.0:
        raise ValueError("Track has insufficient displacement for route reconstruction")
    normalized = station / station[-1]
    return np.interp(progress, normalized, x), np.interp(progress, normalized, y)


def smooth_series(values: np.ndarray, window: int = 21, order: int = 3) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    return savgol_filter(values, window_length=window, polyorder=order, mode="interp")


def vehicle_dimensions(raw_path: Path) -> pd.DataFrame:
    columns = [
        "carId",
        "boundingBox1Xft",
        "boundingBox1Yft",
        "boundingBox2Xft",
        "boundingBox2Yft",
        "boundingBox4Xft",
        "boundingBox4Yft",
    ]
    raw = pd.read_csv(raw_path, usecols=columns)
    p1 = raw[["boundingBox1Xft", "boundingBox1Yft"]].to_numpy(float)
    p2 = raw[["boundingBox2Xft", "boundingBox2Yft"]].to_numpy(float)
    p4 = raw[["boundingBox4Xft", "boundingBox4Yft"]].to_numpy(float)
    edge_12 = np.linalg.norm(p1 - p2, axis=1) * 0.3048
    edge_14 = np.linalg.norm(p1 - p4, axis=1) * 0.3048
    raw["length_m"] = np.maximum(edge_12, edge_14)
    raw["width_m"] = np.minimum(edge_12, edge_14)
    raw.loc[~raw["length_m"].between(3.0, 10.0), "length_m"] = np.nan
    raw.loc[~raw["width_m"].between(1.3, 3.5), "width_m"] = np.nan
    dimensions = raw.groupby("carId", as_index=False)[["length_m", "width_m"]].median()
    dimensions["length_m"] = dimensions["length_m"].fillna(5.0)
    dimensions["width_m"] = dimensions["width_m"].fillna(2.1)
    return dimensions


def track_routes(tracks: pd.DataFrame) -> pd.DataFrame:
    ordered = tracks.sort_values(["carId", "frameNum"])
    summary = ordered.groupby("carId").agg(
        entry_lane=("laneId", "first"),
        exit_lane=("laneId", "last"),
        frames=("frameNum", "size"),
        x_first=("carCenterXm", "first"),
        x_last=("carCenterXm", "last"),
        y_first=("carCenterYm", "first"),
        y_last=("carCenterYm", "last"),
    )
    summary["displacement_m"] = np.hypot(
        summary["x_last"] - summary["x_first"],
        summary["y_last"] - summary["y_first"],
    )
    return summary


def build_route_map(
    tracks_by_id: dict[int, pd.DataFrame],
    route_summary: pd.DataFrame,
    dimensions: pd.DataFrame,
    entry_lane: int,
    exit_lane: int,
    excluded_followers: set[int],
) -> dict[str, np.ndarray | int | float]:
    eligible = route_summary[
        (route_summary["entry_lane"] == entry_lane)
        & (route_summary["exit_lane"] == exit_lane)
        & (route_summary["frames"] >= 100)
        & (route_summary["displacement_m"] >= 20.0)
    ]
    source_ids = [int(i) for i in eligible.index if int(i) not in excluded_followers]
    if len(source_ids) < 2:
        raise RuntimeError(f"Route {entry_lane}->{exit_lane} has fewer than two non-target tracks")

    progress = np.linspace(0.0, 1.0, ROUTE_POINTS)
    x_samples: list[np.ndarray] = []
    y_samples: list[np.ndarray] = []
    used_ids: list[int] = []
    for car_id in source_ids:
        try:
            x, y = resample_track(tracks_by_id[car_id], progress)
        except ValueError:
            continue
        x_samples.append(x)
        y_samples.append(y)
        used_ids.append(car_id)
    if len(used_ids) < 2:
        raise RuntimeError(f"Route {entry_lane}->{exit_lane} has insufficient usable tracks")

    x_matrix = np.vstack(x_samples)
    y_matrix = np.vstack(y_samples)
    route_x = smooth_series(np.median(x_matrix, axis=0))
    route_y = smooth_series(np.median(y_matrix, axis=0))
    dx = np.gradient(route_x)
    dy = np.gradient(route_y)
    heading = np.unwrap(np.arctan2(dy, dx))
    normal_x = -np.sin(heading)
    normal_y = np.cos(heading)
    lateral = (x_matrix - route_x) * normal_x + (y_matrix - route_y) * normal_y

    width_lookup = dimensions.set_index("carId")["width_m"]
    median_half_vehicle_width = float(
        np.nanmedian([width_lookup.get(car_id, 2.1) / 2.0 for car_id in used_ids])
    )
    left_width = np.quantile(lateral, 0.95, axis=0) + median_half_vehicle_width + 0.35
    right_width = -np.quantile(lateral, 0.05, axis=0) + median_half_vehicle_width + 0.35
    left_width = np.clip(smooth_series(left_width), 2.5, 12.0)
    right_width = np.clip(smooth_series(right_width), 2.5, 12.0)
    station = cumulative_station(route_x, route_y)

    return {
        "x": route_x.astype(np.float64),
        "y": route_y.astype(np.float64),
        "heading": heading.astype(np.float64),
        "left_width": left_width.astype(np.float64),
        "right_width": right_width.astype(np.float64),
        "station": station.astype(np.float64),
        "source_count": len(used_ids),
        "median_left_width_m": float(np.median(left_width)),
        "median_right_width_m": float(np.median(right_width)),
    }


def nearest_route_station(x: float, y: float, route: dict[str, np.ndarray]) -> float:
    distances = (route["x"] - x) ** 2 + (route["y"] - y) ** 2
    return float(route["station"][int(np.argmin(distances))])


def prepare_cache(
    protocol_path: Path = DEFAULT_PROTOCOL,
    cache_path: Path = DEFAULT_CACHE,
    metadata_path: Path = DEFAULT_METADATA,
    maximum_candidates: int | None = None,
) -> None:
    protocol = load_protocol(protocol_path)
    maximum_candidates = maximum_candidates or int(
        protocol["perception"]["maximum_dynamic_candidates"]
    )
    cleaned_path = Path(protocol["data"]["cleaned_tracks"])
    raw_path = Path(protocol["data"]["raw_tracks"])
    pairs_path = Path(protocol["data"]["selected_pairs"])
    calibration_fraction = float(protocol["data"]["calibration_fraction"])
    radius = float(protocol["perception"]["candidate_radius_m"])

    dimensions = vehicle_dimensions(raw_path)
    tracks = pd.read_csv(cleaned_path).sort_values(["carId", "frameNum"]).reset_index(drop=True)
    tracks = tracks.merge(dimensions, on="carId", how="left")
    tracks["length_m"] = tracks["length_m"].fillna(5.0)
    tracks["width_m"] = tracks["width_m"].fillna(2.1)
    pairs = pd.read_csv(pairs_path)
    if len(pairs) != int(protocol["data"]["trajectory_count"]):
        raise RuntimeError("Selected-pair count does not match the frozen protocol")

    tracks_by_id = {
        int(car_id): group.reset_index(drop=True)
        for car_id, group in tracks.groupby("carId", sort=False)
    }
    route_summary = track_routes(tracks)
    excluded_followers = set(int(v) for v in pairs["follower_id"])
    unique_routes = (
        pairs[["entry_lane", "exit_lane"]]
        .drop_duplicates()
        .sort_values(["entry_lane", "exit_lane"])
    )
    route_maps: dict[tuple[int, int], dict] = {}
    for row in unique_routes.itertuples(index=False):
        key = (int(row.entry_lane), int(row.exit_lane))
        route_maps[key] = build_route_map(
            tracks_by_id,
            route_summary,
            dimensions,
            key[0],
            key[1],
            excluded_followers,
        )

    frame_groups: dict[int, np.ndarray] = {}
    environment_columns = [
        "carId",
        "carCenterXm",
        "carCenterYm",
        "velocity",
        "heading",
        "length_m",
        "width_m",
    ]
    for frame, group in tracks.groupby("frameNum", sort=False):
        frame_groups[int(frame)] = group[environment_columns].to_numpy(float)

    pair_records: list[dict] = []
    maximum_frames = 0
    for selection in pairs.itertuples(index=False):
        follower = tracks_by_id[int(selection.follower_id)]
        leader = tracks_by_id[int(selection.leader_id)]
        follower = follower[follower["frameNum"].between(selection.start_frame, selection.end_frame)]
        leader = leader[leader["frameNum"].between(selection.start_frame, selection.end_frame)]
        pair = follower.merge(leader, on="frameNum", suffixes=("_follower", "_leader"))
        pair = pair.sort_values("frameNum").reset_index(drop=True)
        if len(pair) < 20:
            raise RuntimeError(f"Pair {selection.leader_id}->{selection.follower_id} is too short")
        maximum_frames = max(maximum_frames, len(pair))
        pair_records.append({"selection": selection, "pair": pair})

    pair_count = len(pair_records)
    observed = np.full((pair_count, maximum_frames, 6), np.nan, dtype=np.float64)
    leader_state = np.full((pair_count, maximum_frames, 4), np.nan, dtype=np.float64)
    environment = np.full(
        (pair_count, maximum_frames, maximum_candidates, 7), np.nan, dtype=np.float64
    )
    environment[..., 6] = 0.0
    frame_numbers = np.full((pair_count, maximum_frames), -1, dtype=np.int32)
    lengths = np.zeros(pair_count, dtype=np.int32)
    split_indices = np.zeros(pair_count, dtype=np.int32)
    pair_metadata = np.zeros((pair_count, 7), dtype=np.int32)
    route_x = np.zeros((pair_count, ROUTE_POINTS), dtype=np.float64)
    route_y = np.zeros_like(route_x)
    route_heading = np.zeros_like(route_x)
    route_left_width = np.zeros_like(route_x)
    route_right_width = np.zeros_like(route_x)
    route_station = np.zeros_like(route_x)
    candidate_counts: list[int] = []
    replaced_for_leader = 0

    for pair_index, record in enumerate(pair_records):
        selection = record["selection"]
        pair = record["pair"]
        n = len(pair)
        split = max(2, min(n - 2, int(np.floor(calibration_fraction * n))))
        lengths[pair_index] = n
        split_indices[pair_index] = split
        pair_metadata[pair_index] = np.array(
            [
                int(selection.trajectory_no),
                int(selection.leader_id),
                int(selection.follower_id),
                int(selection.entry_lane),
                int(selection.exit_lane),
                int(selection.start_frame),
                int(selection.end_frame),
            ],
            dtype=np.int32,
        )
        key = (int(selection.entry_lane), int(selection.exit_lane))
        route = route_maps[key]
        route_x[pair_index] = route["x"]
        route_y[pair_index] = route["y"]
        route_heading[pair_index] = route["heading"]
        route_left_width[pair_index] = route["left_width"]
        route_right_width[pair_index] = route["right_width"]
        route_station[pair_index] = route["station"]

        observed[pair_index, :n, 0] = pair["carCenterXm_follower"]
        observed[pair_index, :n, 1] = pair["carCenterYm_follower"]
        observed[pair_index, :n, 2] = pair["velocity_follower"]
        observed[pair_index, :n, 3] = pair["heading_follower"]
        observed[pair_index, :n, 4] = pair["acceleration_follower"]
        observed[pair_index, :n, 5] = [
            nearest_route_station(float(x), float(y), route)
            for x, y in zip(pair["carCenterXm_follower"], pair["carCenterYm_follower"])
        ]
        leader_state[pair_index, :n, 0] = pair["carCenterXm_leader"]
        leader_state[pair_index, :n, 1] = pair["carCenterYm_leader"]
        leader_state[pair_index, :n, 2] = pair["velocity_leader"]
        leader_state[pair_index, :n, 3] = pair["heading_leader"]
        frame_numbers[pair_index, :n] = pair["frameNum"].to_numpy(np.int32)

        follower_id = int(selection.follower_id)
        leader_id = int(selection.leader_id)
        for time_index, pair_row in pair.iterrows():
            frame = int(pair_row["frameNum"])
            all_states = frame_groups[frame]
            eligible = all_states[all_states[:, 0] != follower_id]
            distances = np.hypot(
                eligible[:, 1] - float(pair_row["carCenterXm_follower"]),
                eligible[:, 2] - float(pair_row["carCenterYm_follower"]),
            )
            within = np.flatnonzero(distances <= radius)
            ordered_indices = within[np.argsort(distances[within])]
            selected_indices = ordered_indices[:maximum_candidates].tolist()
            leader_positions = np.flatnonzero(eligible[:, 0].astype(int) == leader_id)
            if len(leader_positions):
                leader_position = int(leader_positions[0])
                if leader_position not in selected_indices:
                    replaced_for_leader += 1
                    if len(selected_indices) >= maximum_candidates:
                        selected_indices[-1] = leader_position
                    else:
                        selected_indices.append(leader_position)
            selected_states = eligible[selected_indices]
            candidate_counts.append(len(selected_states))
            count = len(selected_states)
            if count:
                environment[pair_index, time_index, :count, :6] = selected_states[:, 1:7]
                environment[pair_index, time_index, :count, 6] = 1.0

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        observed=observed,
        leader=leader_state,
        environment=environment,
        frame_numbers=frame_numbers,
        lengths=lengths,
        split_indices=split_indices,
        pair_metadata=pair_metadata,
        route_x=route_x,
        route_y=route_y,
        route_heading=route_heading,
        route_left_width=route_left_width,
        route_right_width=route_right_width,
        route_station=route_station,
        dt=np.array([1.0 / float(protocol["data"]["sampling_frequency_hz"])]),
        maximum_candidates=np.array([maximum_candidates], dtype=np.int32),
    )

    route_audit = []
    for key, route in route_maps.items():
        route_audit.append(
            {
                "entry_lane": key[0],
                "exit_lane": key[1],
                "non_target_source_tracks": int(route["source_count"]),
                "median_left_boundary_distance_m": route["median_left_width_m"],
                "median_right_boundary_distance_m": route["median_right_width_m"],
            }
        )
    metadata = {
        "cache_file": str(cache_path),
        "source_files": {
            "cleaned_tracks": {"path": str(cleaned_path), "sha256": sha256_file(cleaned_path)},
            "raw_tracks": {"path": str(raw_path), "sha256": sha256_file(raw_path)},
            "selected_pairs": {"path": str(pairs_path), "sha256": sha256_file(pairs_path)},
        },
        "pair_count": pair_count,
        "total_frames": int(lengths.sum()),
        "calibration_frames": int(split_indices.sum()),
        "holdout_frames": int((lengths - split_indices).sum()),
        "maximum_frames_per_pair": int(maximum_frames),
        "maximum_dynamic_candidates": int(maximum_candidates),
        "candidate_count_mean": float(np.mean(candidate_counts)),
        "candidate_count_min": int(np.min(candidate_counts)),
        "candidate_count_max": int(np.max(candidate_counts)),
        "frames_where_leader_was_forced_into_candidate_set": int(replaced_for_leader),
        "target_follower_ids_excluded_from_route_map": sorted(excluded_followers),
        "route_reconstruction": route_audit,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--maximum-candidates", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    prepare_cache(
        protocol_path=arguments.protocol,
        cache_path=arguments.cache,
        metadata_path=arguments.metadata,
        maximum_candidates=arguments.maximum_candidates,
    )
