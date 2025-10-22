"""Camera trajectory generation utilities and ComfyUI node for GEN3C workflows.

This module is duplicated/adapted from native ComfyUI concepts but extended to output
structured trajectories suitable for GEN3C camera control and Gaussian Splat datasets.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    near: float
    far: float

    def as_matrix(self) -> List[List[float]]:
        return [
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ]


@dataclass
class CameraExtrinsics:
    rotation: np.ndarray  # shape (3, 3)
    translation: np.ndarray  # shape (3,)

    def camera_to_world(self) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = self.translation
        return transform

    def world_to_camera(self) -> np.ndarray:
        world_to_cam = np.eye(4, dtype=np.float64)
        rot = self.rotation
        trans = self.translation.reshape(3, 1)
        world_to_cam[:3, :3] = rot.T
        world_to_cam[:3, 3] = (-rot.T @ trans).reshape(3,)
        return world_to_cam


@dataclass
class CameraFrame:
    frame_index: int
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics

    def to_payload(self) -> Dict[str, object]:
        cam_to_world = self.extrinsics.camera_to_world()
        world_to_cam = self.extrinsics.world_to_camera()
        return {
            "frame": self.frame_index,
            "intrinsics": self.intrinsics.as_matrix(),
            "extrinsics": {
                "camera_to_world": cam_to_world.tolist(),
                "world_to_camera": world_to_cam.tolist(),
            },
            "width": self.intrinsics.width,
            "height": self.intrinsics.height,
            "near": self.intrinsics.near,
            "far": self.intrinsics.far,
        }


# ---------------------------------------------------------------------------
# Maths helpers (duplicated/adapted from native ComfyUI camera utilities)
# ---------------------------------------------------------------------------


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError("Cannot normalize vector with near-zero magnitude.")
    return vec / norm


def _look_at(
    position: np.ndarray,
    target: np.ndarray,
    up_hint: np.ndarray,
    handedness: Literal["right", "left"],
) -> Tuple[np.ndarray, np.ndarray]:
    forward = _normalize(target - position)
    if handedness == "right":
        right = _normalize(np.cross(forward, up_hint))
    else:
        right = _normalize(np.cross(up_hint, forward))
    up = _normalize(np.cross(right, forward))

    if handedness == "right":
        rotation = np.stack([right, up, -forward], axis=1)
    else:
        rotation = np.stack([right, up, forward], axis=1)

    return rotation.astype(np.float64), position.astype(np.float64)


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------


def _generate_angles(total_frames: int, turns: float, start_angle: float = 0.0) -> Iterable[float]:
    span = turns * 2.0 * math.pi
    for idx in range(total_frames):
        yield start_angle + span * (idx / max(total_frames, 1))


def _orbit(
    total_frames: int,
    radius: float,
    height: float,
    target: np.ndarray,
    up: np.ndarray,
    handedness: Literal["right", "left"],
    turns: float = 1.0,
    start_angle: float = 0.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    frames: List[Tuple[np.ndarray, np.ndarray]] = []
    for angle in _generate_angles(total_frames, turns, start_angle):
        position = np.array([
            math.cos(angle) * radius,
            height,
            math.sin(angle) * radius,
        ], dtype=np.float64)
        rotation, translation = _look_at(position, target, up, handedness)
        frames.append((rotation, translation))
    return frames


def _dolly(
    total_frames: int,
    start_distance: float,
    end_distance: float,
    height: float,
    target: np.ndarray,
    up: np.ndarray,
    handedness: Literal["right", "left"],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    frames: List[Tuple[np.ndarray, np.ndarray]] = []
    for idx in range(total_frames):
        t = idx / max(total_frames - 1, 1)
        distance = (1.0 - t) * start_distance + t * end_distance
        position = np.array([0.0, height, distance], dtype=np.float64)
        rotation, translation = _look_at(position, target, up, handedness)
        frames.append((rotation, translation))
    return frames


def _truck(
    total_frames: int,
    span: float,
    depth: float,
    height: float,
    target_offset: np.ndarray,
    up: np.ndarray,
    handedness: Literal["right", "left"],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    frames: List[Tuple[np.ndarray, np.ndarray]] = []
    start = -span * 0.5
    end = span * 0.5
    for idx in range(total_frames):
        t = idx / max(total_frames - 1, 1)
        x_pos = (1.0 - t) * start + t * end
        position = np.array([x_pos, height, depth], dtype=np.float64)
        rotation, translation = _look_at(position, target_offset, up, handedness)
        frames.append((rotation, translation))
    return frames


def _tilt(
    total_frames: int,
    radius: float,
    height: float,
    target: np.ndarray,
    up: np.ndarray,
    handedness: Literal["right", "left"],
    tilt_degrees: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    frames: List[Tuple[np.ndarray, np.ndarray]] = []
    base_position = np.array([0.0, height, radius], dtype=np.float64)
    tilt_radians = math.radians(tilt_degrees)
    for idx in range(total_frames):
        t = idx / max(total_frames - 1, 1)
        offset = math.sin((t * 2.0 - 1.0) * 0.5 * math.pi) * tilt_radians
        target_offset = target + np.array([0.0, math.tan(offset) * radius, 0.0])
        rotation, translation = _look_at(base_position, target_offset, up, handedness)
        frames.append((rotation, translation))
    return frames


def _spiral(
    total_frames: int,
    start_radius: float,
    end_radius: float,
    height: float,
    target: np.ndarray,
    up: np.ndarray,
    handedness: Literal["right", "left"],
    turns: float = 2.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    frames: List[Tuple[np.ndarray, np.ndarray]] = []
    for idx, angle in enumerate(_generate_angles(total_frames, turns)):
        t = idx / max(total_frames - 1, 1)
        radius = (1.0 - t) * start_radius + t * end_radius
        position = np.array([
            math.cos(angle) * radius,
            height + math.sin(angle * 0.5) * height * 0.25,
            math.sin(angle) * radius,
        ], dtype=np.float64)
        rotation, translation = _look_at(position, target, up, handedness)
        frames.append((rotation, translation))
    return frames


def _dolly_out(
    total_frames: int,
    start_distance: float,
    end_distance: float,
    height: float,
    target: np.ndarray,
    up: np.ndarray,
    handedness: Literal["right", "left"],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Dolly out (push out from target)."""
    return _dolly(total_frames, start_distance, end_distance, height, target, up, handedness)


def _truck_right(
    total_frames: int,
    span: float,
    depth: float,
    height: float,
    target_offset: np.ndarray,
    up: np.ndarray,
    handedness: Literal["right", "left"],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Truck right (move right while looking at target)."""
    frames: List[Tuple[np.ndarray, np.ndarray]] = []
    start = span * 0.5
    end = -span * 0.5
    for idx in range(total_frames):
        t = idx / max(total_frames - 1, 1)
        x_pos = (1.0 - t) * start + t * end
        position = np.array([x_pos, height, depth], dtype=np.float64)
        rotation, translation = _look_at(position, target_offset, up, handedness)
        frames.append((rotation, translation))
    return frames


def _crane_up(
    total_frames: int,
    start_height: float,
    end_height: float,
    radius: float,
    target: np.ndarray,
    up: np.ndarray,
    handedness: Literal["right", "left"],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Crane up (vertical lift while looking at target)."""
    frames: List[Tuple[np.ndarray, np.ndarray]] = []
    for idx in range(total_frames):
        t = idx / max(total_frames - 1, 1)
        height = (1.0 - t) * start_height + t * end_height
        position = np.array([0.0, height, radius], dtype=np.float64)
        rotation, translation = _look_at(position, target, up, handedness)
        frames.append((rotation, translation))
    return frames


def _crane_down(
    total_frames: int,
    start_height: float,
    end_height: float,
    radius: float,
    target: np.ndarray,
    up: np.ndarray,
    handedness: Literal["right", "left"],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Crane down (vertical descent while looking at target)."""
    return _crane_up(total_frames, start_height, end_height, radius, target, up, handedness)


def _arc_left(
    total_frames: int,
    radius: float,
    height: float,
    target: np.ndarray,
    up: np.ndarray,
    handedness: Literal["right", "left"],
    arc_degrees: float = 90.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Arc left (partial orbit from front to left side)."""
    frames: List[Tuple[np.ndarray, np.ndarray]] = []
    start_angle = math.pi  # Start at front (facing -Z)
    arc_radians = math.radians(arc_degrees)
    for idx in range(total_frames):
        t = idx / max(total_frames - 1, 1)
        angle = start_angle + arc_radians * t
        position = np.array([
            math.cos(angle) * radius,
            height,
            math.sin(angle) * radius,
        ], dtype=np.float64)
        rotation, translation = _look_at(position, target, up, handedness)
        frames.append((rotation, translation))
    return frames


def _arc_right(
    total_frames: int,
    radius: float,
    height: float,
    target: np.ndarray,
    up: np.ndarray,
    handedness: Literal["right", "left"],
    arc_degrees: float = 90.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Arc right (partial orbit from front to right side)."""
    frames: List[Tuple[np.ndarray, np.ndarray]] = []
    start_angle = math.pi  # Start at front (facing -Z)
    arc_radians = math.radians(arc_degrees)
    for idx in range(total_frames):
        t = idx / max(total_frames - 1, 1)
        angle = start_angle - arc_radians * t
        position = np.array([
            math.cos(angle) * radius,
            height,
            math.sin(angle) * radius,
        ], dtype=np.float64)
        rotation, translation = _look_at(position, target, up, handedness)
        frames.append((rotation, translation))
    return frames


def _boom_shot(
    total_frames: int,
    start_radius: float,
    end_radius: float,
    start_height: float,
    end_height: float,
    target: np.ndarray,
    up: np.ndarray,
    handedness: Literal["right", "left"],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Boom shot (combined dolly and crane motion)."""
    frames: List[Tuple[np.ndarray, np.ndarray]] = []
    for idx in range(total_frames):
        t = idx / max(total_frames - 1, 1)
        # Ease in/out curve
        t_ease = 0.5 - 0.5 * math.cos(t * math.pi)
        radius = (1.0 - t_ease) * start_radius + t_ease * end_radius
        height = (1.0 - t_ease) * start_height + t_ease * end_height
        position = np.array([0.0, height, radius], dtype=np.float64)
        rotation, translation = _look_at(position, target, up, handedness)
        frames.append((rotation, translation))
    return frames


def _figure_eight(
    total_frames: int,
    radius: float,
    height: float,
    target: np.ndarray,
    up: np.ndarray,
    handedness: Literal["right", "left"],
    loops: float = 1.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Figure-8 pattern around target."""
    frames: List[Tuple[np.ndarray, np.ndarray]] = []
    for idx in range(total_frames):
        t = (idx / max(total_frames - 1, 1)) * loops
        angle = t * 2.0 * math.pi
        x = radius * math.sin(angle)
        z = radius * math.sin(angle * 2.0) * 0.5
        position = np.array([x, height, z], dtype=np.float64)
        rotation, translation = _look_at(position, target, up, handedness)
        frames.append((rotation, translation))
    return frames


def _hemisphere(
    total_frames: int,
    radius: float,
    target: np.ndarray,
    up: np.ndarray,
    handedness: Literal["right", "left"],
    elevation_range: float = 60.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Hemisphere orbit (varies both horizontal angle and elevation)."""
    frames: List[Tuple[np.ndarray, np.ndarray]] = []
    max_elevation = math.radians(elevation_range)
    for idx in range(total_frames):
        t = idx / max(total_frames - 1, 1)
        azimuth = t * 2.0 * math.pi
        elevation = math.sin(t * 2.0 * math.pi) * max_elevation
        x = radius * math.cos(elevation) * math.cos(azimuth)
        y = radius * math.sin(elevation)
        z = radius * math.cos(elevation) * math.sin(azimuth)
        position = np.array([x, y, z], dtype=np.float64)
        rotation, translation = _look_at(position, target, up, handedness)
        frames.append((rotation, translation))
    return frames


def _interpolate_keyframes(
    keyframes: List[Dict[str, object]],
    total_frames: int,
    up: np.ndarray,
    handedness: Literal["right", "left"],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if not keyframes:
        raise ValueError("At least one keyframe is required for custom trajectories.")

    frames: List[Tuple[np.ndarray, np.ndarray]] = []
    sorted_keyframes = sorted(keyframes, key=lambda item: int(item.get("frame", 0)))
    positions = [np.array(kf.get("position", [0.0, 0.0, 0.0]), dtype=np.float64) for kf in sorted_keyframes]
    targets = [np.array(kf.get("target", [0.0, 0.0, 0.0]), dtype=np.float64) for kf in sorted_keyframes]
    frame_markers = [int(kf.get("frame", idx * (total_frames - 1) // max(len(sorted_keyframes) - 1, 1))) for idx, kf in enumerate(sorted_keyframes)]

    for idx in range(total_frames):
        if idx <= frame_markers[0]:
            segment = 0
            segment_t = 0.0
        elif idx >= frame_markers[-1]:
            segment = len(frame_markers) - 2
            segment_t = 1.0
        else:
            segment = next(i for i in range(len(frame_markers) - 1) if frame_markers[i] <= idx <= frame_markers[i + 1])
            span = frame_markers[segment + 1] - frame_markers[segment]
            segment_t = (idx - frame_markers[segment]) / max(span, 1)

        position = (1.0 - segment_t) * positions[segment] + segment_t * positions[segment + 1]
        target = (1.0 - segment_t) * targets[segment] + segment_t * targets[segment + 1]
        rotation, translation = _look_at(position, target, up, handedness)
        frames.append((rotation, translation))

    return frames


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


PRESET_NAMES = (
    "orbit",
    "dolly_in",
    "dolly_out",
    "truck_left",
    "truck_right",
    "crane_up",
    "crane_down",
    "arc_left",
    "arc_right",
    "tilt",
    "spiral",
    "boom_shot",
    "figure_eight",
    "hemisphere",
    "custom"
)


def build_intrinsics(
    width: int,
    height: int,
    fov_degrees: float,
    principal_x: float,
    principal_y: float,
    near: float,
    far: float,
) -> CameraIntrinsics:
    fov_radians = math.radians(max(min(fov_degrees, 175.0), 1.0))
    focal = 0.5 * width / math.tan(fov_radians * 0.5)
    fx = focal
    fy = focal * (height / max(width, 1))
    cx = width * 0.5 + principal_x
    cy = height * 0.5 + principal_y
    return CameraIntrinsics(width, height, fx, fy, cx, cy, near, far)


def generate_trajectory(
    preset: str,
    total_frames: int,
    intrinsics: CameraIntrinsics,
    handedness: Literal["right", "left"],
    orbit_radius: float = 4.0,
    orbit_height: float = 0.0,
    dolly_start: float = 8.0,
    dolly_end: float = 3.5,
    truck_span: float = 4.0,
    truck_depth: float = 6.0,
    crane_start_height: float = 1.0,
    crane_end_height: float = 5.0,
    arc_degrees: float = 90.0,
    tilt_degrees: float = 30.0,
    spiral_start: float = 6.0,
    spiral_end: float = 2.5,
    boom_start_radius: float = 8.0,
    boom_end_radius: float = 3.0,
    boom_start_height: float = 1.0,
    boom_end_height: float = 5.0,
    orbit_turns: float = 1.0,
    hemisphere_elevation: float = 60.0,
    keyframes_json: Optional[str] = None,
) -> List[CameraFrame]:
    # Validate inputs
    if total_frames < 1:
        raise ValueError(f"total_frames must be >= 1, got {total_frames}")

    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    if preset not in PRESET_NAMES:
        raise ValueError(f"Unknown preset '{preset}'. Supported: {', '.join(PRESET_NAMES)}")

    if preset == "orbit":
        rig = _orbit(total_frames, orbit_radius, orbit_height, target, up, handedness, turns=orbit_turns)
    elif preset == "dolly_in":
        rig = _dolly(total_frames, dolly_start, dolly_end, orbit_height, target, up, handedness)
    elif preset == "dolly_out":
        rig = _dolly_out(total_frames, dolly_end, dolly_start, orbit_height, target, up, handedness)
    elif preset == "truck_left":
        rig = _truck(total_frames, truck_span, truck_depth, orbit_height, target, up, handedness)
    elif preset == "truck_right":
        rig = _truck_right(total_frames, truck_span, truck_depth, orbit_height, target, up, handedness)
    elif preset == "crane_up":
        rig = _crane_up(total_frames, crane_start_height, crane_end_height, orbit_radius, target, up, handedness)
    elif preset == "crane_down":
        rig = _crane_down(total_frames, crane_end_height, crane_start_height, orbit_radius, target, up, handedness)
    elif preset == "arc_left":
        rig = _arc_left(total_frames, orbit_radius, orbit_height, target, up, handedness, arc_degrees)
    elif preset == "arc_right":
        rig = _arc_right(total_frames, orbit_radius, orbit_height, target, up, handedness, arc_degrees)
    elif preset == "tilt":
        rig = _tilt(total_frames, orbit_radius, orbit_height, target, up, handedness, tilt_degrees)
    elif preset == "spiral":
        rig = _spiral(total_frames, spiral_start, spiral_end, orbit_height, target, up, handedness, turns=max(orbit_turns, 1.0))
    elif preset == "boom_shot":
        rig = _boom_shot(total_frames, boom_start_radius, boom_end_radius, boom_start_height, boom_end_height, target, up, handedness)
    elif preset == "figure_eight":
        rig = _figure_eight(total_frames, orbit_radius, orbit_height, target, up, handedness, loops=orbit_turns)
    elif preset == "hemisphere":
        rig = _hemisphere(total_frames, orbit_radius, target, up, handedness, hemisphere_elevation)
    elif preset == "custom":
        try:
            keyframes = json.loads(keyframes_json or "[]")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse keyframes JSON: {exc}")
        rig = _interpolate_keyframes(keyframes, total_frames, up, handedness)
    else:
        raise AssertionError("Unhandled preset")

    frames: List[CameraFrame] = []
    for idx, (rotation, translation) in enumerate(rig):
        extrinsics = CameraExtrinsics(rotation, translation)
        frames.append(CameraFrame(idx, intrinsics, extrinsics))
    return frames


__all__ = [
    "CameraIntrinsics",
    "CameraExtrinsics",
    "CameraFrame",
    "PRESET_NAMES",
    "build_intrinsics",
    "generate_trajectory",
]
