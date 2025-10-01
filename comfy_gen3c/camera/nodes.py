"""ComfyUI nodes for camera tooling dedicated to GEN3C workflows."""

from __future__ import annotations

import json
from typing import Any, Dict

from .trajectory import (
    PRESET_NAMES,
    build_intrinsics,
    generate_trajectory,
)


class Gen3CCameraTrajectory:
    """Generates per-frame intrinsics and extrinsics for GEN3C control."""

    @classmethod
    def INPUT_TYPES(cls):
        presets = list(PRESET_NAMES)
        return {
            "required": {
                "frames": ("INT", {"default": 121, "min": 1, "max": 4096, "tooltip": "Total number of frames for trajectory"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "tooltip": "Frames per second for video generation"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8, "tooltip": "Frame width in pixels (must be multiple of 8)"}),
                "height": ("INT", {"default": 576, "min": 64, "max": 4096, "step": 8, "tooltip": "Frame height in pixels (must be multiple of 8)"}),
                "fov_degrees": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 175.0, "step": 0.1, "tooltip": "Horizontal field of view in degrees"}),
                "principal_x": ("FLOAT", {"default": 0.0, "min": -1024.0, "max": 1024.0, "step": 0.1, "tooltip": "Horizontal principal point offset (0 = center)"}),
                "principal_y": ("FLOAT", {"default": 0.0, "min": -1024.0, "max": 1024.0, "step": 0.1, "tooltip": "Vertical principal point offset (0 = center)"}),
                "near_plane": ("FLOAT", {"default": 0.01, "min": 1e-4, "max": 100.0, "step": 0.001, "tooltip": "Near clipping plane distance"}),
                "far_plane": ("FLOAT", {"default": 1000.0, "min": 1.0, "max": 10000.0, "step": 0.5, "tooltip": "Far clipping plane distance"}),
                "handedness": (("right", "left"), {"default": "right", "tooltip": "Coordinate system: 'right' (OpenGL/Nerfstudio) or 'left' (DirectX)"}),
                "preset": (presets, {"default": "orbit", "tooltip": "Camera motion preset - choose from 14 cinematic camera moves"}),
                "orbit_radius": ("FLOAT", {"default": 6.0, "min": 0.1, "max": 200.0, "tooltip": "Orbit/Arc/Hemisphere: radius from center point"}),
                "orbit_height": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 50.0, "tooltip": "Height above ground plane (used by most presets)"}),
                "orbit_turns": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "tooltip": "Orbit/Spiral/Figure-8: number of complete rotations/loops"}),
                "dolly_start": ("FLOAT", {"default": 12.0, "min": 0.5, "max": 200.0, "tooltip": "Dolly In/Out: starting distance from target"}),
                "dolly_end": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 200.0, "tooltip": "Dolly In/Out: ending distance from target"}),
                "truck_span": ("FLOAT", {"default": 6.0, "min": 0.1, "max": 200.0, "tooltip": "Truck Left/Right: total lateral movement distance"}),
                "truck_depth": ("FLOAT", {"default": 10.0, "min": 0.5, "max": 200.0, "tooltip": "Truck Left/Right: distance from target"}),
                "crane_start_height": ("FLOAT", {"default": 1.0, "min": -50.0, "max": 50.0, "tooltip": "Crane Up/Down: starting height"}),
                "crane_end_height": ("FLOAT", {"default": 5.0, "min": -50.0, "max": 50.0, "tooltip": "Crane Up/Down: ending height"}),
                "arc_degrees": ("FLOAT", {"default": 90.0, "min": 15.0, "max": 180.0, "tooltip": "Arc Left/Right: arc sweep angle in degrees (90° = quarter circle)"}),
                "tilt_degrees": ("FLOAT", {"default": 25.0, "min": -89.0, "max": 89.0, "tooltip": "Tilt: vertical rotation angle in degrees"}),
                "spiral_start": ("FLOAT", {"default": 10.0, "min": 0.5, "max": 200.0, "tooltip": "Spiral: starting radius"}),
                "spiral_end": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 200.0, "tooltip": "Spiral: ending radius"}),
                "boom_start_radius": ("FLOAT", {"default": 8.0, "min": 0.5, "max": 200.0, "tooltip": "Boom Shot: starting distance from target"}),
                "boom_end_radius": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 200.0, "tooltip": "Boom Shot: ending distance from target"}),
                "boom_start_height": ("FLOAT", {"default": 1.0, "min": -50.0, "max": 50.0, "tooltip": "Boom Shot: starting height"}),
                "boom_end_height": ("FLOAT", {"default": 5.0, "min": -50.0, "max": 50.0, "tooltip": "Boom Shot: ending height"}),
                "hemisphere_elevation": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 90.0, "tooltip": "Hemisphere: maximum elevation angle in degrees"}),
            },
            "optional": {
                "keyframes_json": ("STRING", {"multiline": True, "default": "[]", "tooltip": "Custom keyframes as JSON array: [{\"frame\": 0, \"position\": [x,y,z], \"target\": [x,y,z]}]"}),
            },
        }

    RETURN_TYPES = ("GEN3C_TRAJECTORY", "STRING")
    RETURN_NAMES = ("trajectory", "trajectory_json")
    FUNCTION = "create"
    CATEGORY = "GEN3C/Camera"
    DESCRIPTION = "Generate camera trajectories for GEN3C with built-in presets (orbit, dolly, truck, spiral) or custom keyframes. Connect to Gen3CDiffusion for camera-controlled video generation."

    def create(
        self,
        frames: int,
        fps: int,
        width: int,
        height: int,
        fov_degrees: float,
        principal_x: float,
        principal_y: float,
        near_plane: float,
        far_plane: float,
        handedness: str,
        preset: str,
        orbit_radius: float,
        orbit_height: float,
        orbit_turns: float,
        dolly_start: float,
        dolly_end: float,
        truck_span: float,
        truck_depth: float,
        crane_start_height: float,
        crane_end_height: float,
        arc_degrees: float,
        tilt_degrees: float,
        spiral_start: float,
        spiral_end: float,
        boom_start_radius: float,
        boom_end_radius: float,
        boom_start_height: float,
        boom_end_height: float,
        hemisphere_elevation: float,
        keyframes_json: str = "[]",
    ):
        intrinsics = build_intrinsics(
            width=width,
            height=height,
            fov_degrees=fov_degrees,
            principal_x=principal_x,
            principal_y=principal_y,
            near=near_plane,
            far=far_plane,
        )

        frames_data = generate_trajectory(
            preset=preset,  # type: ignore[arg-type]
            total_frames=frames,
            intrinsics=intrinsics,
            handedness=handedness,  # type: ignore[arg-type]
            orbit_radius=orbit_radius,
            orbit_height=orbit_height,
            orbit_turns=orbit_turns,
            dolly_start=dolly_start,
            dolly_end=dolly_end,
            truck_span=truck_span,
            truck_depth=truck_depth,
            crane_start_height=crane_start_height,
            crane_end_height=crane_end_height,
            arc_degrees=arc_degrees,
            tilt_degrees=tilt_degrees,
            spiral_start=spiral_start,
            spiral_end=spiral_end,
            boom_start_radius=boom_start_radius,
            boom_end_radius=boom_end_radius,
            boom_start_height=boom_start_height,
            boom_end_height=boom_end_height,
            hemisphere_elevation=hemisphere_elevation,
            keyframes_json=keyframes_json,
        )

        payload = {
            "fps": fps,
            "frames": [frame.to_payload() for frame in frames_data],
            "handedness": handedness,
        }
        return (payload, json.dumps(payload, indent=2))


NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "Gen3C_CameraTrajectory": Gen3CCameraTrajectory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gen3C_CameraTrajectory": "GEN3C Camera Trajectory",
}
