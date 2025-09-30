"""Trajectory preview and visualization utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


class TrajectoryPreview:
    """Generate visual previews of camera trajectories."""

    def __init__(self, image_size: Tuple[int, int] = (800, 600)):
        self.image_size = image_size

    def plot_trajectory_3d(
        self,
        trajectory: Dict[str, Any],
        output_path: Optional[Path] = None,
        show_frustums: bool = True,
        show_path: bool = True
    ) -> Image.Image:
        """Create a 3D plot of camera trajectory."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            matplotlib_available = True
        except ImportError:
            matplotlib_available = False

        if not matplotlib_available:
            # Fallback to simple 2D representation
            return self._plot_trajectory_2d(trajectory, output_path)

        frames = trajectory.get("frames", [])
        if not frames:
            return self._create_empty_plot("No trajectory data")

        # Extract camera positions and orientations
        positions = []
        forward_vectors = []

        for frame in frames:
            transform = frame.get("extrinsics", {}).get("camera_to_world")
            if transform:
                matrix = np.array(transform)
                pos = matrix[:3, 3]
                forward = -matrix[:3, 2]  # -Z axis in camera space
                positions.append(pos)
                forward_vectors.append(forward)

        if not positions:
            return self._create_empty_plot("No valid poses found")

        positions = np.array(positions)
        forward_vectors = np.array(forward_vectors)

        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot camera positions
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c='red', s=20, alpha=0.7, label='Camera positions')

        # Plot trajectory path
        if show_path and len(positions) > 1:
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                   'b-', alpha=0.5, linewidth=2, label='Camera path')

        # Plot camera frustums (simplified)
        if show_frustums:
            for i in range(0, len(positions), max(1, len(positions) // 10)):  # Sample every 10th camera
                pos = positions[i]
                forward = forward_vectors[i] * 0.5  # Scale frustum size
                ax.quiver(pos[0], pos[1], pos[2],
                         forward[0], forward[1], forward[2],
                         color='green', alpha=0.6, arrow_length_ratio=0.1)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Camera Trajectory Preview\n{len(positions)} frames')
        ax.legend()

        # Make axes equal
        max_range = np.array([positions[:,0].max()-positions[:,0].min(),
                            positions[:,1].max()-positions[:,1].min(),
                            positions[:,2].max()-positions[:,2].min()]).max() / 2.0
        mid_x = (positions[:,0].max()+positions[:,0].min()) * 0.5
        mid_y = (positions[:,1].max()+positions[:,1].min()) * 0.5
        mid_z = (positions[:,2].max()+positions[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Save or convert to PIL Image
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')

        # Convert to PIL Image
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = Image.frombarray(buf).convert('RGB')
        plt.close(fig)

        return image

    def _plot_trajectory_2d(
        self,
        trajectory: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Image.Image:
        """Fallback 2D trajectory plot when matplotlib is not available."""
        frames = trajectory.get("frames", [])
        if not frames:
            return self._create_empty_plot("No trajectory data")

        # Extract positions (top-down view)
        positions = []
        for frame in frames:
            transform = frame.get("extrinsics", {}).get("camera_to_world")
            if transform:
                matrix = np.array(transform)
                pos = matrix[:3, 3]
                positions.append([pos[0], pos[2]])  # X-Z plane (top-down)

        if not positions:
            return self._create_empty_plot("No valid poses found")

        positions = np.array(positions)

        # Create image
        img = Image.new('RGB', self.image_size, 'white')
        draw = ImageDraw.Draw(img)

        # Normalize positions to image space
        if len(positions) > 1:
            min_pos = positions.min(axis=0)
            max_pos = positions.max(axis=0)
            range_pos = max_pos - min_pos

            # Add padding
            padding = 50
            scale = min((self.image_size[0] - 2*padding) / max(range_pos[0], 1e-6),
                       (self.image_size[1] - 2*padding) / max(range_pos[1], 1e-6))

            # Convert to image coordinates
            img_positions = []
            for pos in positions:
                x = padding + (pos[0] - min_pos[0]) * scale
                y = padding + (pos[1] - min_pos[1]) * scale
                img_positions.append((x, y))

            # Draw trajectory path
            if len(img_positions) > 1:
                draw.line(img_positions, fill='blue', width=2)

            # Draw camera positions
            for i, pos in enumerate(img_positions):
                r = 3 if i % 5 == 0 else 2  # Larger dots every 5th frame
                color = 'red' if i == 0 else 'green' if i == len(img_positions)-1 else 'blue'
                draw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r], fill=color)

            # Add labels
            try:
                font = ImageFont.load_default()
            except:
                font = None

            draw.text((10, 10), f"Camera Trajectory (Top View)", fill='black', font=font)
            draw.text((10, 30), f"{len(positions)} frames", fill='black', font=font)
            draw.text((10, self.image_size[1]-40), "Red: Start, Green: End", fill='black', font=font)

        else:
            draw.text((10, 10), "Single camera position", fill='black')

        if output_path:
            img.save(output_path)

        return img

    def create_frustum_plot(
        self,
        trajectory: Dict[str, Any],
        frame_indices: Optional[List[int]] = None
    ) -> Image.Image:
        """Create a detailed frustum visualization."""
        frames = trajectory.get("frames", [])
        if not frames:
            return self._create_empty_plot("No trajectory data")

        if frame_indices is None:
            # Sample frames evenly
            frame_indices = list(range(0, len(frames), max(1, len(frames) // 8)))

        img = Image.new('RGB', self.image_size, 'white')
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.load_default()
        except:
            font = None

        # Draw title
        draw.text((10, 10), "Camera Frustums Preview", fill='black', font=font)

        # Get intrinsics for frustum shape
        first_frame = frames[0]
        intrinsics = first_frame.get("intrinsics", [[800, 0, 400], [0, 800, 300], [0, 0, 1]])
        fx, fy = intrinsics[0][0], intrinsics[1][1]
        cx, cy = intrinsics[0][2], intrinsics[1][2]
        width = first_frame.get("width", 800)
        height = first_frame.get("height", 600)

        # Calculate FOV
        fov_x = 2 * math.atan(width / (2 * fx)) * 180 / math.pi
        fov_y = 2 * math.atan(height / (2 * fy)) * 180 / math.pi

        draw.text((10, 40), f"FOV: {fov_x:.1f}° x {fov_y:.1f}°", fill='black', font=font)
        draw.text((10, 60), f"Resolution: {width}x{height}", fill='black', font=font)

        # Extract and normalize positions
        positions = []
        orientations = []

        for idx in frame_indices:
            if idx < len(frames):
                frame = frames[idx]
                transform = frame.get("extrinsics", {}).get("camera_to_world")
                if transform:
                    matrix = np.array(transform)
                    pos = matrix[:3, 3]
                    forward = -matrix[:3, 2]  # Camera forward direction
                    positions.append([pos[0], pos[2]])  # Top-down view
                    orientations.append(math.atan2(forward[2], forward[0]))

        if not positions:
            return self._create_empty_plot("No valid poses in selected frames")

        positions = np.array(positions)

        # Normalize to image space
        min_pos = positions.min(axis=0)
        max_pos = positions.max(axis=0)
        range_pos = max_pos - min_pos

        padding = 100
        scale = min((self.image_size[0] - 2*padding) / max(range_pos[0], 1e-6),
                   (self.image_size[1] - 2*padding) / max(range_pos[1], 1e-6))

        # Draw frustums
        for i, (pos, orient) in enumerate(zip(positions, orientations)):
            x = padding + (pos[0] - min_pos[0]) * scale
            y = padding + (pos[1] - min_pos[1]) * scale

            # Draw frustum outline
            frustum_size = 20
            half_fov = math.radians(fov_x / 2)

            # Frustum corners
            corners = []
            for angle_offset in [-half_fov, half_fov]:
                angle = orient + angle_offset
                end_x = x + frustum_size * math.cos(angle)
                end_y = y + frustum_size * math.sin(angle)
                corners.append((end_x, end_y))

            # Draw frustum lines
            color = 'red' if i == 0 else 'green' if i == len(positions)-1 else 'blue'
            draw.line([(x, y), corners[0]], fill=color, width=1)
            draw.line([(x, y), corners[1]], fill=color, width=1)
            draw.line([corners[0], corners[1]], fill=color, width=1)

            # Draw camera position
            draw.ellipse([x-3, y-3, x+3, y+3], fill=color)

            # Add frame number
            draw.text((x+5, y-10), str(frame_indices[i]), fill=color, font=font)

        return img

    def _create_empty_plot(self, message: str) -> Image.Image:
        """Create an empty plot with error message."""
        img = Image.new('RGB', self.image_size, 'white')
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.load_default()
        except:
            font = None

        # Center the message
        bbox = draw.textbbox((0, 0), message, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (self.image_size[0] - text_width) // 2
        y = (self.image_size[1] - text_height) // 2

        draw.text((x, y), message, fill='red', font=font)
        return img

    def generate_stats_image(self, trajectory: Dict[str, Any]) -> Image.Image:
        """Generate an image with trajectory statistics."""
        img = Image.new('RGB', self.image_size, 'white')
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.load_default()
        except:
            font = None

        frames = trajectory.get("frames", [])
        y_offset = 20
        line_height = 25

        # Title
        draw.text((20, y_offset), "Trajectory Statistics", fill='black', font=font)
        y_offset += line_height * 2

        # Basic info
        stats = [
            f"Total frames: {len(frames)}",
            f"FPS: {trajectory.get('fps', 'unknown')}",
            f"Handedness: {trajectory.get('handedness', 'unknown')}",
            f"Source: {trajectory.get('source', 'unknown')}",
        ]

        if frames:
            first_frame = frames[0]
            stats.extend([
                f"Resolution: {first_frame.get('width', '?')}x{first_frame.get('height', '?')}",
                f"Near plane: {first_frame.get('near', '?')}",
                f"Far plane: {first_frame.get('far', '?')}",
            ])

            # Intrinsics info
            intrinsics = first_frame.get("intrinsics", [[0, 0, 0], [0, 0, 0], [0, 0, 1]])
            fx, fy = intrinsics[0][0], intrinsics[1][1]
            cx, cy = intrinsics[0][2], intrinsics[1][2]

            stats.extend([
                f"Focal length: fx={fx:.1f}, fy={fy:.1f}",
                f"Principal point: ({cx:.1f}, {cy:.1f})",
            ])

            # Calculate trajectory extent
            positions = []
            for frame in frames:
                transform = frame.get("extrinsics", {}).get("camera_to_world")
                if transform:
                    matrix = np.array(transform)
                    pos = matrix[:3, 3]
                    positions.append(pos)

            if positions:
                positions = np.array(positions)
                extent = positions.max(axis=0) - positions.min(axis=0)
                max_extent = np.max(extent)
                stats.extend([
                    f"Trajectory extent: {max_extent:.2f} units",
                    f"X range: {extent[0]:.2f}",
                    f"Y range: {extent[1]:.2f}",
                    f"Z range: {extent[2]:.2f}",
                ])

        # Draw stats
        for stat in stats:
            draw.text((20, y_offset), stat, fill='black', font=font)
            y_offset += line_height

        return img


def plot_trajectory(
    trajectory: Dict[str, Any],
    output_dir: Optional[Path] = None,
    create_all: bool = True
) -> Dict[str, Image.Image]:
    """Generate trajectory visualizations."""
    preview = TrajectoryPreview()
    plots = {}

    if create_all or not output_dir:
        plots['3d_plot'] = preview.plot_trajectory_3d(trajectory)
        plots['frustums'] = preview.create_frustum_plot(trajectory)
        plots['stats'] = preview.generate_stats_image(trajectory)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, image in plots.items():
            image.save(output_dir / f"trajectory_{name}.png")

    return plots


__all__ = ["TrajectoryPreview", "plot_trajectory"]