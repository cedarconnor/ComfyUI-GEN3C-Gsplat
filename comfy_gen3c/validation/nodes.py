"""ComfyUI nodes for dataset validation and quality control."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import json

from .dataset_validator import DatasetValidator, ValidationResult
from .trajectory_preview import TrajectoryPreview, plot_trajectory
from .quality_filters import QualityFilter


class Gen3CDatasetValidator:
    """Validate GEN3C datasets for training quality."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_path": ("STRING", {"default": ""}),
                "min_frames": ("INT", {"default": 3, "min": 1, "max": 10}),
                "max_frames": ("INT", {"default": 1000, "min": 10, "max": 10000}),
                "generate_report": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("quality_score", "validation_status", "issues", "stats")
    FUNCTION = "validate_dataset"
    CATEGORY = "GEN3C/Validation"

    def validate_dataset(
        self,
        dataset_path: str,
        min_frames: int,
        max_frames: int,
        generate_report: bool
    ) -> Tuple[float, str, str, str]:

        if not dataset_path or not Path(dataset_path).exists():
            return (0.0, "FAILED", "Dataset path does not exist", "{}")

        try:
            validator = DatasetValidator(min_frames=min_frames, max_frames=max_frames)
            result = validator.validate_dataset(Path(dataset_path))

            status = "PASSED" if result.is_valid else "FAILED"
            if not result.is_valid and result.score > 0.5:
                status = "WARNING"

            issues_text = "\n".join(result.issues) if result.issues else "No issues found"
            stats_text = json.dumps(result.stats, indent=2)

            # Generate detailed report if requested
            if generate_report:
                report_path = Path(dataset_path) / "validation_report.txt"
                self._generate_validation_report(result, report_path)

            return (result.score, status, issues_text, stats_text)

        except Exception as e:
            return (0.0, "ERROR", f"Validation failed: {str(e)}", "{}")

    def _generate_validation_report(self, result: ValidationResult, output_path: Path):
        """Generate detailed validation report."""
        with open(output_path, 'w') as f:
            f.write("GEN3C Dataset Validation Report\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"Overall Status: {'PASSED' if result.is_valid else 'FAILED'}\n")
            f.write(f"Quality Score: {result.score:.3f}\n\n")

            if result.issues:
                f.write("Issues Found:\n")
                for issue in result.issues:
                    f.write(f"  - {issue}\n")
                f.write("\n")

            if result.warnings:
                f.write("Warnings:\n")
                for warning in result.warnings:
                    f.write(f"  - {warning}\n")
                f.write("\n")

            f.write("Statistics:\n")
            f.write(json.dumps(result.stats, indent=2))


class Gen3CTrajectoryPreview:
    """Generate trajectory preview visualizations."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trajectory": ("GEN3C_TRAJECTORY", {}),
                "plot_type": (["3d", "frustums", "stats", "all"], {"default": "all"}),
                "output_dir": ("STRING", {"default": "${output_dir}/trajectory_preview"}),
            },
            "optional": {
                "show_frustums": ("BOOLEAN", {"default": True}),
                "show_path": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview_image", "output_path")
    FUNCTION = "generate_preview"
    CATEGORY = "GEN3C/Validation"

    def generate_preview(
        self,
        trajectory: Dict[str, Any],
        plot_type: str,
        output_dir: str,
        show_frustums: bool = True,
        show_path: bool = True
    ) -> Tuple[torch.Tensor, str]:

        output_path = Path(output_dir.replace("${output_dir}", str(Path.cwd() / "output")))
        output_path = output_path.expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            preview = TrajectoryPreview()

            if plot_type == "3d":
                image = preview.plot_trajectory_3d(trajectory, show_frustums=show_frustums, show_path=show_path)
                image.save(output_path / "trajectory_3d.png")
            elif plot_type == "frustums":
                image = preview.create_frustum_plot(trajectory)
                image.save(output_path / "trajectory_frustums.png")
            elif plot_type == "stats":
                image = preview.generate_stats_image(trajectory)
                image.save(output_path / "trajectory_stats.png")
            else:  # "all"
                plots = plot_trajectory(trajectory, output_path, create_all=True)
                image = plots.get('3d_plot', plots.get('stats'))

            # Convert PIL Image to ComfyUI tensor format
            if image:
                import numpy as np
                image_array = np.array(image)
                # Convert to (1, H, W, C) format for ComfyUI
                image_tensor = torch.from_numpy(image_array).unsqueeze(0).float() / 255.0
            else:
                # Create empty tensor if no image generated
                image_tensor = torch.zeros(1, 400, 600, 3)

            return (image_tensor, str(output_path))

        except Exception as e:
            # Return error image
            error_image = torch.zeros(1, 400, 600, 3)
            return (error_image, f"Error: {str(e)}")


class Gen3CQualityFilter:
    """Filter low-quality frames from datasets."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_path": ("STRING", {"default": ""}),
                "trajectory": ("GEN3C_TRAJECTORY", {}),
                "quality_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "min_blur_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "min_brightness": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_brightness": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("GEN3C_TRAJECTORY", "STRING", "INT", "INT")
    RETURN_NAMES = ("filtered_trajectory", "filter_report", "frames_kept", "frames_removed")
    FUNCTION = "filter_quality"
    CATEGORY = "GEN3C/Validation"

    def filter_quality(
        self,
        dataset_path: str,
        trajectory: Dict[str, Any],
        quality_threshold: float,
        min_blur_threshold: float,
        min_brightness: float,
        max_brightness: float
    ) -> Tuple[Dict[str, Any], str, int, int]:

        if not dataset_path or not Path(dataset_path).exists():
            return (trajectory, "Dataset path does not exist", 0, 0)

        try:
            quality_filter = QualityFilter(
                min_blur_threshold=min_blur_threshold,
                min_brightness_threshold=min_brightness,
                max_brightness_threshold=max_brightness,
                min_overall_threshold=quality_threshold
            )

            filtered_trajectory, removed_indices = quality_filter.filter_low_quality_frames(
                dataset_path, trajectory, quality_threshold
            )

            original_count = len(trajectory.get("frames", []))
            filtered_count = len(filtered_trajectory.get("frames", []))
            removed_count = len(removed_indices)

            # Generate report
            report_lines = [
                f"Quality Filtering Report",
                f"Original frames: {original_count}",
                f"Frames kept: {filtered_count}",
                f"Frames removed: {removed_count}",
                f"Quality threshold: {quality_threshold:.2f}",
                f"Removal rate: {removed_count/max(original_count, 1)*100:.1f}%"
            ]

            if removed_indices:
                report_lines.append(f"Removed frame indices: {removed_indices[:10]}{'...' if len(removed_indices) > 10 else ''}")

            report = "\n".join(report_lines)

            return (filtered_trajectory, report, filtered_count, removed_count)

        except Exception as e:
            return (trajectory, f"Quality filtering failed: {str(e)}", 0, 0)


class Gen3CTrajectoryQualityAnalysis:
    """Analyze trajectory quality metrics."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trajectory": ("GEN3C_TRAJECTORY", {}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("overall_score", "smoothness", "coverage", "baseline_quality", "rotation_diversity", "analysis_report")
    FUNCTION = "analyze_trajectory"
    CATEGORY = "GEN3C/Validation"

    def analyze_trajectory(
        self,
        trajectory: Dict[str, Any]
    ) -> Tuple[float, float, float, float, float, str]:

        try:
            quality_filter = QualityFilter()
            trajectory_quality = quality_filter.assess_trajectory_quality(trajectory)

            # Generate detailed report
            report_lines = [
                "Trajectory Quality Analysis",
                "=" * 30,
                f"Overall Score: {trajectory_quality.overall_score:.3f}",
                f"Smoothness: {trajectory_quality.smoothness_score:.3f}",
                f"Coverage: {trajectory_quality.coverage_score:.3f}",
                f"Baseline Quality: {trajectory_quality.baseline_score:.3f}",
                f"Rotation Diversity: {trajectory_quality.rotation_diversity:.3f}",
                f"Speed Consistency: {trajectory_quality.speed_consistency:.3f}",
            ]

            if trajectory_quality.issues:
                report_lines.append("\nIssues Found:")
                for issue in trajectory_quality.issues:
                    report_lines.append(f"  - {issue}")

            # Add recommendations
            report_lines.append("\nRecommendations:")
            if trajectory_quality.smoothness_score < 0.5:
                report_lines.append("  - Consider smoothing the camera path")
            if trajectory_quality.coverage_score < 0.5:
                report_lines.append("  - Increase camera movement range for better coverage")
            if trajectory_quality.rotation_diversity < 0.5:
                report_lines.append("  - Add more diverse viewing angles")

            report = "\n".join(report_lines)

            return (
                trajectory_quality.overall_score,
                trajectory_quality.smoothness_score,
                trajectory_quality.coverage_score,
                trajectory_quality.baseline_score,
                trajectory_quality.rotation_diversity,
                report
            )

        except Exception as e:
            return (0.0, 0.0, 0.0, 0.0, 0.0, f"Analysis failed: {str(e)}")


class Gen3CQuality:
    """Unified quality control node for validation, filtering, analysis, and preview."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["validate", "filter", "analyze", "preview", "all"], {
                    "default": "all",
                    "tooltip": "Quality operation: validate dataset, filter frames, analyze trajectory, preview, or run all checks"
                }),
                "trajectory": ("GEN3C_TRAJECTORY", {"tooltip": "Camera trajectory for quality analysis"}),
            },
            "optional": {
                "dataset_path": ("STRING", {"default": "", "tooltip": "Dataset directory path (required for validate/filter modes)"}),

                # Validation settings
                "min_frames": ("INT", {"default": 3, "min": 1, "max": 10, "tooltip": "Minimum frames required"}),
                "max_frames": ("INT", {"default": 1000, "min": 10, "max": 10000, "tooltip": "Maximum frames allowed"}),

                # Filter settings
                "quality_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Overall quality threshold"}),
                "min_blur_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Minimum blur threshold"}),
                "min_brightness": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Minimum brightness"}),
                "max_brightness": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Maximum brightness"}),

                # Preview settings
                "plot_type": (["3d", "frustums", "stats", "all"], {"default": "3d", "tooltip": "Visualization type"}),
                "output_dir": ("STRING", {"default": "${output_dir}/quality", "tooltip": "Output directory for reports/previews"}),
            },
        }

    RETURN_TYPES = ("GEN3C_TRAJECTORY", "FLOAT", "STRING", "IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("filtered_trajectory", "quality_score", "report", "preview_image", "status", "frames_kept", "frames_removed")
    FUNCTION = "run_quality_check"
    CATEGORY = "GEN3C/Quality"
    DESCRIPTION = "Unified quality control: validate datasets, filter low-quality frames, analyze trajectory metrics, and generate previews. Run individual checks or comprehensive quality pipeline."

    def run_quality_check(
        self,
        mode: str,
        trajectory: Dict[str, Any],
        dataset_path: str = "",
        min_frames: int = 3,
        max_frames: int = 1000,
        quality_threshold: float = 0.4,
        min_blur_threshold: float = 0.3,
        min_brightness: float = 0.15,
        max_brightness: float = 0.85,
        plot_type: str = "3d",
        output_dir: str = "${output_dir}/quality",
    ) -> Tuple[Dict[str, Any], float, str, torch.Tensor, str, int, int]:

        filtered_trajectory = trajectory
        quality_score = 0.0
        report_lines = []
        preview_image = torch.zeros(1, 400, 600, 3)
        status = "Success"
        frames_kept = len(trajectory.get("frames", []))
        frames_removed = 0

        output_path = Path(output_dir.replace("${output_dir}", str(Path.cwd() / "output")))
        output_path = output_path.expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Mode: Validate Dataset
            if mode in ["validate", "all"]:
                if dataset_path and Path(dataset_path).exists():
                    validator = DatasetValidator(min_frames=min_frames, max_frames=max_frames)
                    result = validator.validate_dataset(Path(dataset_path))
                    quality_score = result.score

                    report_lines.append("=== DATASET VALIDATION ===")
                    report_lines.append(f"Status: {'PASSED' if result.is_valid else 'FAILED'}")
                    report_lines.append(f"Quality Score: {result.score:.3f}")
                    if result.issues:
                        report_lines.append("Issues:")
                        report_lines.extend([f"  - {issue}" for issue in result.issues])
                    report_lines.append("")

            # Mode: Trajectory Quality Analysis
            if mode in ["analyze", "all"]:
                quality_filter = QualityFilter()
                traj_quality = quality_filter.assess_trajectory_quality(trajectory)

                report_lines.append("=== TRAJECTORY ANALYSIS ===")
                report_lines.append(f"Overall Score: {traj_quality.overall_score:.3f}")
                report_lines.append(f"Smoothness: {traj_quality.smoothness_score:.3f}")
                report_lines.append(f"Coverage: {traj_quality.coverage_score:.3f}")
                report_lines.append(f"Baseline Quality: {traj_quality.baseline_score:.3f}")
                report_lines.append(f"Rotation Diversity: {traj_quality.rotation_diversity:.3f}")

                if traj_quality.issues:
                    report_lines.append("Issues:")
                    report_lines.extend([f"  - {issue}" for issue in traj_quality.issues])

                report_lines.append("")
                quality_score = max(quality_score, traj_quality.overall_score)

            # Mode: Filter Low-Quality Frames
            if mode in ["filter", "all"]:
                if dataset_path and Path(dataset_path).exists():
                    quality_filter = QualityFilter(
                        min_blur_threshold=min_blur_threshold,
                        min_brightness_threshold=min_brightness,
                        max_brightness_threshold=max_brightness,
                        min_overall_threshold=quality_threshold
                    )

                    filtered_trajectory, removed_indices = quality_filter.filter_low_quality_frames(
                        dataset_path, trajectory, quality_threshold
                    )

                    original_count = len(trajectory.get("frames", []))
                    frames_kept = len(filtered_trajectory.get("frames", []))
                    frames_removed = len(removed_indices)

                    report_lines.append("=== QUALITY FILTERING ===")
                    report_lines.append(f"Original frames: {original_count}")
                    report_lines.append(f"Frames kept: {frames_kept}")
                    report_lines.append(f"Frames removed: {frames_removed}")
                    report_lines.append(f"Removal rate: {frames_removed/max(original_count, 1)*100:.1f}%")
                    report_lines.append("")

            # Mode: Generate Preview
            if mode in ["preview", "all"]:
                try:
                    preview = TrajectoryPreview()
                    if plot_type == "3d":
                        image = preview.plot_trajectory_3d(trajectory, show_frustums=True, show_path=True)
                    elif plot_type == "frustums":
                        image = preview.create_frustum_plot(trajectory)
                    elif plot_type == "stats":
                        image = preview.generate_stats_image(trajectory)
                    else:  # "all"
                        plots = plot_trajectory(trajectory, output_path, create_all=True)
                        image = plots.get('3d_plot', plots.get('stats'))

                    if image:
                        import numpy as np
                        image_array = np.array(image)
                        preview_image = torch.from_numpy(image_array).unsqueeze(0).float() / 255.0
                        image.save(output_path / f"trajectory_{plot_type}.png")
                        report_lines.append(f"Preview saved to: {output_path / f'trajectory_{plot_type}.png'}")
                except Exception as e:
                    report_lines.append(f"Preview generation failed: {str(e)}")

            # Save comprehensive report
            if mode == "all":
                report_path = output_path / "quality_report.txt"
                with open(report_path, 'w') as f:
                    f.write("\n".join(report_lines))
                report_lines.append(f"\nFull report saved to: {report_path}")

            report = "\n".join(report_lines) if report_lines else "No operations performed"

        except Exception as e:
            status = f"Error: {str(e)}"
            report = f"Quality check failed: {str(e)}"

        return (filtered_trajectory, quality_score, report, preview_image, status, frames_kept, frames_removed)


NODE_CLASS_MAPPINGS = {
    "Gen3C_Quality": Gen3CQuality,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gen3C_Quality": "GEN3C Quality",
}