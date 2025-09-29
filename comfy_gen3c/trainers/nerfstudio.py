"""Nerfstudio Splatfacto trainer node for ComfyUI."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional


class SplatTrainerNerfstudio:
    """Wraps the Nerfstudio splatfacto trainer via CLI calls."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_dir": ("STRING", {"multiline": False}),
                "workspace_dir": ("STRING", {"default": "${output_dir}/nerfstudio_runs"}),
                "run_name": ("STRING", {"default": "gen3c_splat"}),
                "max_iterations": ("INT", {"default": 30_000, "min": 1, "max": 2_000_000}),
                "save_every": ("INT", {"default": 5_000, "min": 1, "max": 200_000}),
                "skip_training": ("BOOLEAN", {"default": False}),
                "export_after_train": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "additional_args": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("run_dir", "export_dir")
    FUNCTION = "train_and_export"
    CATEGORY = "GEN3C/Training"

    def _resolve_binary(self, name: str) -> Optional[str]:
        return shutil.which(name)

    def _run_command(self, command: List[str], cwd: Optional[Path] = None) -> None:
        subprocess.run(command, check=True, cwd=str(cwd) if cwd else None)

    def train_and_export(
        self,
        dataset_dir: str,
        workspace_dir: str,
        run_name: str,
        max_iterations: int,
        save_every: int,
        skip_training: bool,
        export_after_train: bool,
        additional_args: str = "",
    ):
        dataset_path = Path(dataset_dir).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory '{dataset_path}' does not exist.")

        workspace_path = Path(workspace_dir.replace("${output_dir}", str(Path.cwd() / "output"))).expanduser().resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)
        run_path = workspace_path / run_name
        run_path.mkdir(parents=True, exist_ok=True)

        ns_train = self._resolve_binary("ns-train")
        if ns_train is None:
            ns_train = [sys.executable, "-m", "nerfstudio.scripts.train"]
        else:
            ns_train = [ns_train]

        ns_export = self._resolve_binary("ns-export")
        if ns_export is None:
            ns_export = [sys.executable, "-m", "nerfstudio.scripts.export"]
        else:
            ns_export = [ns_export]

        if not skip_training:
            train_cmd: List[str] = ns_train + [
                "splatfacto",
                "--data", str(dataset_path),
                "--output-dir", str(run_path),
                "--max-num-iterations", str(max_iterations),
                "--steps-per-save", str(save_every),
            ]
            if additional_args.strip():
                train_cmd.extend(additional_args.strip().split())
            self._run_command(train_cmd)

        export_dir = run_path / "exports"
        if export_after_train:
            config_path = run_path / "config.yml"
            if not config_path.exists():
                # Fallback: locate newest config file within run directory
                configs = sorted(run_path.glob("**/config.yml"))
                if configs:
                    config_path = configs[-1]
                else:
                    raise FileNotFoundError(f"Unable to locate Nerfstudio config inside '{run_path}'.")

            export_dir.mkdir(parents=True, exist_ok=True)
            export_cmd = ns_export + [
                "gaussian-splat",
                "--load-config", str(config_path),
                "--output-dir", str(export_dir),
            ]
            self._run_command(export_cmd)

        return (str(run_path), str(export_dir) if export_after_train else "")


NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "SplatTrainer_Nerfstudio": SplatTrainerNerfstudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SplatTrainer_Nerfstudio": "Splat Trainer (Nerfstudio)",
}
