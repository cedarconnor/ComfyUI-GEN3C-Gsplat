"""GEN3C Gaussian Splat custom node pack for ComfyUI."""

from __future__ import annotations

from .camera import nodes as camera_nodes
from .export_nodes import NODE_CLASS_MAPPINGS as export_node_map
from .export_nodes import NODE_DISPLAY_NAME_MAPPINGS as export_display_map
from .gen3c import NODE_CLASS_MAPPINGS as gen3c_nodes, NODE_DISPLAY_NAME_MAPPINGS as gen3c_display
from .trainers import nerfstudio as nerfstudio_nodes, gsplat as gsplat_nodes

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for mapping in (
    camera_nodes.NODE_CLASS_MAPPINGS,
    export_node_map,
    gen3c_nodes,
    nerfstudio_nodes.NODE_CLASS_MAPPINGS,
    gsplat_nodes.NODE_CLASS_MAPPINGS,
):
    NODE_CLASS_MAPPINGS.update(mapping)

for mapping in (
    camera_nodes.NODE_DISPLAY_NAME_MAPPINGS,
    export_display_map,
    gen3c_display,
    nerfstudio_nodes.NODE_DISPLAY_NAME_MAPPINGS,
    gsplat_nodes.NODE_DISPLAY_NAME_MAPPINGS,
):
    NODE_DISPLAY_NAME_MAPPINGS.update(mapping)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
