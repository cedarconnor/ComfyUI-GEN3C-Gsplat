"""GEN3C (Cosmos) node exports."""

from .loader import NODE_CLASS_MAPPINGS as loader_nodes, NODE_DISPLAY_NAME_MAPPINGS as loader_display
from .diffusion import NODE_CLASS_MAPPINGS as diffusion_nodes, NODE_DISPLAY_NAME_MAPPINGS as diffusion_display

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for mapping in (loader_nodes, diffusion_nodes):
    NODE_CLASS_MAPPINGS.update(mapping)

for mapping in (loader_display, diffusion_display):
    NODE_DISPLAY_NAME_MAPPINGS.update(mapping)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
