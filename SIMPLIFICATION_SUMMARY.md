# Workflow Simplification Summary

## Overview
The ComfyUI-GEN3C-Gsplat node pack has been simplified from **13 nodes to 7 core nodes**, reducing complexity while maintaining full functionality.

## Node Changes

### Before (13 nodes)
**Camera:**
- Gen3C_CameraTrajectory (24+ required parameters)

**Export:**
- Cosmos_Gen3C_InferExport
- Cosmos_Gen3C_DirectExport
- Gen3C_VideoToDataset

**Pose Recovery:**
- Gen3C_PoseRecovery (unified)
- Gen3C_PoseDepth_FromVideo (legacy)
- Gen3C_PoseDepth_FromImages (legacy)

**Validation:**
- Gen3C_DatasetValidator
- Gen3C_TrajectoryPreview
- Gen3C_QualityFilter
- Gen3C_TrajectoryQualityAnalysis

**Training:**
- SplatTrainer_gsplat
- SplatTrainer_Nerfstudio

---

### After (7 nodes)
**Camera:**
- **Gen3C_Camera** - Simplified with 7 required params, advanced options in optional inputs

**Export:**
- **Gen3C_Export** - Unified node that auto-detects input type (Cosmos inference, video file, or explicit trajectory)

**Pose Recovery:**
- **Gen3C_PoseRecovery** - Unified node for both video files and image sequences

**Quality:**
- **Gen3C_Quality** - All-in-one quality control with modes: validate, filter, analyze, preview, or all

**Training:**
- SplatTrainer_gsplat (unchanged)
- SplatTrainer_Nerfstudio (unchanged)

---

## Key Improvements

### 1. Camera Node Simplification
**Before:** 24+ required parameters visible at once (overwhelming)
**After:** 7 required parameters, 10 optional parameters

Required parameters reduced from:
- frames, fps, width, height, fov, principal_x, principal_y, near, far, handedness, preset
- orbit_radius, orbit_height, orbit_turns, dolly_start, dolly_end, truck_span, truck_depth
- crane_start/end_height, arc_degrees, tilt_degrees, spiral_start/end
- boom_start/end_radius, boom_start/end_height, hemisphere_elevation

To just:
- preset, frames, fps, width, height, radius, height_offset

### 2. Export Node Unification
**Before:** 3 separate nodes for different scenarios
- Cosmos_Gen3C_InferExport (explicit trajectory)
- Cosmos_Gen3C_DirectExport (extract from latents)
- Gen3C_VideoToDataset (video pose recovery)

**After:** 1 smart node
- Gen3C_Export auto-detects whether you're providing:
  - Video file path → runs pose recovery automatically
  - Latents → extracts trajectory automatically
  - Explicit trajectory → uses it directly

### 3. Quality Control Consolidation
**Before:** 4 separate nodes for quality operations
**After:** 1 unified node with mode selector
- Mode options: validate, filter, analyze, preview, or all
- Run comprehensive quality pipeline with single node

### 4. Pose Recovery Cleanup
**Before:** 3 nodes (1 unified + 2 legacy for backward compatibility)
**After:** 1 unified node
- Handles both video files and image sequences
- Single source_type parameter to switch between workflows

---

## Migration Guide

### Camera Trajectories
**Old workflow:**
```
Gen3C_CameraTrajectory
├─ Set 24+ parameters
└─ Connect trajectory output
```

**New workflow:**
```
Gen3C_Camera
├─ Choose preset
├─ Set radius & height_offset
└─ (Optional) Adjust advanced parameters
```

### Dataset Export
**Old workflow (3 different paths):**
```
# Path 1: Inference
Gen3CDiffusion → Cosmos_Gen3C_InferExport

# Path 2: Direct
Gen3CDiffusion → Cosmos_Gen3C_DirectExport

# Path 3: Video
Video → Gen3C_VideoToDataset
```

**New workflow (unified):**
```
# All paths:
Any Input → Gen3C_Export
```

The export node auto-detects:
- If video_path provided → runs pose recovery
- If latents provided → extracts trajectory
- If trajectory provided → uses it directly

### Quality Control
**Old workflow:**
```
Dataset → Gen3C_DatasetValidator → validation
         → Gen3C_QualityFilter → filtered frames
         → Gen3C_TrajectoryQualityAnalysis → metrics
         → Gen3C_TrajectoryPreview → visualization
```

**New workflow:**
```
Dataset → Gen3C_Quality (mode="all") → comprehensive report
```

Or use specific modes: validate, filter, analyze, preview

---

## Simplified Workflow Examples

### Complete GEN3C → Splat Pipeline (Memory-based)
```
Gen3C_Camera (orbit preset)
    ↓ trajectory
Gen3CDiffusion
    ↓ images, latents
Gen3C_Export (write_to_disk=false)
    ↓ dataset
SplatTrainer_gsplat
    ↓ ply_path
Output
```

### Video → Splat Pipeline
```
Gen3C_Export
├─ video_path: "input.mp4"
├─ write_to_disk: false
└─ (auto runs pose recovery)
    ↓ dataset
SplatTrainer_gsplat
    ↓ ply_path
Output
```

### Quality-Controlled Pipeline
```
Gen3C_Camera → Gen3CDiffusion → Gen3C_Export (write_to_disk=true)
                                     ↓ dataset_dir, trajectory
                               Gen3C_Quality (mode="all")
                                     ↓ filtered_trajectory, report
                               Gen3C_Export (write_to_disk=false)
                                     ↓ dataset
                               SplatTrainer_gsplat
                                     ↓ ply_path
                                   Output
```

---

## Benefits

1. **Reduced Cognitive Load:** 7 nodes instead of 13, fewer decisions to make
2. **Cleaner UI:** Camera node shows 7 params instead of 24 by default
3. **Auto-Detection:** Export node handles all scenarios intelligently
4. **Unified Quality:** One node for all quality operations
5. **Easier Learning:** Simpler node graph for beginners
6. **Full Flexibility:** Advanced users can still access all parameters via optional inputs

---

## Backward Compatibility

The original complex nodes are still available in the codebase but not registered. If you need them for existing workflows, they can be re-enabled by updating the NODE_CLASS_MAPPINGS in the respective files.

---

## Summary Statistics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Total Nodes | 13 | 7 | -46% |
| Camera Required Params | 24+ | 7 | -71% |
| Export Nodes | 3 | 1 | -67% |
| Quality Nodes | 4 | 1 | -75% |
| Pose Recovery Nodes | 3 | 1 | -67% |

**Overall:** 46% fewer nodes, 71% fewer camera parameters, unified workflows.
