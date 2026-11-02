# Room Visualiser Plan

## Goal

Build a Python-based room visualiser that takes:

- a room image
- a flooring product image

and produces:

- a realistic output image where the visible floor in the room is replaced with the supplied flooring product

The system should be modular, testable, and able to improve from an MVP into a production-ready pipeline.

## Recommended Stack

- Python 3.12
- PyTorch for model inference
- SAM2 for promptable image segmentation of the floor region
- Depth Anything V2 for monocular depth estimation and floor-plane reasoning
- OpenCV for geometry, homography, warping, masking, and compositing
- NumPy for array operations
- Pillow for image I/O and final exports
- scikit-image for seam handling and image utilities
- FastAPI for a service layer
- Gradio for an internal demo UI
- pytest for unit and regression testing

## Important Constraints

- `SAM2` officially requires modern PyTorch and recommends WSL on Windows. For this project, we should plan around `WSL2 + Ubuntu + CUDA` if an NVIDIA GPU is available.
- `Depth Anything V2` has a license split: the Small model is Apache-2.0, while the larger Base/Large variants are CC-BY-NC-4.0. If this project is for commercial use, that affects model choice.
- A flooring "product image" can mean very different things:
  - a flat texture swatch
  - a plank or tile packshot
  - a lifestyle photo

  The easiest path is a clean top-down or near-flat product texture image that can be tiled.

## Success Criteria

- Detect the visible floor region with minimal leakage into walls, skirting, rugs, or furniture.
- Estimate floor perspective well enough that the product texture looks anchored to the room.
- Preserve room lighting cues, shadows, and occlusions so the result feels believable.
- Produce repeatable outputs through a Python CLI and API.
- Save intermediate debug artifacts so failures are easy to diagnose.

## High-Level Pipeline

1. Load and validate the room image and flooring image.
2. Estimate a floor mask using `SAM2` plus geometry and depth cues.
3. Predict a depth map using `Depth Anything V2`.
4. Infer the visible floor plane and approximate camera perspective.
5. Convert the flooring product image into a tileable texture or repeatable pattern.
6. Warp the flooring texture into the room floor plane.
7. Reapply occlusions, edges, and lighting from the source room image.
8. Export the final composite plus debug outputs such as:
   - floor mask
   - depth map
   - plane overlay
   - warped texture
   - final composite

## Delivery Strategy

The best way to de-risk this project is not to start with full automation. We should first prove that the geometry and compositing look convincing with a manually supplied floor mask, then automate segmentation and plane estimation once the rendering path is solid.

## Phases

### Phase 0 - Discovery, setup, and feasibility

**Objective**

Set up the Python environment, verify model inference, and define the exact product input assumptions.

**Work**

- Create a Python project skeleton.
- Set up `venv` or `poetry` and lock dependency versions.
- Validate local inference for:
  - `SAM2`
  - `Depth Anything V2`
- Decide whether the project will run in:
  - local CLI mode only
  - local API mode
  - API plus demo UI
- Collect 10-20 representative room images and product images for development.
- Define whether commercial licensing matters.

**Deliverables**

- Working environment setup guide
- Sample inference notebook or script
- Small development dataset
- Written assumptions about product image format

**Exit Criteria**

- We can run both models locally on sample inputs.
- We have agreed on the flooring image format for the MVP.

### Phase 1 - Baseline MVP with manual floor mask

**Objective**

Prove that texture warping and compositing can look believable before solving automatic segmentation.

**Work**

- Build a script that accepts:
  - room image
  - flooring image
  - manually supplied floor polygon or mask
- Implement perspective warp using homography.
- Tile the flooring texture across the target floor plane.
- Blend warped flooring into the room image.
- Preserve edges with alpha feathering and mask cleanup.
- Export debug images for every stage.

**Deliverables**

- CLI command such as `python -m visualiser.cli --room ... --flooring ... --mask ...`
- First believable floor replacement outputs

**Exit Criteria**

- At least 5 test rooms produce visually plausible results using manual masks.

### Phase 2 - Automatic floor segmentation

**Objective**

Replace the manual floor mask with a reliable automatic mask.

**Work**

- Use `SAM2` image mode to generate candidate masks for the scene.
- Rank candidate masks using floor-like heuristics:
  - located in the lower part of the image
  - touches lower image boundary
  - broad horizontal extent
  - consistent with depth gradient
- Use `Depth Anything V2` to reject masks that do not behave like a receding floor plane.
- Add optional user guidance modes:
  - single click on the floor
  - rough box over floor
  - scribble hints
- Post-process masks with morphology and contour cleanup.
- Preserve non-floor occluders such as furniture legs and rugs where possible.

**Deliverables**

- `sam2_segmenter.py`
- Automatic mask generation pipeline
- Debug overlays showing chosen candidate vs rejected candidates

### Phase 2.7 - Recover nearby visible floor islands

**Objective**

Add a follow-up pass after the main floor segment is selected so disconnected-but-visible floor
patches can be recovered when they match the chosen floor.

**Work**

- Build a floor sample from the interior core of the selected floor segment.
- Compare nearby candidate regions against that sample using appearance and depth consistency.
- Merge accepted floor islands into the final visible floor mask while keeping the main selected
  floor as the geometry anchor for perspective.
- Save debug masks that show the anchor segment, recovered islands, and the floor sample core.

**Exit Criteria**

- Automatic floor mask is acceptable on most development images.
- User-guided fallback works when zero-shot segmentation fails.

### Phase 3 - Floor plane estimation and room geometry

**Objective**

Estimate how the floor recedes into the room so the flooring texture can be placed with correct perspective and scale.

**Work**

- Generate monocular depth from `Depth Anything V2`.
- Fit a dominant floor plane from masked floor pixels.
- Detect strong room lines with OpenCV for perspective cues:
  - wall-floor boundaries
  - skirting lines
  - tile or plank edges if visible
- Estimate vanishing direction and horizon approximation where possible.
- Compute a homography or pseudo-homography for texture placement.
- Add manual override controls for:
  - plank direction
  - texture scale
  - floor anchor position

**Deliverables**

- `plane_estimator.py`
- Geometry debug visualization
- Stable texture placement controls

**Exit Criteria**

- Texture orientation and perspective remain stable across multiple room layouts.

### Phase 4 - Flooring texture preparation

**Objective**

Convert the supplied product image into a repeatable, controllable floor material.

**Work**

- Support initial MVP input type:
  - flat swatch
  - plank sample
  - tile sample
- Normalize color space, crop borders, and remove background from the product image if needed.
- Build a repeatable tile texture.
- Support layout patterns:
  - straight planks
  - staggered planks
  - square tile
  - optional herringbone later
- Add real-world scale controls such as:
  - plank width
  - plank length
  - tile size

**Deliverables**

- `material_builder.py`
- Texture tiling pipeline
- Pattern presets for supported flooring types

**Exit Criteria**

- Flooring texture repeats cleanly without obvious seams in most outputs.

### Phase 5 - Realistic compositing and relighting

**Objective**

Make the synthetic floor look like it belongs in the original room.

**Work**

- Extract shading cues from the original floor image.
- Separate broad illumination from floor surface detail where possible.
- Modulate the warped flooring texture with room brightness and shadow patterns.
- Preserve soft shadows from furniture and room edges.
- Blend edges carefully near skirting, thresholds, and furniture contacts.
- Add optional reflection or gloss simulation only if the flooring type needs it.
- Add color matching controls so the output does not look too saturated or flat.

**Deliverables**

- `compositor.py`
- Improved realism on shadowed and partially occluded scenes

**Exit Criteria**

- Results look materially grounded rather than like a flat pasted texture.

### Phase 6 - Productization: CLI, API, and demo UI

**Objective**

Wrap the pipeline in tools that make it usable and testable.

**Work**

- Create a clean project structure with separate modules for:
  - segmentation
  - depth
  - geometry
  - materials
  - compositing
  - I/O
- Build a CLI entrypoint for batch and single-image runs.
- Build a FastAPI service endpoint such as `/visualise`.
- Add a Gradio demo for quick internal testing.
- Cache model instances so repeated runs are faster.
- Add config files for model paths, thresholds, and defaults.

**Deliverables**

- CLI tool
- FastAPI app
- Gradio demo
- Config-driven runtime

**Exit Criteria**

- A user can generate a result without touching internal code.

### Phase 7 - Evaluation, QA, and hardening

**Objective**

Make the system reliable enough for broader usage.

**Work**

- Build a regression set of room and flooring combinations.
- Save before/after results for visual comparison.
- Add tests for:
  - image loading
  - mask cleanup
  - homography calculations
  - texture tiling
  - output serialization
- Track common failure modes:
  - rugs mistaken for floor
  - glossy floors confusing depth
  - cluttered rooms
  - low-light scenes
  - extreme wide-angle images
- Add fallback paths:
  - user clicks to guide `SAM2`
  - manual mask upload
  - manual scale/orientation correction

**Deliverables**

- Regression suite
- Test suite
- Failure-mode checklist
- Tuning guide

**Exit Criteria**

- The pipeline is stable enough to demo confidently on unseen images.

## Proposed Repository Structure

```text
visualiser/
  plan.MD
  README.md
  pyproject.toml
  requirements.txt
  checkpoints/
  data/
    samples/
    regression/
  src/
    visualiser/
      __init__.py
      cli.py
      config.py
      io_utils.py
      pipeline.py
      segmenter/
        sam2_segmenter.py
        mask_postprocess.py
      depth/
        depth_anything.py
      geometry/
        plane_estimator.py
        homography.py
      materials/
        material_builder.py
        tiler.py
      compositing/
        compositor.py
        relighting.py
      api/
        app.py
      demo/
        gradio_app.py
  scripts/
    run_demo.py
    batch_visualise.py
  tests/
    test_masks.py
    test_homography.py
    test_materials.py
```

## MVP Scope Recommendation

For the first shippable version, keep the scope intentionally narrow:

- image in, image out only
- single room photo
- one dominant visible floor plane
- user can optionally guide the floor with one click or one box
- flooring input is a clean tileable texture or near-flat sample image
- no video support
- no full 3D reconstruction

This will let us solve the hardest realism problems without overbuilding.

## Technical Risks

- Floor segmentation can fail in cluttered rooms, especially with rugs or furniture covering most of the floor.
- Monocular depth is relative, not true scale, so manual scale controls will still be useful.
- A product image that is not tileable may require separate seam removal or material synthesis work.
- Lighting realism is often the difference between "works" and "looks fake"; this needs explicit time in the plan.
- Windows support for GPU-heavy research repos can be painful, which is why WSL2 is the safest default.

## Recommended First Build Order

1. Set up environment and verify `SAM2` and `Depth Anything V2` inference.
2. Build the manual-mask prototype.
3. Add floor texture tiling and perspective controls.
4. Add automatic floor segmentation.
5. Add depth-guided plane estimation.
6. Improve realism with relighting and occlusion handling.
7. Wrap everything in CLI, API, and demo tooling.

## Questions That Will Affect The Design

1. Is this intended for commercial use, or only internal/prototype use?
2. What kind of flooring input image will we receive: flat swatch, plank sample, tile sample, or product lifestyle photo?
3. Do you want the first version to be fully automatic, or is a light user-guided fallback acceptable?
4. Do we need to support only wood floors first, or also tile, laminate, stone, and patterned layouts?
5. Is the first target a local tool, a web API, or a simple demo UI?

## References

- SAM2 official repo: https://github.com/facebookresearch/sam2
- Depth Anything V2 official repo: https://github.com/DepthAnything/Depth-Anything-V2
