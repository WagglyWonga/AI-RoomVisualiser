# Room Visualiser

Python project for replacing the visible floor in a room photo with a supplied flooring product image.

## Phase 0 Status

Phase 0 is now scaffolded. The repository currently includes:

- a Python package layout under `src/visualiser`
- environment and model preflight checks
- smoke-test entrypoints for `SAM2` and `Depth Anything V2` readiness
- configuration via environment variables or `.env`
- placeholder directories for checkpoints, data, outputs, and third-party model repos

The current local Phase 0 path is validated on:

- Python `3.12`
- CPU PyTorch
- `SAM2` large checkpoint
- `Depth Anything V2` small (`vits`) checkpoint

This phase is about getting the project into a reliable starting state before we build the actual floor-replacement pipeline.

## Recommended Runtime

For model work, use:

- WSL2 with Ubuntu on Windows
- Python 3.12
- PyTorch installed separately for your CUDA or CPU target

Notes:

- The local Windows Python on this machine is `3.13`, but this project baseline is `3.12`.
- The `SAM2` maintainers explicitly recommend WSL on Windows.
- `Depth Anything V2` can be used through its official repo or through the Hugging Face `transformers` integration.

## Suggested Setup

1. Create and activate a Python 3.12 virtual environment.

PowerShell example:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python --version
```

If `venv` completes without `pip` in a restricted environment, use the global Python 3.12
interpreter for repo tasks and create the virtual environment from a normal local shell instead:

```powershell
py -3.12 scripts\preflight.py
py -3.12 scripts\smoke_test_models.py
```
2. Install PyTorch for your hardware using the official PyTorch selector.

For the current CPU-based Windows path, the working baseline is:

```powershell
python -m pip install torch torchvision matplotlib
```

3. Install this project:

```bash
pip install -e ".[dev]"
```

4. Clone the external model repos into `third_party/`:

```bash
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
git clone https://github.com/DepthAnything/Depth-Anything-V2.git third_party/Depth-Anything-V2
```

5. Install `SAM2` from the cloned repo:

```powershell
$env:SAM2_BUILD_CUDA="0"
pip install -e third_party/sam2
```

6. Download the checkpoints used by the current smoke test:

```powershell
curl.exe -L "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" -o "checkpoints\sam2.1_hiera_large.pt"
curl.exe -L "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true" -o "checkpoints\depth_anything_v2_vits.pth"
```

7. Copy `.env.example` to `.env` and fill in the model checkpoint paths.

The provided `.env.example` already points at:

- `sam2.1_hiera_large.pt`
- `depth_anything_v2_vits.pth`
- `third_party/Depth-Anything-V2/assets/examples/demo01.jpg`

8. Run the preflight check:

```bash
python scripts/preflight.py
```

9. Run the model readiness smoke test:

```bash
python scripts/smoke_test_models.py
```

For the repo-backed `Depth Anything V2` smoke test, no separate package install is required beyond
the cloned repo plus the PyTorch/OpenCV dependencies already installed in this project.

If you want to use the upstream `Depth-Anything-V2` demo scripts directly, install their extra requirements:

```bash
pip install -r third_party/Depth-Anything-V2/requirements.txt
```

If you prefer the Hugging Face depth backend instead of the repo backend:

```bash
pip install -e ".[hf]"
```

## Configuration

Configuration is driven by `.env` or environment variables. The important values are:

- `SAM2_REPO_DIR`
- `SAM2_CHECKPOINT`
- `SAM2_MODEL_CONFIG`
- `DEPTH_BACKEND`
- `DEPTH_ANYTHING_REPO_DIR`
- `DEPTH_ANYTHING_CHECKPOINT`
- `DEPTH_ANYTHING_ENCODER`

See `.env.example` for the full list.

## CLI Commands

Once the package is installed, the following commands are available:

```bash
visualiser env-check
visualiser smoke-test
visualiser manual-visualise
visualiser segment-floor
visualiser auto-visualise
```

Equivalent module usage:

```bash
python -m visualiser.cli env-check
python -m visualiser.cli smoke-test
python -m visualiser.cli manual-visualise
python -m visualiser.cli segment-floor
python -m visualiser.cli auto-visualise
```

## Phase 1 Manual MVP

Phase 1 now includes a manual-mask floor replacement pipeline. It takes:

- a room image
- a flooring texture image
- either a floor mask image or a floor polygon JSON file
- optionally a four-corner quad JSON file for perspective control

The command saves:

- `final.png`
- `run.json`
- debug images such as the warped texture, alpha mask, and quad overlay

Example using a mask plus an explicit quad:

```powershell
python scripts\manual_visualise.py `
  --room path\to\room.jpg `
  --flooring path\to\flooring.jpg `
  --mask path\to\floor_mask.png `
  --quad path\to\floor_quad.json `
  --output-dir outputs\my-first-run
```

Example using only a mask and letting the tool infer the floor quad:

```powershell
python scripts\manual_visualise.py `
  --room path\to\room.jpg `
  --flooring path\to\flooring.jpg `
  --mask path\to\floor_mask.png
```

Polygon and quad JSON files should be shaped like:

```json
[[70, 175], [470, 172], [535, 355], [10, 355]]
```

Useful tuning flags:

- `--texture-scale 1.2`
- `--texture-rotation 90`
- `--feather-radius 12`
- `--shading-strength 0.7`
- `--no-debug-images`

## Phase 2 Automatic Segmentation

Phase 2 adds automatic floor segmentation using `SAM2` candidate masks ranked with
`Depth Anything V2` depth cues and floor-shape heuristics.

Phase 2.7 extends that selection with a floor-island recovery pass. After the main floor
segment is chosen, the pipeline samples a core patch from that selected floor and looks for
nearby candidate regions that match the same floor appearance and row-aligned depth behavior.
This is meant to recover visible floor patches separated by chair legs, furniture gaps, or
similar thin occluders without simply flooding nearby non-floor regions.

Phase 2.8 adds an occluder-aware refinement pass after that. The segmenter looks for
room objects that are touching the chosen floor, temporarily inpaints those regions to
approximate the hidden floor, reruns floor expansion on the filled image, and then keeps
those touching objects in a preserve mask so they are placed back on top in the final
visualisation.

To inspect the automatically selected floor mask and candidate overlays:

```powershell
python scripts\segment_floor.py `
  --room inputs\test_room.jpg `
  --output-dir outputs\segment-demo `
  --device cpu
```

To run the full automatic visualiser:

```powershell
python scripts\auto_visualise.py `
  --room inputs\test_room.jpg `
  --flooring inputs\product.jpg `
  --output-dir outputs\auto-demo `
  --device cpu
```

To run the visualiser across every image in `inputs\rooms` and every image in `inputs\floors`:

```powershell
python scripts\batch_auto_visualise.py `
  --rooms-dir inputs\rooms `
  --floors-dir inputs\floors `
  --output-dir outputs `
  --device cpu
```

This creates one folder per combination using the naming pattern `room-name-floor-name`,
for example `outputs\room10-product`.

The batch run also writes:

- `outputs\batch-auto-visualise-summary.json`
- `outputs\batch-auto-visualise-summary.csv`
- `outputs\batch-auto-visualise-gallery.html`

The HTML gallery gives you a quick visual review of every generated `final.png` and links back
to each combination folder, its `run.json`, and the original room/floor input images.

Key Phase 2 outputs:

- `selected_mask.png`
- `anchor_mask.png`
- `recovered_mask.png`
- `sample_core_mask.png`
- `occluder_mask.png`
- `phase28_recovered_mask.png`
- `phase28_inpainted_room.png` when the second pass runs
- `segment_floor.json` or `run.json`
- candidate overlays such as `candidate_01.png`
- `depth_preview.png`
- `final.png` for the full automatic visualiser run

## Windows Bootstrap Helper

You can also create the local Python 3.12 environment with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_windows.ps1
```

The bootstrap script verifies that `pip` is actually available inside the new environment.

## External References

- SAM2 official repo: https://github.com/facebookresearch/sam2
- Depth Anything V2 official repo: https://github.com/DepthAnything/Depth-Anything-V2
