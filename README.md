# MOPA Heightmap Studio

Turn a photo into a **LightBurn 3D Sliced**–ready heightmap, optimized for **JPT MOPA fiber galvo** engraving on metal.

Built as a thin app on top of [ZoeDepth](#upstream-notice). The model is not modified.

> **Status:** v9.0-spa — Angular + FastAPI SPA. The legacy Gradio UI has been removed. See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) and [`docs/SONNET_UI_MIGRATION_BRIEF.md`](docs/SONNET_UI_MIGRATION_BRIEF.md) for the migration history.

---

## Quick start

### 1. Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[ui]"
```

### 1b. Install (NVIDIA GPU, faster)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[ui]"
```

### 2. Launch the API server

```powershell
.\.venv\Scripts\uvicorn.exe apps.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Serve the Angular SPA

```powershell
# Production build (one-time):
cd apps\web
npm ci
npx ng build --configuration=production

# Serve the dist folder (or use any static file server):
npx serve dist\web
```

Open `http://localhost:3000` (or the port `serve` reports). The Wizard and Studio routes are available at `/wizard` and `/studio`.

### 4. Or use the CLI

```powershell
mopa-heightmap photo.jpg --profile mopa_60w_brass --export-preview
mopa-heightmap --make-ramp ramp.png
```

Outputs land in `outputs/` next to the input by default. Each export writes:

| File | Purpose |
|---|---|
| `<name>_lightburn.png` | 8-bit grayscale to drop into a LightBurn **3D Sliced** layer. |
| `<name>_master16.png` | 16-bit master for re-tuning later without re-running depth. |
| `<name>_preview.png` | Shaded relief preview for sanity-check. |
| `<name>_settings.json` | Sidecar — every parameter, model, profile, hash, timestamp. Reproducible re-runs. |

---

## Sharing material profiles

Material profiles are plain YAML files. To install one a friend sent you:

```powershell
# Drop it here — picked up automatically on the next launch.
mkdir $HOME\.mopa-heightmap\profiles
copy custom_brass.yaml $HOME\.mopa-heightmap\profiles\
```

User-scope profiles take precedence over the shipped ones with the same name. The Studio UI lists everything from both locations in one dropdown.

See [`profiles/mopa_60w_brass.yaml`](profiles/mopa_60w_brass.yaml) for the schema.

---

## What this app produces

The app emits one or more **grayscale heightmaps** plus a sidecar JSON. You import the heightmap into LightBurn, set the layer to **3D Sliced**, and the included material profile's `lightburn_starting_point` block tells you the speed / power / frequency / pulse-width / line-interval to start with.

**v1 does not** generate `.lbrn2` project files yet. That lands in Phase 4 — see [`docs/PLAN.md`](docs/PLAN.md) §22. v1.5 (Phase 3) will ship `.clb` Cut Library exports so importing into LightBurn populates the Material Library directly.

---

## Calibration

Engrave the calibration ramp (`mopa-heightmap --make-ramp ramp.png`) on the actual material at the speed/power you plan to use. Eleven discrete gray steps will produce eleven measurable depths. Phase 3 wires those measurements back into the profile as a gray→depth LUT so future exports hit a target depth budget directly.

---

## Run the test suite

```powershell
pip install -e ".[dev]"
pytest -q
```

---

## Architecture in 30 seconds

```
ui/mopa_studio.py  ─┐
apps/zoe2lightburn ─┴─► HeightmapService ──► ZoeDepth (untouched)
                          │
                          └─► heightmap.process → exporter → atomic PNG + sidecar JSON
```

- `zoedepth/laser/service.py` — single orchestrator. Owns the loaded model + a small depth cache. Same code path for CLI and UI.
- `zoedepth/laser/heightmap.py` — Stage C post-processing (percentile clip → polarity → tone curve → smoothing → sharpen → dither).
- `zoedepth/laser/exporter.py` — atomic writes (`*.tmp` → `os.replace`), three naming modes (`overwrite` | `timestamp` | `counter`), sidecar JSON.
- `zoedepth/laser/profiles.py` — YAML profile loader + schema validator. User-scope dir + repo dir.
- `zoedepth/laser/settings.py` — `~/.mopa-heightmap/settings.json` (output naming, preview cap, default model, device).

Read [`docs/PLAN.md`](docs/PLAN.md) for the full multi-pass / multi-layer / LightBurn-native export roadmap.

---

## Upstream notice

This project is a fork of [ZoeDepth](https://github.com/isl-org/ZoeDepth) by Intel ISL. **The original ZoeDepth project is no longer maintained by Intel.** All upstream ZoeDepth research code, training scripts, and evaluation tooling remain in this repository for reference but are not the focus of this fork.

The original README content begins below.

---

# **ZoeDepth: Combining relative and metric depth** (Original implementation)  <!-- omit in toc -->
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isl-org/ZoeDepth)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/shariqfarooq/ZoeDepth)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) ![PyTorch](https://img.shields.io/badge/PyTorch_v1.10.1-EE4C2C?&logo=pytorch&logoColor=white) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zoedepth-zero-shot-transfer-by-combining/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=zoedepth-zero-shot-transfer-by-combining)

>#### [ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth](https://arxiv.org/abs/2302.12288)
> ##### [Shariq Farooq Bhat](https://shariqfarooq123.github.io), [Reiner Birkl](https://www.researchgate.net/profile/Reiner-Birkl), [Diana Wofk](https://dwofk.github.io/), [Peter Wonka](http://peterwonka.net/), [Matthias Müller](https://matthias.pw/)

[[Paper]](https://arxiv.org/abs/2302.12288)

![teaser](assets/zoedepth-teaser.png)

## **Table of Contents** <!-- omit in toc -->
- [**Usage**](#usage)
  - [Using torch hub](#using-torch-hub)
  - [Using local copy](#using-local-copy)
    - [Using local torch hub](#using-local-torch-hub)
    - [or load the models manually](#or-load-the-models-manually)
  - [Using ZoeD models to predict depth](#using-zoed-models-to-predict-depth)
- [**Environment setup**](#environment-setup)
- [**Sanity checks** (Recommended)](#sanity-checks-recommended)
- [Model files](#model-files)
- [**Evaluation**](#evaluation)
  - [Evaluating offical models](#evaluating-offical-models)
  - [Evaluating local checkpoint](#evaluating-local-checkpoint)
- [**Training**](#training)
- [**Gradio demo**](#gradio-demo)
- [**Citation**](#citation)


## **Usage**
It is recommended to fetch the latest [MiDaS repo](https://github.com/isl-org/MiDaS) via torch hub before proceeding:
```python
import torch

torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
```
### **ZoeDepth models** <!-- omit in toc -->
### Using torch hub
```python
import torch

repo = "isl-org/ZoeDepth"
# Zoe_N
model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)

# Zoe_K
model_zoe_k = torch.hub.load(repo, "ZoeD_K", pretrained=True)

# Zoe_NK
model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
```
### Using local copy
Clone this repo:
```bash
git clone https://github.com/isl-org/ZoeDepth.git && cd ZoeDepth
```
#### Using local torch hub
You can use local source for torch hub to load the ZoeDepth models, for example: 
```python
import torch

# Zoe_N
model_zoe_n = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)
```

#### or load the models manually
```python
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

# ZoeD_N
conf = get_config("zoedepth", "infer")
model_zoe_n = build_model(conf)

# ZoeD_K
conf = get_config("zoedepth", "infer", config_version="kitti")
model_zoe_k = build_model(conf)

# ZoeD_NK
conf = get_config("zoedepth_nk", "infer")
model_zoe_nk = build_model(conf)
```

### Using ZoeD models to predict depth 
```python
##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)


# Local file
from PIL import Image
image = Image.open("/path/to/image.jpg").convert("RGB")  # load
depth_numpy = zoe.infer_pil(image)  # as numpy

depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image

depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor



# Tensor 
from zoedepth.utils.misc import pil_to_batched_tensor
X = pil_to_batched_tensor(image).to(DEVICE)
depth_tensor = zoe.infer(X)



# From URL
from zoedepth.utils.misc import get_image_from_url

# Example URL
URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS4W8H_Nxk_rs3Vje_zj6mglPOH7bnPhQitBH8WkqjlqQVotdtDEG37BsnGofME3_u6lDk&usqp=CAU"


image = get_image_from_url(URL)  # fetch
depth = zoe.infer_pil(image)

# Save raw
from zoedepth.utils.misc import save_raw_16bit
fpath = "/path/to/output.png"
save_raw_16bit(depth, fpath)

# Colorize output
from zoedepth.utils.misc import colorize

colored = colorize(depth)

# save colored output
fpath_colored = "/path/to/output_colored.png"
Image.fromarray(colored).save(fpath_colored)
```

## LightBurn Heightmap Workflow

This fork now includes a focused export path for turning ZoeDepth predictions into LightBurn 3D Sliced-ready metal heightmaps without modifying the underlying model implementation.

The main entrypoint is [apps/zoe2lightburn.py](apps/zoe2lightburn.py), which wraps the existing `infer_pil(...)` inference API and adds:

- percentile-based depth normalization
- `black = deepest` or `white = deepest` export polarity
- background flattening for medallions, plaques, and coins
- smoothing plus optional detail recovery
- 8-bit LightBurn PNG export and 16-bit master export
- preview and calibration ramp generation
- YAML material profiles for MOPA workflows

Example:

```bash
python apps/zoe2lightburn.py input.jpg \
  --output out/coin_heightmap.png \
  --profile mopa_60w_brass \
  --near 5 \
  --far 95 \
  --gamma 0.72 \
  --flatten-background \
  --black-is-deep \
  --export-preview \
  --export-calibration-ramp
```

This produces a LightBurn-ready grayscale output plus companion files:

```text
out/
├─ coin_heightmap_lightburn.png
├─ coin_heightmap_master16.png
├─ coin_heightmap_preview.png
├─ coin_heightmap_ramp.png
└─ coin_heightmap_settings.json
```

To generate a standalone calibration ramp:

```bash
python apps/zoe2lightburn.py --make-ramp out/mopa_256_ramp.png
```

Profiles live in [profiles/mopa_60w_brass.yaml](profiles/mopa_60w_brass.yaml), [profiles/mopa_60w_stainless.yaml](profiles/mopa_60w_stainless.yaml), [profiles/mopa_60w_aluminum.yaml](profiles/mopa_60w_aluminum.yaml), and [profiles/mopa_60w_copper.yaml](profiles/mopa_60w_copper.yaml).

## **Environment setup**
The project depends on :
- [pytorch](https://pytorch.org/) (Main framework)
- [timm](https://timm.fast.ai/)  (Backbone helper for MiDaS)
- pillow, matplotlib, scipy, h5py, opencv (utilities)

Install environment using `environment.yml` : 

Using [mamba](https://github.com/mamba-org/mamba) (fastest):
```bash
mamba env create -n zoe --file environment.yml
mamba activate zoe
```
Using conda : 

```bash
conda env create -n zoe --file environment.yml
conda activate zoe
```

## **Sanity checks** (Recommended)
Check if models can be loaded: 
```bash
python sanity_hub.py
```
Try a demo prediction pipeline:
```bash
python sanity.py
```
This will save a file `pred.png` in the root folder, showing RGB and corresponding predicted depth side-by-side.
## Model files
Models are defined under `models/` folder, with `models/<model_name>_<version>.py` containing model definitions and  `models/config_<model_name>.json` containing configuration.

Single metric head models (Zoe_N and Zoe_K from the paper) have the common definition and are defined under `models/zoedepth` while as the multi-headed model (Zoe_NK) is defined under `models/zoedepth_nk`.
## **Evaluation**
Download the required dataset and change the `DATASETS_CONFIG` dictionary in `utils/config.py` accordingly. 
### Evaluating offical models
On NYU-Depth-v2 for example:

For ZoeD_N:
```bash
python evaluate.py -m zoedepth -d nyu
```

For ZoeD_NK:
```bash
python evaluate.py -m zoedepth_nk -d nyu
```

### Evaluating local checkpoint
```bash
python evaluate.py -m zoedepth --pretrained_resource="local::/path/to/local/ckpt.pt" -d nyu
```
Pretrained resources are prefixed with `url::` to indicate weights should be fetched from a url, or `local::` to indicate path is a local file. Refer to `models/model_io.py` for details. 

The dataset name should match the corresponding key in `utils.config.DATASETS_CONFIG` .

## **Training**
Download training datasets as per instructions given [here](https://github.com/cleinc/bts/tree/master/pytorch#nyu-depvh-v2). Then for training a single head model on NYU-Depth-v2 :
```bash
python train_mono.py -m zoedepth --pretrained_resource=""
```

For training the Zoe-NK model:
```bash
python train_mix.py -m zoedepth_nk --pretrained_resource=""
```
## **Gradio demo**
We provide a UI demo built using [gradio](https://gradio.app/). To get started, install UI requirements:
```bash
pip install -r ui/ui_requirements.txt
```
Then launch the gradio UI:
```bash
python -m ui.app
```

The UI is also hosted on HuggingFace🤗 [here](https://huggingface.co/spaces/shariqfarooq/ZoeDepth)
## **Citation**
```
@misc{https://doi.org/10.48550/arxiv.2302.12288,
  doi = {10.48550/ARXIV.2302.12288},
  
  url = {https://arxiv.org/abs/2302.12288},
  
  author = {Bhat, Shariq Farooq and Birkl, Reiner and Wofk, Diana and Wonka, Peter and Müller, Matthias},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth},
  
  publisher = {arXiv},
  
  year = {2023},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```













