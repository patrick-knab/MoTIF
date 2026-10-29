# MoTIF — Concepts in Motion

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2509.20899)
[![Checkpoints](https://img.shields.io/badge/Checkpoints-Hugging%20Face-yellow.svg)](https://huggingface.co/P4ddyki/MoTIF/tree/main)

MoTIF (Moving Temporal Interpretable Framework) extends concept bottleneck modeling from static images to videos. It combines concept activations from vision-language models with temporal modeling to explain video predictions through global concept importance, local concept windows, and temporal dependency maps.

**Accepted at NeurIPS 2026 (main track).** MoTIF also received the Best Paper Award at CompLearn @ ICML 2026 and was accepted as a Spotlight paper at XAI4CV @ CVPR 2026.

![MoTIF overview: VLM-based concept discovery and global, local, and temporal explanations for a shoot-bow video](assets/FIG_1_neu.png?v=20261027)

## Abstract

Concept Bottleneck Models (CBMs) enable interpretable image classification by structuring predictions around human-understandable concepts, but extending this paradigm to video remains challenging due to the difficulty of extracting concepts and modeling them over time. In this paper, we introduce MoTIF (Moving Temporal Interpretable Framework), a transformer-based concept architecture that operates on sequences of temporally grounded concept activations, by employing per-concept temporal self-attention to model when individual concepts recur and how their temporal patterns contribute to predictions. Central to the framework is a class-conditioned VLM-based concept discovery module that extracts object- and action-centric textual concepts from training videos, yielding temporally expressive concept sets without manual concept annotation. Across multiple video benchmarks, this combination improves over global concept bottlenecks and remains competitive within the interpretable concept-bottleneck setting, while narrowing the gap to strong black-box video baselines that we report as contextual references.

## Highlights

- Concept bottlenecks for video classification with temporal reasoning over concept activations
- Two training variants: temporal `MoTIF` and space-time `MoTIF-ST`
- Three explanation views: global relevance, local windows, and temporal attention maps
- Support for multiple vision-language backbones, including CLIP, SigLIP, CLIP4Clip, and Perception Encoder variants
- Optional agentic concept extraction pipeline for VLM-generated concepts and ablations
- Example workflows for UCF-101, HMDB-51, Something-Something v2, and Breakfast Actions

## Repository Contents

- `train_MoTIF.py`: train the temporal MoTIF model (`--help` lists the command-line overrides)
- `train_MoTIF-ST.py`: train the space-time MoTIF-ST variant
- `embedding.py`: generate the window-level video embeddings used by the models
- `save_videos.py`: prepare derived `Video_data/` and `Image_data/` folders from raw datasets
- `finetune_pe_l14_ssv2.py`: backbone adaptation on SSv2 manifests (`--adapter lora` or `--adapter partial`)
- `make_ssv2_manifests.py`: build the SSv2 class list and train/validation manifests from the official label files
- `MoTIF.ipynb`, `MoTIF-ST.ipynb`: inspect predictions, concepts, and explanations
- `Synthetic_temporal_pattern.ipynb`: synthetic temporal pattern experiments
- `interventions.ipynb`, `corrective_interventions_manual.ipynb`, `app_corrective_interventions.py`: intervention and correction workflows
- `slurm_cbm_extract_windows.sbatch`: SLURM launcher for VLM-based concept extraction (vLLM server + extractor)
- `utils/extract_cbm_concepts_from_image_data_windows.py`: the concept extraction CLI used by the launcher
- `utils/manual_concepts.py`: hand-written fallback concept vocabularies (one list entry per concept)
- `utils/lora.py`: dependency-free LoRA adapters for the vision tower (apply, save, load, merge)
- `utils/paths.py`: repository-local data layout shared by all scripts
- `MoTIF_DCBM_Style/`: DCBM-style concept extraction and ablation utilities (expects the layout documented in its own README)
- `Videos/`: small example videos for qualitative inspection
- `Datasets/`, `Embeddings/`, `Models/`: placeholder folders for external data and generated artifacts
- `utils/`: model, embedding, explanation, intervention, and data-loading code

All scripts resolve `Datasets/`, `Embeddings/`, `Models/` and concept runs relative to the repository root, so they can be started from any working directory.

## What Is Not Stored in Git

Large artifacts are intentionally not tracked in this repository:

- trained checkpoints in `Models/`
- generated embeddings in `Embeddings/`
- raw datasets and derived `Video_data/` / `Image_data/` folders under `Datasets/`
- VLM concept runs in `concept_extraction_out_batch/` and adaptation outputs in `pe_l14_ssv2_*/`
- cache files, bytecode, and other local machine artifacts

Pretrained checkpoints and PE-L/14 embeddings are available on [Hugging Face](https://huggingface.co/P4ddyki/MoTIF/tree/main). Embeddings can also be regenerated locally with `embedding.py`.

## Setup

MoTIF was run in our experiments with Python 3.13.5 on GPU machines. A CUDA-enabled GPU is strongly recommended, especially for embedding generation and training.

### Reproducible setup with uv (recommended)

Install [uv](https://docs.astral.sh/uv/) (Python 3.11–3.13) and create the locked environment:

```bash
uv sync --locked
```

Run commands inside that environment with `uv run`:

```bash
uv run --locked python utils/extract_cbm_concepts_from_image_data_windows.py --help
uv run --locked python finetune_pe_l14_ssv2.py --help
```

The committed `uv.lock` pins the resolved dependency graph, including PyTorch 2.8.0 and torchvision 0.23.0. The environment was smoke-tested on macOS with Python 3.11; CUDA training and live Qwen/vLLM inference require a suitable GPU host.

### pip setup

If you do not use uv, install the main dependencies with pip:

```bash
pip install -r requirements.txt
```

### Optional vLLM server (separate Linux/CUDA environment)

The extraction client is included in the main environment and can connect to a local or remote OpenAI-compatible server. The vLLM server requirements pin a different PyTorch/CUDA stack; install them in a separate environment on the GPU host:

```bash
uv venv --python 3.11 .venv-vllm
uv pip install --python .venv-vllm/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements_vllm.txt
```

See `slurm_cbm_extract_windows.sbatch` for the server/extraction launcher. Paths supplied as `file://` media must exist on the server host and be permitted by vLLM's `--allowed-local-media-path` setting. The serving environment is separate from `uv.lock` and has not been validated on this Mac.

## Data Layout

Place the original dataset files under `Datasets/<DatasetName>/Data/` together with the official split files. The preprocessing step then creates the derived `Video_data/` and `Image_data/` folders used by the training and concept extraction code.

Expected layout:

```text
Datasets/
  HMDB/
    Data/<class>/<clip>.avi            # or Data/<class>/<class>/<clip>.avi, see below
    testTrainMulti_7030_splits/        # from test_train_splits.rar (official HMDB-51 download)
    Video_data/<class>/<clip>.mp4      # created by save_videos.py
    Image_data/<class>/<clip>/*.jpg    # created by save_videos.py
  UCF101/
    Data/<class>/<clip>.avi
    ucfTrainTestlist/                  # official split lists (trainlist0k.txt / testlist0k.txt)
    Video_data/
    Image_data/
  Something2/
    Data/<id>.webm
    labels/                            # official labels.json, train.json, validation.json, test.json, test-answers.csv
    Video_data/<id>.mp4
    Image_data/<id>/*.jpg
  Breakfast/
    Data/P03/cam01/*.avi ...
    Video_data/
    Image_data/
```

Notes:

- `save_videos.py` reads from `Data/` and the split files, and writes the derived `Video_data/` and `Image_data/` folders.
- The split files are **not** included in this repository; download them from the respective dataset websites and place them as shown above.
- HMDB-51 ships one archive per class. Both `Data/<class>/<clip>.avi` and `Data/<class>/<class>/<clip>.avi` (each archive extracted into a folder of its own name) are accepted.
- Dataset videos must be obtained from the original dataset providers and used under their respective licenses.

## Quickstart

The commands below reproduce the default configuration of the training scripts: HMDB-51, split `s1`, PE-L/14 backbone, window size 8.

### 1. Install dependencies

```bash
uv sync --locked            # or: pip install -r requirements.txt
```

### 2. Place raw datasets and split files under `Datasets/<DatasetName>/`

Follow each dataset's official download instructions (see Data Layout).

### 3. Create derived video and frame folders

```bash
python save_videos.py --dataset HMDB          # also: UCF101, Something2, Breakfast
```

### 4. Generate embeddings

```bash
python embedding.py --dataset hmdb51 --clip-model pe-l14 --window-size 8
```

This writes `Embeddings/Videos/HMDB/True_8_clip_pe-l14.pkl`, which is where the training scripts look for it (`<random>_<window_size>_clip_<backbone>.pkl`). Use `--pe-video-batch-size` / `--pe-target-t` to trade speed for memory on the Perception Encoder path, and `--help` for the other backbones. Alternatively download `Embeddings/HMDB_pe-l14.pkl` from Hugging Face and pass it with `--embedding-path` in step 6.

### 5. Provide concepts

The training scripts use VLM-extracted (agentic) concepts by default and expect them in `concept_extraction_out_batch/HMDB/8B`, which is exactly the `--output_root` written by `slurm_cbm_extract_windows.sbatch`. To create that folder, start a vLLM server with `Qwen/Qwen3-VL-8B-Thinking` (the launcher does this) and run:

```bash
python utils/extract_cbm_concepts_from_image_data_windows.py \
  --image_data_dir Datasets/HMDB/Image_data \
  --output_root concept_extraction_out_batch/HMDB/8B \
  --dataset_name hmdb --instances_per_class 5 --shuffle_instances --seed 0 \
  --llm_server_url http://127.0.0.1:8001/v1 --llm_model Qwen/Qwen3-VL-8B-Thinking \
  --llm_api_key EMPTY --llm_max_tokens 3500 \
  --window_size 30 --max_windows 8 --max_window_overlap 0.25 --max_concepts 6
```

The result is one `concepts.json` per video window under `<output_root>/<class>/<instance>/windows/<window>/`. The training scripts walk the folder recursively, drop instances that belong to the selected test split, and filter near-duplicate concepts (`--similarity-threshold`).

No GPU server at hand? Pass `--manual-concepts` in the next step to train with the hand-written vocabularies in `utils/manual_concepts.py` instead.

### 6. Train the temporal MoTIF model

```bash
python train_MoTIF.py --dataset hmdb51 --clip-model pe-l14 \
  --agent-run-folder concept_extraction_out_batch/HMDB/8B
# or, without a concept run:
python train_MoTIF.py --dataset hmdb51 --clip-model pe-l14 --manual-concepts
# or, with the embeddings downloaded from Hugging Face:
python train_MoTIF.py --dataset hmdb51 --clip-model pe-l14 --manual-concepts \
  --embedding-path Embeddings/HMDB_pe-l14.pkl
```

If the concept run is missing, the script stops immediately with the extraction command above. The remaining hyperparameters live in the `search_space` grid at the bottom of the script; `--help` lists the ones that can be overridden on the command line. Checkpoints are written to `Models/checkpoint_<backbone>_<Dataset>.pkl`.

### 7. Train the space-time MoTIF-ST model

```bash
python train_MoTIF-ST.py --dataset hmdb51 --clip-model pe-l14 \
  --agent-run-folder concept_extraction_out_batch/HMDB/8B
```

### 8. Inspect checkpoints and explanations

Open the notebooks to visualize concepts, temporal attention, interventions, and prediction behavior:

- `MoTIF.ipynb`
- `MoTIF-ST.ipynb`
- `interventions.ipynb`
- `corrective_interventions_manual.ipynb`

## Pretrained Checkpoints

Pretrained checkpoints are available on [Hugging Face](https://huggingface.co/P4ddyki/MoTIF/tree/main). The repository currently includes released PE-L/14 models and embeddings for Breakfast, HMDB-51, and UCF-101:

| File on Hugging Face | Use |
| --- | --- |
| `Models/checkpoint_pe-l14_<Dataset>.pkl` | place in `Models/`; loaded by the notebooks |
| `Embeddings/<Dataset>_pe-l14.pkl` | pass to the training scripts with `--embedding-path` |

## Backbones and Datasets

### Supported vision-language backbones

- CLIP ViT-B/32
- CLIP ViT-B/16
- CLIP ViT-L/14
- CLIP RN50
- CLIP4Clip
- SigLIP
- SigLIP SO400M L/14
- Perception Encoder PE-L/14
- Perception Encoder PE-G/14

### Example datasets

- UCF-101
- HMDB-51
- Something-Something v2
- Breakfast Actions

Using additional datasets is possible, but typically requires adapting the dataset handling in `utils/core/data/`, `utils/paths.py`, `embedding.py`, and the training scripts.

## Agentic Concept Extraction

The optional pipeline for extracting concepts from video frames with a vision-language model consists of:

1. Install the main environment with `uv sync --locked`.
2. Start a vLLM server with a vision-language model in the separate Linux/CUDA serving environment described above.
3. Run `utils/extract_cbm_concepts_from_image_data_windows.py` (exact command in Quickstart step 5), or adapt `slurm_cbm_extract_windows.sbatch`, which starts the server and runs the extractor on one node. Set `DATASET_NAME`, `IMAGE_DATA_DIR` and `OUTPUT_ROOT` in the launcher for other datasets; SSv2 additionally needs `--ssv2_labels_json Datasets/Something2/labels/train.json`.
4. Pass the `--output_root` folder to `train_MoTIF.py` or `train_MoTIF-ST.py` via `--agent-run-folder`.

The extraction pipeline supports ablation settings such as `json_only`, `json+action`, and `action_only` (the training scripts use `json+action`). `MoTIF_DCBM_Style/` contains the DCBM-style ablation toolkit; its bridge modules expect the directory layout documented in `MoTIF_DCBM_Style/README.md`.

## SSv2 Backbone Adaptation (LoRA)

For Something-Something v2 the PE-L/14 backbone is adapted before the embeddings are computed. The released procedure has three steps, all driven by the official SSv2 split files:

### 1. Build manifests

```bash
python save_videos.py --dataset Something2
python make_ssv2_manifests.py --out-dir pe_l14_ssv2_lora --train-per-class 50 --seed 0
```

This writes `class_names.json` (templates in the order of `labels.json`, with `[something]` placeholders collapsed exactly as `utils/motif.py` names SSv2 classes) plus `train_manifest.jsonl` / `val_manifest.jsonl` with one `{"video_path": ..., "class_idx": ...}` per video. `--train-per-class` draws a balanced training subset; omit it to use the full training split.

### 2. Train the LoRA adapter

```bash
uv run --locked python finetune_pe_l14_ssv2.py --adapter lora \
  --manifest pe_l14_ssv2_lora/train_manifest.jsonl \
  --val-manifest pe_l14_ssv2_lora/val_manifest.jsonl \
  --class-names pe_l14_ssv2_lora/class_names.json \
  --out-dir pe_l14_ssv2_lora --backbone pe-l14 \
  --lora-rank 8 --lora-alpha 16 --lora-targets attn --lora-last-blocks 0 \
  --epochs 5 --batch-size 4 --grad-accum 8 --lr 1e-4
```

The objective is cross-entropy over `logit_scale * <video, class text>` similarities with frozen class-text embeddings; only the LoRA matrices (`utils/lora.py`, applied to the attention q/k/v and output projections, optionally `--lora-targets attn,mlp`) and the logit scale are trained. `best.pt` / `latest.pt` contain the adapter weights and the LoRA configuration only.

### 3. Embed SSv2 with the adapted backbone and train MoTIF

```bash
python embedding.py --dataset something2 --clip-model pe-l14 --window-size 8 \
  --lora-adapter pe_l14_ssv2_lora/best.pt
python train_MoTIF.py --dataset something2 --clip-model pe-l14 \
  --agent-run-folder concept_extraction_out_batch/Something2/8B
```

`embedding.py` merges the adapter into the backbone before embedding, so the resulting pickle is a regular `VideoEmbedder`.

`--adapter partial` (fine-tune the last `--train-last-blocks` blocks, the output layers and the logit scale; full state dict in the checkpoint) remains available as an alternative but is not the paper's procedure.

## Citation

If you use MoTIF in your research, please cite:

```bibtex
@misc{knab2025conceptsinmotion,
  title={Concepts in Motion: Temporal Concept Bottleneck Model for Interpretable Video Classification},
  author={Patrick Knab and Sascha Marton and Philipp J. Schubert and Drago Guggiana and Christian Bartelt},
  year={2025},
  eprint={2509.20899},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2509.20899}
}
```

## Acknowledgements

- Parts of `utils/core` are adapted from the Perception Encoder framework.
- Thanks to the CORE research group at TU Clausthal and Ramblr.ai Research for support.

## Contact

For questions, bug reports, or collaboration inquiries, please open a GitHub issue.
