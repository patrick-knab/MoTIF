# MoTIF — Concepts in Motion

[**Read the Paper (arXiv)**](https://arxiv.org/pdf/2509.20899)

## Abstract

Conceptual models such as Concept Bottleneck Models (CBMs) have driven substantial progress in improving interpretability for image classification by leveraging human‑interpretable concepts. However, extending these models from static images to sequences of images, such as video data, introduces a significant challenge due to the temporal dependencies inherent in videos, which are essential for capturing actions and events. In this work, we introduce MoTIF (Moving Temporal Interpretable Framework), an architectural design inspired by a transformer that adapts the concept bottleneck framework for video classification and handles sequences of arbitrary length. Within the video domain, concepts refer to semantic entities such as objects, attributes, or higher‑level components (e.g., "bow", "mount", "shoot") that reoccur across time—forming motifs collectively describing and explaining actions. Our design explicitly enables three complementary perspectives: global concept importance across the entire video, local concept relevance within specific windows, and temporal dependencies of a concept over time. Our results demonstrate that the concept‑based modeling paradigm can be effectively transferred to video data, enabling a better understanding of concept contributions in temporal contexts while maintaining competitive performance. 

---

## Key Features

- **Concept Bottlenecks for Video**: map frames/clips to a shared image–text space and obtain concept activations by cosine similarity.
- **Per‑Channel Temporal Self‑Attention**: concept channels stay independent; attention happens over time within each concept.
- **Three Explanation Views**: global concept relevance, local window concepts, and attention‑based temporal maps.
- **Plug‑and‑Play Backbones**: designed to work with CLIP and related vision–language models.
- **Multiple Datasets**: examples provided for UCF‑101, HMDB‑51, Something‑Something v2, and Breakfast Actions.

---

## Getting Started

### 1) Environment

- Python 3.10+ (tested with 3.13.5)
- CUDA‑enabled GPU recommended (checkpoints and scripts assume a GPU environment)

Create and activate an environment, then install requirements:

```bash
pip install -r requirements.txt
```

### 2) Data

Place your datasets under `Datasets/` (see the folder structure below). If you want to generate small demo clips or frames, you can use:

```bash
python save_videos.py
```

### 3) Create Embeddings

Compute (or recompute) the video/frame embeddings used by MoTIF:

```bash
python embedding.py
```


### 4) Train MoTIF

MoTIF’s training entry point is:

```bash
python train_MoTIF.py
```

Adjust hyperparameters in the script or via CLI flags (if exposed).

### 5) Explore and Visualize

- Open `MoTIF.ipynb` to visualize concept activations, attention over time, and example predictions.
- Place model checkpoints in `Models/` (see the notebook and code comments for expected paths).

---

## Pretrained Checkpoints

Pretrained MoTIF checkpoints will be uploaded soon. We will publish them under the repository “Releases” page and mirror links here. In the meantime, you can train your own models following the steps above and save checkpoints in `Models/`.

---

## Backbones and Datasets

### Vision–Language Backbones
- CLIP ViT‑B/32 — [Hugging Face: openai/clip‑vit‑base‑patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- CLIP ViT‑B/16 — [Hugging Face: openai/clip‑vit‑base‑patch16](https://huggingface.co/openai/clip-vit-base-patch16)
- CLIP ViT‑L/14 — [Hugging Face: openai/clip‑vit‑large‑patch14](https://huggingface.co/openai/clip-vit-large-patch14)
- (Optional) SigLIP L/14 — [Hugging Face: google/siglip‑so400m‑patch14‑384](https://huggingface.co/google/siglip-so400m-patch14-384)
- Perception Encoder (PE‑L/14) — [Official Repo on GitHub](https://github.com/facebookresearch/perception_models)

### Datasets
- UCF‑101 — [Project page](https://www.crcv.ucf.edu/data/UCF101.php)
- HMDB‑51 — [Project page](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
- Something‑Something v2 — [20BN dataset page](https://www.qualcomm.com/developer/software/something-something-v-2-dataset)
- Breakfast Actions — [Dataset page](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/)

Please follow each dataset’s license and terms of use.

Note: If you use other datasets, you will need to adapt the dataset logic in the code (e.g., train/val/test splits, preprocessing, and loaders). Relevant places include `utils/core/data/` (e.g., `data.py`, `preprocessor.py`, `dataloader.py`) and any dataset‑specific handling in `embedding.py` and `train_MoTIF.py`.

---

## Folder Structure

- `Datasets/` — dataset placeholders
- `Embeddings/` — generated embeddings (created by scripts)
- `Models/` — trained model checkpoints
- `Videos/` — example videos used in the paper/one‑pager
- `utils/` — library code (vision encoder, projector, dataloaders, transforms, etc.)
- `index.html` — minimal one‑pager describing MoTIF (open locally in a browser)
- `embedding.py`, `save_videos.py`, `train_MoTIF.py` — main scripts
- `MoTIF.ipynb` — notebook for inspection and visualization

---

## Quick Tips

- If you change the dataset or backbone, regenerate embeddings before training.
- The attention visualizations are concept‑wise and time‑wise; they should not mix information across concepts.
- GPU memory usage depends on the number of concepts and the temporal window length.

---

## Citation

If you use MoTIF in your research, please consider citing:

```bibtex
@misc{knab2025conceptsmotiontemporalbottlenecks,
      title={Concepts in Motion: Temporal Bottlenecks for Interpretable Video Classification}, 
      author={Patrick Knab and Sascha Marton and Philipp J. Schubert and Drago Guggiana and Christian Bartelt},
      year={2025},
      eprint={2509.20899},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.20899}, 
}
```

---

## Acknowledgements

- Parts of the `utils/core` codebase are adapted from the Perception Encoder framework.
- Thanks to the CORE research group at TU Clausthal and Ramblr.ai Research for support.

---

## Contact

For questions and discussion, please open an issue or contact the authors.