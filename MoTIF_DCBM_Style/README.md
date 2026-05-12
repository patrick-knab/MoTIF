# Ablation Toolkit

This folder contains a small generic toolkit for:

- launching concept extraction for any supported dataset
- recording which ablation stages a concept run is intended to support
- running a single AgentMoTIF ablation without using a dataset-specific `run_ablations_*.py` file

Supported datasets:

- `breakfast`
- `hmdb51`
- `ucf101`
- `something2`

Supported ablation stages:

- `json_only`
- `json+action`
- `json+action+visual`
- `json+action+visual+motion`
- `action_only`
- `visual_only`

## Files

- `extract_concepts_for_ablation.py`
- `extract_cbm_concepts_from_image_data_windows.py`
- `run_single_ablation.py`
- `utils/`

## Local Utils

The toolkit folder now also contains local utility bridge files:

- `utils/cbm_concept_extraction_utils.py`
- `utils/concept_handling.py`
- `utils/motif.py`
- `utils/video_embedder.py`

These are the toolkit-local entrypoints used by the scripts in this folder.

`utils/cbm_concept_extraction_utils.py` is now vendored locally and includes the crop / motion-GIF extraction logic used by the concept pipeline.

## SAM3 Note

This toolkit now contains the local extraction-side utility code, but the current repo does not contain a complete standalone text-conditioned `SAM3`/AgentSAM inference adapter that plugs into `collect_concept_detections(...)`.

`SAM3` itself is not present in this folder.

What is local now:

- VLM concept proposal
- dataset/window selection
- crop deduplication
- motion GIF generation
- concept JSON writing

What is still not present as a standalone local implementation in this folder:

- a concrete `inference_fn(processor, frame_path, concept)` backend that produces `pred_boxes` / `pred_scores` from a SAM3-style detector/segmentor stack

So the toolkit is more self-contained than before, but the actual segmentation/detection backend still depends on whatever AgentSAM/SAM-style runner you use externally.

## Notes

The concept extractor backend writes one concept bundle per window. That bundle can then be reused for all ablation stages at training time. Because of that, the extraction launcher stores the requested stages as metadata in `ablation_stage_manifest.json` inside the output run folder.

## Example

```bash
cd VideoCBM/AgentMoTIF

python ablation_toolkit/extract_concepts_for_ablation.py \
  --dataset breakfast \
  --llm-server-url http://127.0.0.1:8000/v1 \
  --llm-model Qwen/Qwen3-VL-8B-Thinking \
  --instances-per-class 5 \
  --window-size 80 \
  --ablation-stage all

python ablation_toolkit/run_single_ablation.py \
  --dataset breakfast \
  --clip-model pe-l14 \
  --ablation-stage visual_only \
  --test-split s1 \
  --window-size 32 \
  --agentsam-run-folder vlm-Qwen-Qwen3-VL-30B-A3B-Thinking_ipc5_seed0_ws80
```
