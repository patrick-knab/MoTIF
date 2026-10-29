# Per-concept vs. full attention: blinded LLM-as-judge study

Material for the LLM-as-judge comparison of Section 3.2.1: on videos where
MoTIF's per-concept temporal attention (`diagonal_attention=true`, called
*channel-reserved* in the code) and a full cross-concept attention variant
predict the same class, two video-capable judges compare the two models'
matched three-level explanations. Over 400 trials, the diagonal-attention
explanations were preferred in 67.3% of comparisons vs. 17.3% for full
attention (15.4% ties).

## Protocol

- **Matched models.** For each dataset the two arms reuse exactly the same
  frozen concept activations, vocabulary, backbone inputs, labels and split;
  only the temporal block differs (three seeds; the judged pair uses seed 42;
  SHA-256 hashes of the checkpoints are in `study_metadata.json`). The
  checkpoints themselves are not included.
- **Two replications, 200 trials each**, so that every dataset is judged with
  both backbones:

  | Replication | Breakfast | HMDB51 | Config | Study | Analysis |
  |---|---|---|---|---|---|
  | main | CLIP RN50 | CLIP ViT-B/32 | `config.json` | `results/study` | `results/analysis` |
  | cross-backbone | CLIP ViT-B/32 | CLIP RN50 | `config_cross_backbone.json` | `results/cross_backbone/study` | `results/cross_backbone/analysis` |

  Each replication samples 50 correct and 50 incorrect common-prediction
  videos per dataset with a fixed seed.
- **Judges.** `Qwen/Qwen3.5-27B` and `Qwen/Qwen3.6-35B-A3B` served with
  vLLM, 64 sampled frames per video, temperature 0.2, structured JSON output,
  thinking disabled. Each judge scores every trial in three randomized runs
  (seeds 17, 29, 43), so each trial receives six ratings. Before scoring, an
  independent media sanity pass recorded a literal description of each video
  and flagged absent, blank or corrupt frames; none occurred.
- **Blinding.** Condition labels are hidden and the presentation order is
  shuffled per request. Besides the two real explanations, each request
  contains two negative controls, `scrambled_concepts` (concept names permuted)
  and `class_name_only`. Ground-truth labels are never revealed. The judge
  rates every explanation on concept identity coherence, evidence relevance,
  temporal localization and diagnostic usefulness (1-5) and states a pairwise
  preference between the two real explanations (`judge.py`).
- **Analysis** (`analyze.py`). The six ratings of a trial are aggregated
  within the trial before any statistic is computed; 95% confidence intervals
  use 10,000 video-level bootstrap resamples. Pre-registered criteria require
  channel-reserved to be non-inferior to full attention (margin -0.25) on
  every dimension and both real explanations to beat the controls.

## Results

Pairwise preference (channel-reserved / full attention / tie), pooled over
both judges and all three runs:

| | main (200 trials) | cross-backbone (200 trials) | pooled (400 trials) |
|---|---:|---:|---:|
| all | 63.3% / 19.6% / 17.1% | 71.2% / 15.1% / 13.8% | **67.3% / 17.3% / 15.4%** |
| Breakfast | 44.7% / 29.0% / 26.3% | 61.3% / 23.3% / 15.3% | 53.0% / 26.2% / 20.8% |
| HMDB51 | 82.0% / 10.2% / 7.8% | 81.0% / 6.8% / 12.2% | 81.5% / 8.5% / 10.0% |

The pooled row is the number reported in the paper; both replications and both
judges show the same direction on both datasets, with a smaller margin on the
long multi-stage Breakfast videos. Mean score differences (channel-reserved
minus full attention, 1-5 scale) are +1.18 / +1.36 for concept identity
coherence, +1.23 / +1.45 for evidence relevance, +1.02 / +1.20 for temporal
localization and +1.18 / +1.37 for diagnostic usefulness (main /
cross-backbone). All pooled primary criteria and control checks pass in both
replications. Running `analyze.py` (below) writes these tables with their
confidence intervals, the long-form scores and preferences, and the
Krippendorff agreement between judges and runs. The judge scores measure explanation
quality and semantic coherence; they do not by themselves establish
architectural isolation.

## Files

- `results/{study,cross_backbone/study}/responses/<judge>/seed_<s>_fidelity.jsonl`:
  raw judge responses, including the blinded condition map, the ratings and
  the model's JSON output.
- `results/{study,cross_backbone/study}/stimuli.internal.jsonl`: the blinded
  explanation texts of every condition; `manifest.jsonl`: trial list with
  video paths relative to the repository's `Datasets/`; `study_metadata.json`.
- `judge.py`: prompt construction, blinding and response schema.
  `analyze.py` (with `common.py` and `../llm_judge/`): the analysis.
  `config.json`, `config_cross_backbone.json`: judges, seeds and settings.

## Reproduce

```bash
cd evaluation_studies
python -m attention_isolation.analyze --config attention_isolation/config.json \
  --study-dir attention_isolation/results/study \
  --output attention_isolation/results/analysis --bootstrap-replicates 10000
python -m attention_isolation.analyze --config attention_isolation/config_cross_backbone.json \
  --study-dir attention_isolation/results/cross_backbone/study \
  --output attention_isolation/results/cross_backbone/analysis --bootstrap-replicates 10000
```

Prompts can be inspected without a model server:

```bash
python -m attention_isolation.judge --study-dir attention_isolation/results/study \
  --judge-index 0 --run-seed 17 --stage fidelity --limit 1 --dry-run
```
