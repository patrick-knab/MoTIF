# Explanation evaluation studies

Supplementary material for the two explanation evaluations reported in the
paper, with raw data, code and regenerated analyses:

| Study | Paper | Folder | Result |
|---|---|---|---|
| Human evaluation of local explanations (8 raters, 24 trials, 72 ratings) | Section 3.2.2 | [`human_study/`](human_study/) | global+local preferred in 55.6% of ratings vs. 23.6% global-only (20.8% ties); mean preference +0.319, trial-bootstrap 95% CI [0.083, 0.542] |
| Blinded LLM-as-judge comparison of per-concept (diagonal) vs. full cross-concept attention (400 trials) | Section 3.2.1 | [`attention_isolation/`](attention_isolation/) | diagonal-attention explanations preferred in 67.3% of comparisons vs. 17.3% (15.4% ties) |

`llm_judge/` holds the two analysis helpers imported by `attention_isolation/`.

## Running

The analyses only need the included response files and run on CPU from this
folder:

```bash
cd evaluation_studies
python -m human_study.analyze --output human_study/results
python -m attention_isolation.analyze --config attention_isolation/config.json \
  --study-dir attention_isolation/results/study \
  --output attention_isolation/results/analysis --bootstrap-replicates 10000
python -m attention_isolation.analyze --config attention_isolation/config_cross_backbone.json \
  --study-dir attention_isolation/results/cross_backbone/study \
  --output attention_isolation/results/cross_backbone/analysis --bootstrap-replicates 10000
```

Only raw data and the scripts that evaluate it are included; the analysis
outputs are produced by these commands. Paths in the data files are relative
to the repository root.
