# Human evaluation of local explanations

Complete material for the blinded human preference study of Section 3.2.2:
frozen stimuli, anonymous rater assignments, the rating interface, every
submitted response, the analysis code and its outputs. The text shown to
raters is reproduced verbatim in [`RATER_INSTRUCTIONS.md`](RATER_INSTRUCTIONS.md);
the interface itself is [`app.py`](app.py) (Streamlit).

## Design

- 24 video-model trials: three from each dataset x backbone x
  prediction-correctness stratum (Breakfast and HMDB51; CLIP RN50 and
  ViT-B/32; correct and incorrect predictions), drawn from the frozen
  LLM-judge stimulus pools of the attention-isolation study.
- Every video has more than four model windows, and the cited integer-second
  interval covers at most 40% of the video duration.
- Two conditions with a matched display budget of two sections and eight
  ranked concept rows each. *Global-only* shows eight global concepts.
  *Compact multi-level* shows the same first four global concepts plus four
  concepts localized to the most important temporal window, shown with a
  second player that plays only that window. The compact condition is
  deliberately not the longer complete three-level explanation: it isolates
  the incremental value of local/temporal information at equal length.
- Eight anonymous adult raters, three independent ratings per trial, 72
  ratings in total. A/B placement is randomized per rater and trial; ground
  truth and prediction correctness are hidden. Forced choice with a tie
  option, plus a confidence rating and the response time.

## Raters and procedure

Raters were adults who are not authors of the paper and needed no
machine-learning expertise. Each received only an anonymous code from
`assignments.csv`; no names or other personal information were collected.
Raters were told that participation was voluntary and could be stopped at any
time, and confirmed in the interface that they were at least 18 and consented
to participate. They were compensated above the statutory minimum wage of the
country in which the study was conducted (Germany). The study posed no
foreseeable risk beyond the time commitment.

Frozen before any response file was opened: stop only after all assigned
ratings are complete; include every submitted response (exclude only for a
documented technical failure, decided without inspecting the choice; none
occurred); the pooled +1/0/-1 preference score is the endpoint, everything
else is descriptive; no change to trials, rendering or instructions after
collection begins. One display-only clarification was made after `rater01`
and `rater02` had finished: the existing concept order was made visually
explicit with ordinal numbers and a sentence stating that the order reflects
positive contribution within each scope. No stimuli, order, conditions,
questions or options changed, so all ratings are pooled under the same protocol
`motif_human_usefulness`.

## Results

72 of 72 assigned ratings were completed by 8 raters on 24 trials.

| Compact multi-level | Global-only | Tie | Compact among decisive | Mean score (+1/0/-1) | Trial-bootstrap 95% CI |
|---:|---:|---:|---:|---:|---|
| 55.6% | 23.6% | 20.8% | 70.2% | +0.319 | [0.083, 0.542] |

Exact agreement among the three raters of a trial was 47.2%. The confidence
interval resamples trials (10,000 replicates) after averaging ratings within
trial and treats the recruited raters as fixed; the study does not support
population-wide claims about all users. Running `analyze.py` (below) writes
`results/` with `RESULTS.md`, pooled and subgroup summaries
(`preference_summary.csv`) and every decoded rating
(`responses_decoded.csv`).

## Files

- `responses/rater01.jsonl` ... `rater08.jsonl`: raw submitted responses
  (anonymous code, trial, blinded A/B choice, confidence, response time;
  nothing else).
- `assignments.csv`: anonymous rater assignments with the hidden A/B mapping
  that decodes the responses.
- `study_bundle.jsonl`: frozen trial stimuli and condition identities
  (internal; never shown to raters).
- `app.py`: the rating interface (Streamlit). `analyze.py`: the
  pre-registered analysis.
- The H.264 clips served to raters are not included; they are re-encoded
  Breakfast and HMDB51 videos whose source files are listed in
  `study_bundle.jsonl` (`source_video_path`, relative to `Datasets/`).

## Reproduce

```bash
cd evaluation_studies
python -m human_study.analyze --output human_study/results
```
