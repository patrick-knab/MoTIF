# Text shown to raters

This file reproduces, verbatim, the text displayed by the rating interface
(`app.py`) for the usefulness trials reported in the paper. Raters used the
interface in a web browser; the layout is described in brackets. The landing
page also introduces a second trial type (concept grounding) that the same
sessions contained; those trials are not part of the reported study and are
not included here.

## Landing page

**Page title:** Video explanation study

[Blue information banner:]

> No machine-learning expertise is required. Use ordinary visual judgment:
> watch what happens in the video and decide which explanation or concept list
> would be clearer and more useful to you.

[Expandable section "Instructions and consent", open by default:]

> This study contains two types of anonymized A/B comparison:
>
> 1. **Explanation usefulness:** decide which explanation better helps you
>    understand why a video model made its displayed prediction.
> 2. **Local concept grounding:** watch a short clip and decide which concept
>    list better matches what is visibly present in that clip.
>
> The **model prediction** is the activity label produced by the video model.
> A **concept** is a short object or action word that the model identified as
> evidence for that prediction. A concept listed across the whole video may
> matter somewhere in the video but does not say when. A localized concept is
> tied to the displayed time interval.
>
> Concepts are shown in ranked order according to their positive contribution
> to the model prediction within the stated scope: rank 1 has the largest
> contribution, followed by ranks 2, 3, and so on. The complete-video and
> cited-segment sections are separate rankings, so each can begin at rank 1.
>
> For example, if a model predicts "making tea," a whole-video explanation
> might list `kettle`, `cup`, and `pour`. A localized explanation might
> additionally say that `pour` and `cup` are important specifically from
> `00:12` to `00:15`.
>
> For usefulness trials, evidence "across the whole video" does not say when a
> concept appears. Evidence "localized to X–Y" associates concepts with that
> specific segment. One explanation provides more whole-video concepts; the
> other uses the same number of entries but replaces some with localized
> evidence.
>
> For grounding trials, judge only whether the concept words match the visible
> clip. The model prediction is intentionally hidden. "Tie" is appropriate
> when neither side is better.
>
> There are no trick questions. The study stores only your anonymous
> participant code, choices, confidence, and response time. Participation is
> voluntary; you may stop at any time. Do not enter your name or other
> personal information.

[Text field:] Anonymous participant code

[Checkbox, required:] I am at least 18 and consent to participate.

[Progress bar with caption:] Completed {n} of {N} trials

The paragraph "Concepts are shown in ranked order ..." and the ordinal numbers
in front of each concept were added after the first two raters had finished;
nothing else changed.

## Usefulness trial

**Heading:** Task: explanation usefulness

**Model prediction:** `{predicted class}`

[Two video players side by side:]

- **1. Complete video (context)** — caption: "Use the entire video to
  understand the model's prediction."
- **2. Cited segment ({mm:ss–mm:ss})** — caption: "This excerpt comes from the
  same video. Only this interval is associated with the localized concepts."
  (The second player starts at the cited window and stops at its end.)

[Blue information banner:]

> **What differs in this trial:** Both explanations share the first four
> whole-video concepts. One uses its remaining four entries for additional
> whole-video concepts; the other uses them for concepts associated with
> **{mm:ss–mm:ss}**. The same concept may appear in both sections because they
> summarize different temporal scopes.

[Two columns, "Explanation A" and "Explanation B". Each explanation consists
of two boxed sections with eight numbered concept rows in total:]

- Global-only condition: "Complete-video ranking: ranks 1–4 (shared by A and
  B)" followed by "Complete-video ranking: ranks 5–8".
- Compact multi-level condition: "Complete-video ranking: ranks 1–4 (shared by
  A and B)" followed by "Cited-segment ranking: ranks 1–4 ({mm:ss–mm:ss})".

Which condition appears as A and which as B is randomized per rater and trial.

**Question:** Considering the concepts and any cited segment, which
explanation better helps you understand why the model made this prediction?

[Radio buttons:] Explanation A / Tie / Explanation B

**Question:** How confident are you in this choice?

[Radio buttons:] Low / Medium / High

[Checkbox, required:] I inspected the video and the cited time span.

[Button:] Submit and continue

## Validation and completion messages

- "Please answer both questions and confirm that you inspected the video."
  (shown if a submission is incomplete)
- "This code is not in the assignment file. Please check the code."
- "Thank you — your assigned trials are complete."

## Message sent to raters with their code

> Please set aside about 30–40 minutes, use a laptop/desktop with audio
> enabled, and enter the anonymous code I sent you. Some trials ask which
> explanation is more useful; others ask which concept list better matches a
> short clip. Watch the displayed video or clip before answering. You may
> choose Tie.
