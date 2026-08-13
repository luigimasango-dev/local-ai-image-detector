# Published train/test split lists

These two files list, by internal image ID, exactly which images were used
for fitting versus reporting in this submission. They are published so a
reviewer can mechanically verify that no evaluation image was trained on,
rather than having to take it on trust.

- `split_tune.txt` — the **599** images the Platt calibration was fitted on
  (the "TUNE" split).
- `split_holdout.txt` — the **601** images the model was *never* fitted on
  and from which **all reported numbers** in `RESULTS.md` are taken (the
  "HOLDOUT" split).

The two files are generated from `eval_harness/data/manifest.json` (the
1,200-image AIDetectArena v0.1 sample) by `eval_harness/split_eval_set.py`,
which stratifies by (label, generator) at a fixed seed (42) and 50/50 split.
The same files exist inside the gitignored `eval_harness/data/` directory;
these copies are the canonical published form.

Each ID resolves to exactly one image in the dataset, e.g. an ID
`ai_animal_flux_2_flex_animal_01.png` identifies the AI sample for generator
`flux_2_flex` used at that position. There is no overlap between the two
lists — the 1,200 images partition cleanly into 599 TUNE + 601 HOLDOUT.

To check overlap with any evaluation or test set you may have, compare the
set of IDs in these files against it directly.

Why this matters: the submitted model's parameters are the two scalars of a
Platt calibration (A = 0.6758, B = 2.0932) fitted on `split_tune.txt` only,
on top of 43.4 M *frozen* model parameters. The reported balanced accuracy
(0.8338 on pristine images, 0.8321 in the shipped fp16 form, both at the
0.65 threshold) is measured exclusively on `split_holdout.txt`.