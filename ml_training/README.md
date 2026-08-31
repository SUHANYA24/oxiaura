# ML training (BUILD_SPEC Phase 6b)

Everything needed to replace the mocked fraud detectors with real weights, and
to measure how well OCR actually reads your forms. Nothing in here runs during a
request — these are offline scripts you run by hand, and the app only ever loads
their **output** files.

```
app/ai/training/          <- your raw scans (you provide these; git-ignored)
  nic/                       nic-001-back.jpeg ... nic-038-front.jpeg
  proposal/                  proposal-001-back.jpg ... proposal-135-front.jpg
  proposal/_rejected/        scans ingest_raw set aside (duplicates), not deleted
ml_training/
  datasets/ingest_plan.csv <- generated: the rename plan, yours to review
  datasets/fraud/          <- generated: genuine/, tampered/, manifest.csv
  datasets/fraud/reference/  <- you provide: confirmed forgeries (optional)
  datasets/ocr/labels/     <- generated stubs, YOU type in the values
  templates/               <- field-box JSON + calibration overlay PNGs
  saved_models/            <- generated: .pt weights + .metrics.json
```

Run every command from the repository root, with the virtualenv active:

```powershell
python -m ml_training.<script> --help
```

---

## 1. The datasets you have to create by hand

Two of the three datasets cannot be generated. This is the manual work.

### 1a. Raw scans → `app/ai/training/<kind>/`

Drop image files into `app/ai/training/nic/` and `app/ai/training/proposal/`,
under whatever names your camera or phone gave them, then run `ingest_raw`
(§1a-i) to sort and rename them.

| | you have now | usable | good |
|---|---|---|---|
| distinct source scans | 173 | 8+ | 30+ |

38 NIC scans (19 front, 19 back) and 135 proposal pages (67 front, 68 back).
Each *scan* is a source for the fraud trainers, which is what that row counts.
The number of distinct *physical* documents is lower and not known — `ingest_raw`
deliberately does not try to pair a front with its back, because nothing in the
filenames or the images gives a reliable pairing signal.

**Everything downstream is limited by this number.** The fraud trainers
synthesize hundreds of variants from these files, but variants of one scan are
near-duplicates — they teach the model what *that* document looks like, not what
tampering looks like in general. Each new *physically different* document is
worth more than a hundred more augmentations.

Two rules the scripts cannot fix for you:

* One document per file. Do not paste two pages into one image.
* Keep the original camera/scanner JPEG. **Do not re-save, crop, or "clean up"
  a scan in an image editor** — re-encoding destroys exactly the compression
  artifacts the ELA and CNN detectors look for, and a re-saved genuine document
  will look tampered. `ingest_raw` detects and rejects re-encoded copies for
  this reason.

### 1a-i. Sort and rename a fresh drop: `ingest_raw`

```powershell
python -m ml_training.ingest_raw            # classify + write the plan (no writes to scans)
```

Downstream scripts read the *side* of a document out of its filename
(`label_ocr.detect_side`, `evaluate_ocr`'s template lookup), and a folder of
`WhatsApp Image 2026-05-28 at 10.46.27 AM (1).jpeg` and bare UUIDs tells them
nothing. `ingest_raw` classifies each scan and renames it to

```
<kind>-<NNN>-<side>.<original extension>      e.g. proposal-117-back.jpg
```

Numbers are assigned in sorted order of the *original* names and existing
numbers are reserved, so adding scans later never renumbers what is already
there. The rename is a `rename` — the JPEG bytes are never re-encoded.

How the side is decided, and how well:

| kind | signal | measured |
|---|---|---|
| proposal | mass of interior vertical rules (the two-column witness table only exists on the back) | fronts 0.000 (all 68), backs 0.170–0.844 — no overlap |
| nic | Haar face detection at all four quarter-turns, plus peak local ink fraction (the PDF417 block) | 37/38 correct, and the one error was inside the flagged set |

It is a **two-step, review-then-apply** tool. The first run only writes
`datasets/ingest_plan.csv`; nothing on disk moves. Open it, look at any row
marked `low` confidence, and fix the `side` cell if it is wrong — the proposed
filename is regenerated from that cell, so a misclassification is a one-cell
edit rather than a code change. Then:

```powershell
python -m ml_training.ingest_raw --apply     # replay the plan, rename, record it
python -m ml_training.ingest_raw --undo      # put every recorded original name back
```

`--apply` also rewrites the scan paths recorded inside `templates/*.json` and
`datasets/ocr/labels/*.json`, and renames the label files to match, so a rename
does not silently break an already-calibrated template or an already-typed
label file.

If you hand-edit the plan, **any cell containing a comma must be wrapped in
double quotes.** The plan is read back with strict validation that refuses a
malformed row before anything moves; an earlier version discovered the problem
only after renaming every file, which cost that run's undo manifest.

Two other things it reports rather than decides:

* **Duplicates.** Near-identical scans are grouped, and a scan that is a
  *re-encode* of another (same pixels, fewer bytes) is moved to
  `<kind>/_rejected/` instead of being renamed. `config.iter_raw_images` only
  yields files, so a `_rejected/` subfolder leaves the dataset without being
  deleted. Detection is dHash-based and best-effort: a second photo of the same
  card at a very different angle is not caught. The `group` column is
  **metadata only** — `prepare_dataset` splits by source *image*, not by group,
  so several scans of one physical document can still straddle the train/val
  split.
* **Orientation.** Landscape scans are flagged. Rotation is deliberately not
  baked into the filename or the file: see §6, *Sideways scans*.

### 1b. OCR ground truth → `ml_training/datasets/ocr/labels/*.json`

Scaffold one stub per scan, pre-filled with the right field keys:

```powershell
python -m ml_training.label_ocr
```

Then open each JSON and type what you can read on the scan:

```json
{
  "image": "app/ai/training/nic/nic-002-front.jpeg",
  "doc_kind": "nic",
  "side": "front",
  "fields": {
    "nic_number": "200209801097",
    "name": "H P K L PERERA",
    "sex": "MALE",
    "date_of_birth": "2002-04-07"
  }
}
```

* Type the value **exactly as printed**, including case and punctuation. The
  scorer also reports a normalized match, so `2002-04-07` and `2002 04 07` are
  not counted as a failure — but the exact-match column only means something if
  you were literal.
* **Leave a field empty if it is blank on the document.** Empty fields are
  skipped, not counted as misses. Guessing a value inflates the error rate.
* Re-running `label_ocr` never overwrites a file you have edited. Only
  `--force` does that, and it discards your typing.

Check progress at any time:

```powershell
python -m ml_training.label_ocr --status
```

**Do not try to fill in all 173.** `evaluate_ocr` loads only the label files with
at least one value typed in and skips the rest, so a stub you have not touched
costs nothing — no EasyOCR pass, no effect on the score. Ten to twenty labelled
documents per `(kind, side)` is enough to see which fields the extractor gets
wrong; typing 2,600 fields is not the point of the exercise.

Two NIC stubs (`nic__nic-001-back.json`, `nic__nic-002-front.json`) are
**already filled in** as a worked example, so the harness runs out of the box.

### 1c. Known-forgery exemplars → `ml_training/datasets/fraud/reference/` (optional)

Only needed for the Siamese bank (step 5). One image per confirmed forgery,
recycled template, or document you have already rejected. Skip this until you
have real rejected documents — synthetic ones teach the bank nothing.

---

## 2. Fraud dataset: synthesize both classes

```powershell
python -m ml_training.prepare_dataset
```

For each source scan this writes `AUG_PER_IMAGE` genuine variants (capture-style
jitter: rotation, perspective, brightness, gamma, noise) and `TAMPER_PER_IMAGE`
tampered variants using four forgery operations:

| op | what it does |
|---|---|
| `copymove` | duplicates a text region elsewhere on the same page |
| `splice` | pastes a region from a *different* source document |
| `textpatch` | inpaints real ink away and writes different digits over it |
| `requant` | re-encodes one region at low JPEG quality (a classic edit trace) |

The same capture-style jitter is applied to **both** classes on purpose. If only
the tampered images were rotated and noised, the classifier would learn to
detect rotation and report 99% accuracy while detecting nothing.

Output: `datasets/fraud/{genuine,tampered}/` plus `manifest.csv` with columns
`path,label,source,doc_kind,op,split`.

**The split is grouped by source image**, so all variants of one scan land
wholly in train or wholly in validation. Splitting by variant would put
near-duplicates on both sides and report accuracy that does not exist.

Useful flags: `--aug-per-image`, `--tamper-per-image`, `--val-fraction`,
`--max-long-edge`, `--clean` (wipe and regenerate), `--seed`.

---

## 3. Train the CNN tamper classifier

```powershell
python -m ml_training.train_cnn
```

ResNet-18 (ImageNet-initialized) with a single-logit head, `BCEWithLogitsLoss`
weighted by class imbalance, early stopping on validation ROC AUC.

Writes:

* `saved_models/cnn_tamper.pt` — a **bare `state_dict`**, which is what
  `cnn_classifier._load_model` loads with `strict=True`. Do not replace this
  with a checkpoint dict; metrics go to the sibling JSON instead.
* `saved_models/cnn_tamper.metrics.json` — per-epoch history, best AUC, and the
  threshold that separated the classes best.

Point the app at it (the script prints this line for you):

```dotenv
FRAUD_CNN_WEIGHTS=ml_training/saved_models/cnn_tamper.pt
```

Restart the app. `cnn_classifier.predict_tamper_probability` now runs the real
model instead of its SHA-256 mock. Nothing else changes — the score still feeds
the same `0.4 · CNN · 100` term of the aggregate.

**Measured on the 173-scan drop** (2076 images, 1560 train / 516 val, 15 epochs,
CPU, defaults otherwise):

| | best epoch (11) | last epoch (15) |
|---|---|---|
| validation ROC AUC | **0.830** | 0.793 |
| validation accuracy | 0.781 | 0.736 |
| validation loss | 0.454 | 0.525 |
| train loss | 0.437 | 0.304 |

`cnn_tamper.pt` holds epoch 11, and `final_val` in the metrics JSON describes
that epoch rather than the last one. Two things the chart in §4a makes obvious,
both worth knowing before trusting this model:

* **It overfits after epoch 11.** Train loss keeps falling to 0.304 while
  validation loss climbs to 0.525 — a +0.22 gap. With 12 variants generated from
  every one source scan there is a great deal for a ResNet to memorize. More
  epochs will not help; more *source documents* will.
* **The best threshold is unstable.** Across the 15 epochs it lands anywhere in
  0.24–0.65, so the 0.426 reported at epoch 11 is one draw from that spread and
  not a property of the model. AUC 0.830 is the honest headline precisely
  because it does not depend on choosing a cut-off.

For the same reason, treat the printed cut-off as information rather than a
setting to copy: `_CNN_CONCERN = 0.5` in `app/services/fraud_service.py` is
inside the measured spread and is a defensible place to leave it. It only
composes the human-readable flag reason, but it should reflect your model — so if
a *stable* threshold does emerge on a larger drop, move it then.

Flags: `--epochs`, `--batch-size`, `--lr`, `--patience`, `--freeze-backbone`
(much faster, weaker), `--no-pretrained` (if the one-time ~45 MB ImageNet
download fails), `--device`, `--workers` (leave at 0 on Windows).

---

## 4. Train the Siamese embedder

```powershell
python -m ml_training.train_siamese
```

Metric learning with `CosineEmbeddingLoss`. "Similar" here means **the same
physical document or template**, not "tampered" — because
`highest_similarity` asks *does this upload look like a document I have already
rejected?* Positive pairs are two variants of one source (tampered ones
included, so a forged copy still matches its own template); negative pairs come
from different sources.

Writes `saved_models/siamese_embedder.pt` — the ResNet-18 backbone `state_dict`
with `fc = Identity`, matching `siamese_detector._load_embedder`. There is
deliberately **no projection head**: the loss shapes the exact 512-d normalized
vector the detector serves at request time.

```dotenv
FRAUD_SIAMESE_WEIGHTS=ml_training/saved_models/siamese_embedder.pt
```

`siamese_embedder.metrics.json` reports mean positive/negative similarity and
the best separating threshold — the evidence for setting `_SIAMESE_CONCERN`.

**Measured on the 173-scan drop** (1560 train rows / 516 val rows, 130 train
source scans, 128 validation pairs per epoch): early stop at epoch 8 of 20, best
and saved epoch 2, pair ROC AUC **0.999**, mean same-document similarity 0.992
against mean different-document 0.495 — a separation of roughly 0.50, which holds
or widens across every epoch.

Read that 0.999 carefully. The question being scored is *"are these two images
the same document?"*, and a positive pair is two augmented crops of one
photograph — so paper texture, lighting and JPEG noise are all legitimate
evidence for it. Near-perfect is the *expected* result for that question, and it
is the right question for `highest_similarity`, which asks whether an upload
matches something already rejected. It says nothing at all about whether the
model can spot a forgery; that is the CNN's job, and the CNN scores 0.830.
Convergence by epoch 2 tells the same story: ImageNet features already separate
two different photographs, so there was little left to learn.

**This changed `_SIAMESE_CONCERN`.** The best separating cut-off measures
0.964–0.997 across epochs, because same-document pairs cluster just under 1.0.
`app/services/fraud_service.py` used to set `_SIAMESE_CONCERN = 0.7`, chosen
against the deterministic mock; against the trained embedder that lands in the
empty gap between the two populations, calling a 0.7 similarity "similar to a
known document" when a real match measures ~0.99. It is now **0.95** — see §5,
which measures it against a real bank and adds the nuance that 0.495 is a
*cross-kind* average.

Note that `final_val` here describes the *saved* epoch, not the last one: the
script reloads `best_state` before its closing evaluation. It did not always —
an earlier version measured whichever weights the last epoch left in memory, so
the summary line paired "Best epoch 2" with epoch 8's numbers. With early
stopping the two are always at least `--patience` epochs apart.

---

## 4a. Chart what the trainers measured: `plot_metrics`

```powershell
python -m ml_training.plot_metrics
```

Draws three PNGs into `saved_models/` out of files that already exist — the two
`.metrics.json` and `datasets/fraud/manifest.csv`. It never loads a `.pt`, never
runs a model and never opens a scan, so it is cheap to re-run and safe to run
while a training job is still going.

| chart | file | what to look at |
|---|---|---|
| CNN | `cnn_tamper.curves.png` | the gap between the two loss curves; whether AUC stays clear of the 0.5 coin-flip line; how far the cut-off wanders |
| Siamese | `siamese_embedder.curves.png` | the shaded band between same-document and different-document similarity — loss falling while that band stays flat means nothing useful was learned |
| dataset | `dataset_composition.png` | class balance inside *each* split, and variants vs distinct sources per kind (log scale, since they differ by 12×) |

A training curve is a diagnosis, not a grade. The panels are arranged so the two
failure modes this dataset invites are visible without reading any numbers: a
widening train/val gap on the CNN (memorizing variants of documents it has
already seen), and a flat separation band on the Siamese.

Needs `matplotlib`, pinned in `requirements.txt` under a comment saying it is for
this script alone — nothing under `app/` imports it, and a production install can
leave it out. If it is missing, the script prints the install line instead of a
traceback. The backend is forced to `Agg` before pyplot loads, matching the
headless `opencv-python-headless` pin, so no window is ever opened.

Flags: `--which {all,cnn,siamese,dataset}` (anything with no data yet is skipped,
naming the command that would produce it), `--out-dir`, `--dpi` (default 150).

The charts carry aggregate numbers only — no document imagery, no field values —
so unlike everything else under `datasets/` and `app/ai/training/` they are safe
to paste into a report or a slide.

---

## 5. Build the reference bank

```powershell
python -m ml_training.build_reference_bank
```

Embeds every image in `datasets/fraud/reference/` into
`saved_models/reference_embeddings.pt` (a `[N, 512]` tensor) with provenance in
the sibling JSON, including a warning about near-duplicate exemplars.

### Wiring the Siamese bank into the app — done

This used to be a list of steps to do by hand. It is now in the code:

| where | what |
|---|---|
| `app/config.py` | `FRAUD_REFERENCE_BANK` reads the env var, next to the two weight paths |
| `app/services/fraud_service.py` | `_reference_embeddings()` loads the bank once per path per process and `run_detectors` passes it to `highest_similarity` |
| `.env` | set `FRAUD_REFERENCE_BANK=ml_training/saved_models/reference_embeddings.pt` and restart |

The cache is keyed by path rather than held in one global, so the test suite's
several app instances cannot inherit each other's bank. A missing or unreadable
bank logs and returns `None` instead of failing the upload — the detector then
takes its mock path, exactly as before.

Two things this exposed, both fixed:

* `highest_similarity` guarded its real path with `and known_embeddings:`, and
  `bool()` on a `[N, 512]` tensor raises *"Boolean value of Tensor with more than
  one value is ambiguous"*. The bank this script writes **is** such a tensor, so
  the wiring above would have crashed every upload the moment it was configured.
  Now guarded with `len(known_embeddings) > 0`, which works for a tensor and a
  list alike.
* `_SIAMESE_CONCERN` was `0.7`, chosen against mock scores. Measured against the
  trained embedder it is now `0.95`; see the numbers below.

**Measured with a real 2-exemplar bank** (`nic-001-back`, then scoring other val
images through `run_detectors`):

| image | similarity |
|---|---|
| a different variant of the banked document | 0.980 |
| genuine images of a *different* NIC card | 0.692 – 0.724 |

So the useful cut-off is around 0.9, and `0.95` sits clear of both populations.
Note this refines §4's figure: the 0.495 mean different-document similarity is an
average across *kinds*, and a proposal page is nothing like a NIC card. Two
different documents **of the same kind** sit near 0.70, because they share a
layout. Set the threshold against that number, not the cross-kind mean.

Those numbers came from a throwaway bank built for the test and then deleted.
`datasets/fraud/reference/` still holds **0 exemplars**, so `FRAUD_REFERENCE_BANK`
is deliberately unset in `.env` and the Siamese term is a mock in production
today — which is why `evaluate_pipeline` (§8) reports it at AUC 0.474.

Resist the temptation to fill the bank from the synthesized tampered variants to
get it working. The 0.980 row above is the reason: a variant of a source document
matches *any* image of that same document, genuine or not. A bank seeded from
`datasets/fraud/tampered/` would flag every honest re-upload of the scans you
trained on. The bank only means anything when its contents are documents you
actually rejected.

---

## 6. OCR: calibrate a template

The proposal form is a ruled table, which the regex parser in
`app/ai/ocr/extractor.py` handles badly — labels and handwritten values sit in
separate cells, so line-oriented patterns have nothing to match.
`app/ai/ocr/template_extractor.py` instead records where each field *is*, aligns
each new scan to the reference by ORB homography, and OCRs one box at a time.

The project pins `opencv-python-headless`, so there is no window to drag boxes
in. Calibration is: generate an overlay PNG, look at it, edit the JSON.

**Four templates are already calibrated against your scans**, so unless you swap
the forms out you can skip straight to §7:

| template | boxes | reference scan | notes |
|---|---|---|---|
| `proposal_front.json` | 16 | `proposal-020-front.jpg` | customer + nominee blocks |
| `proposal_back.json` | 17 | `proposal-117-back.jpg` | product, bank, employee, supervisor |
| `nic_front.json` | 4 | `nic-002-front.jpeg` | `rotate: 90` |
| `nic_back.json` | 3 | `nic-001-back.jpeg` | `rotate: 90` |

`proposal_front.json` carries two boxes — `agreement_no` and `proposal_no` —
that have no matching key in `label_ocr.FIELD_SETS`. Matching is by name, so
those two are extracted and then never scored. Add the keys to `FIELD_SETS` if
you want them measured; leaving them is harmless.

### What one NIC template can and cannot cover

The 38 NIC scans span at least three card generations with different field
layouts, photographed at arbitrary in-plane rotation, perspective and scale —
some fill the frame, some sit small in the middle of a desk. `nic_front.json` and
`nic_back.json` were calibrated on the current design filling the frame, and ORB
homography can absorb rotation and perspective but **not a different layout**.
So expect them to align on cards of the same generation and to fall back (or
land on the wrong region) on the older ones.

That is a limit of the template approach, not a defect in the scans. The NIC
set's real value is the fraud dataset — 38 distinct sources, up from 2 — where
layout variation is an asset rather than a problem. If you need reliable NIC
field extraction across generations, calibrate one template per generation and
select between them, rather than widening one template until it fits none.

Front and back are **different layouts**, so each gets its own file named
`<kind>_<side>.json`. That is exactly what `evaluate_ocr.py` looks for; a single
`<kind>.json` is accepted as a fallback for one-page documents.

```powershell
# Confirm an existing template still lands where you think it does
python -m ml_training.calibrate_template --doc-kind proposal --side back --preview

# Proposal form: auto-detect the ruled cells as a starting point for a new layout
python -m ml_training.calibrate_template --doc-kind proposal --side front --auto --preview

# NIC card: no ruled table, so read coordinates off a labelled 5% grid
python -m ml_training.calibrate_template --doc-kind nic --side front --rotate 90 --grid
```

After `--auto`, edit the JSON: rename the boxes you want from `field_r03c02` to
real field names, delete the rest, and re-run with `--preview` to confirm the
boxes landed where you meant.

**Name the boxes exactly like the keys in your label files** (`customer_nic`,
`nominee_birthday`, …). `evaluate_ocr.py` matches template fields to labels by
name.

Boxes are `[x, y, width, height]` as fractions of the page, so a template stays
valid across scan resolutions.

### Sideways scans: `rotate`

ID cards are landscape but get photographed inside a portrait frame, so the text
reads bottom-to-top. `--rotate {0,90,180,270}` (counter-clockwise degrees) turns
the scan upright *before* the grid is drawn, so you read coordinates and write
boxes the way you read the document. The value is stored in the template and
re-applied at extraction time; omit the flag later and the calibrator picks it up
from the template, so `--preview` always renders in the frame the boxes were
written in.

This is why rotation lives in the template and **not** in a re-saved image file:
the fraud detectors read JPEG compression artifacts straight from the original
bytes, and re-encoding a scan to straighten it destroys the very evidence they
look for. `cv2.rotate` on a quarter turn is lossless and happens in memory only.

One consequence of the trilingual NIC layout: `nic_front.json`'s `sex` box
deliberately covers only the English word (`Male`), not the whole
`පුරුෂ / ஆண் / Male` row, because a box over the row would feed Sinhala and Tamil
glyphs to an English OCR model. It is tight on purpose — including the `/`
separator made EasyOCR read it as a digit and produce `1 Male`. Several NIC boxes
also carry `"pad": 0.0` for the same reason; the default 0.4% padding is helpful
on a spacious form and harmful on a card where fields sit millimetres apart.

Because the trilingual string is printed as one run, a longer value (`Female`)
shifts the English segment. If `sex` starts missing, widen that box and re-check
with `--preview`.

---

## 7. Score the extraction

```powershell
python -m ml_training.evaluate_ocr
```

Runs both strategies over every labelled scan and prints a per-field table:
found rate, exact match, normalized match, and character error rate. The full
breakdown — including what was extracted versus expected for every field — goes
to `datasets/ocr/report.json`.

Measured on the two NIC scans (7 fields, labels already filled in):

| | found | exact | normalized | mean CER |
|---|---|---|---|---|
| regex | 0.29 | 0.14 | 0.14 | 0.75 |
| template | **1.00** | 0.57 | **0.71** | **0.04** |

The template finds every field; the three that are not exact are genuine
character misreads on a worn card (`06`→`08`, `NEGOMBO`→`NECOMBO`, a dropped
comma), not misplaced boxes. That distinction is the point of the harness.

Read it as a diagnosis, not a grade:

* **found low** — the box is in the wrong place, or the regex has no pattern for
  that field. Check the `--preview` overlay first.
* **found high, exact low, CER low (< ~0.2)** — OCR is reading the right region
  and nearly getting it. Worth normalizing in post-processing.
* **CER near 1.0** — it is reading something else entirely. Usually a box that
  covers the printed label instead of the value.
* **`alignment: FALLBACK`** in the output means the homography failed and the
  scan was merely resized, so every box is approximate. Usually a badly cropped
  or very low-contrast scan.

`--mode regex` / `--mode template` / `--mode both` (default), `--limit N` while
iterating — this loads EasyOCR and runs it per field, so it is slow on CPU.

`template_extractor` is a **library only**: `ocr_service` still uses the regex
path. Switching it over is a runtime behaviour change, so make it deliberately,
once `evaluate_ocr` shows the template is actually better on your documents.

---

## 8. Score what the API actually returns: `evaluate_pipeline`

```powershell
python -m ml_training.evaluate_pipeline              # balanced sample of 200
python -m ml_training.evaluate_pipeline --limit 0    # every val image
```

The trainers tell you how good each *model* is. This tells you how good the
**verdict** is, which is a different number. It calls
`app.services.fraud_service.run_detectors` — the same function `POST /documents`
reaches through `ocr_service` — inside a real app context, so every weight path,
mock fallback and aggregate weight that applies to a live upload applies here
too. Scored on the held-out `val` split, which is grouped by source scan, so
nothing here contributed a variant to CNN training.

Why the distinction matters: the verdict is a blend,

```
aggregate = 0.3*ELA + 0.4*CNN*100 + 0.3*Siamese*100
```

and a detector on its mock still contributes a number to that sum — stable per
file, but unrelated to tampering. The script AUCs each term separately, so a mock
shows up at ~0.5 and you can see what it costs. It also computes the aggregate
with the Siamese term dropped and the rest renormalized, which answers the real
question: *what if the unavailable detector simply did not vote, instead of
voting at random?*

**Measured on all 516 val images** (258 genuine, 258 tampered), with ELA and CNN
real and Siamese on its mock:

| signal | genuine | tampered | AUC |
|---|---|---|---|
| ELA (0-100) | 11.782 | 11.478 | 0.485 |
| CNN P(tampered) | 0.259 | 0.682 | **0.830** |
| Siamese similarity | 0.509 | 0.484 | 0.474 |
| AGGREGATE (0-100) | 29.140 | 45.236 | 0.773 |
| aggregate, no Siamese | 19.826 | 43.893 | **0.834** |

Three findings worth acting on, all of them policy calls rather than bugs:

* **The blend is diluting the one detector that works.** ELA measures 0.485 —
  slightly *backwards*, i.e. no signal on these documents — and the Siamese mock
  measures 0.474. Together they hold 60% of the weight, and the aggregate (0.773)
  therefore scores **worse than the CNN alone** (0.830). Either populate the
  reference bank (§5) so the Siamese term earns its 0.3, or re-weight toward the
  CNN. BUILD_SPEC gives the 0.3/0.4/0.3 split as an "e.g.", so re-weighting is
  permitted — but `tests/test_fraud_routes.py` asserts exact aggregates, so it is
  a deliberate change, not a tweak.
* **`FRAUD_FLAG_THRESHOLD=60` is calibrated for the mocks, not for the trained
  CNN.** At 60: accuracy 0.620, precision 0.984, recall **0.244** — 63 of 258
  tampered caught, 195 missed, 1 genuine false-flagged. The mocks used to spread
  scores uniformly around 50, which made 60 look reasonable; the real CNN pushed
  the whole distribution down. Best cut-off on this sample is **42.63** (accuracy
  0.769), or 32.78 without the Siamese term (0.787). Where to land between "few
  false alarms" and "catches most forgeries" is a business decision.
* **Re-quantization is invisible to this pipeline.** Recall at 60 by operation:
  splice 35%, textpatch 30%, copymove 23%, requant **0%**. A JPEG re-save leaves
  no trace either detector currently looks for. By kind, `nic` (AUC 0.688) is
  harder than `proposal` (0.785) — there are far fewer NIC source scans.

Per-image detail lands in `datasets/pipeline_report.json`, including each term
and both aggregates, so you can re-threshold offline without re-scoring.

`--limit N` takes a label-balanced sample (0 = everything), `--seed` picks which
one, `--config production` scores against production settings. It reads the
dataset and config and writes one JSON file; it changes nothing.

---

## Privacy

`app/ai/training/` holds real identity documents — NIC numbers, full names,
dates of birth, addresses, signatures. It is git-ignored along with
`datasets/` and `saved_models/`, so scans and their generated variants stay
local. Check with `git status --ignored` before committing if you have added
folders of your own, and never attach these files to an issue or a PR.

Note also that trained weights are derived from this data. Treat
`saved_models/*.pt` as confidential — a classifier trained on a few dozen
documents can memorize them — and the same goes for
`datasets/ingest_plan.csv`, which lists every scan by filename.

---

## Full sequence, from nothing

```powershell
# once: drop scans into app/ai/training/{nic,proposal}/ under any filenames

python -m ml_training.ingest_raw             # writes the plan; nothing moves yet
#   -> review datasets/ingest_plan.csv, fix any 'low' confidence side cell
python -m ml_training.ingest_raw --apply

python -m ml_training.prepare_dataset
python -m ml_training.train_cnn
python -m ml_training.train_siamese
#   -> paste the two printed FRAUD_*_WEIGHTS lines into .env, restart the app
python -m ml_training.plot_metrics            # three PNGs in saved_models/
python -m ml_training.evaluate_pipeline       # what the API now returns

python -m ml_training.label_ocr              # then fill some JSON files in by hand
python -m ml_training.label_ocr --status
python -m ml_training.evaluate_ocr
```

The templates for your current forms are already committed, so nothing between
`label_ocr` and `evaluate_ocr` is needed unless the form layout changes — at
which point re-run `calibrate_template` for the affected `--doc-kind`/`--side`.
