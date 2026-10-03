# Changelog

## 0.5.0

A redesign that keeps the knowledge base (`tklearn.kb`) and rewrites the
metrics. Pin `textkit-learn<0.5` if you need the modules listed under
Removed.

### Removed

- `tklearn.nn`, `tklearn.embeddings`, `tklearn.plotting` and most of
  `tklearn.utils`. They are being redesigned.
- `tklearn.agents` (vendored smolagents). Use the `smolagents` package.
- nightjar configs, `Auto*` factories and lookup by name. Import the class
  you need and call its constructor.
- The v0 Wiktionary store (`"wiktionary:v0"`), the DuckDB `TripleStore` and
  `kb.wiktionary.helpers`.
- Dependencies: `nightjar`, `omegaconf`, `diskcache`, `openai`,
  `python-dotenv`.

### Metrics (`tklearn.metrics`)

- Metrics are updated batch by batch: `update(y_true, y_pred)` (or
  `y_score`), `compute()`, `reset()`, and `merge()` to combine metrics
  updated on different shards. Calling a metric computes it for one set of
  inputs. Inputs can be numpy arrays, lists or torch tensors.
- Parameters keep scikit-learn's names and meanings, and results match
  scikit-learn. Instead of `labels`, classification metrics take an
  optional `num_classes`; `average=None` returns one score per class index.
- Classification: `Accuracy`, `BalancedAccuracy`, `ConfusionMatrix`,
  `Precision`, `Recall`, `F1` and `FBeta`, for binary, multiclass and
  multilabel targets.
- Ranking: `AUROC` and `AveragePrecision` (binary, one-vs-rest multiclass
  and multilabel), `ROCCurve`, `PrecisionRecallCurve` and
  `OptimalThreshold` (`criterion="youden"` or `"f1"`), replacing
  `AUC`, `OptimalAUCThreshold` and `OptimalPRThreshold`. Scores are kept
  exactly, or binned with `thresholds=n` for fixed memory.
- Regression: `MeanSquaredError`, `RootMeanSquaredError`,
  `MeanAbsoluteError`, `R2Score`, `PearsonCorrelation` and
  `SpearmanCorrelation`.
- Spans: `SpanPrecision`, `SpanRecall` and `SpanF1` score spans read from
  BIO/IOBES tags, matching seqeval's default mode.
- `MetricCollection.update(**inputs)` passes each metric the inputs it
  accepts; `result()` is now `compute()`.

### Knowledge base (`tklearn.kb`)

- `KnowledgeBase("wiktionary")` is now `KnowledgeBase(WiktionaryStore())`.
  Store data is exposed as typed properties (`lexicon`, `triples`,
  `senses`, ...).
- `WiktionaryStore` (was `WiktionaryArtifactStore`) no longer creates a Hub
  repository or uploads on load. Use `offline=True` to skip the Hub and
  `push_to_hub()` to publish. Existing local caches keep working.
- `augment()` yields `Augmentation` objects instead of dicts, skips the
  unchanged text unless `include_original=True`, and takes `predicates`
  (default `("synonym",)`; the v1.0 data has `synonym`, `antonym`,
  `hypernym` and `category`).
- `extract_mentions(..., exact_form=False)` makes the lemma filter
  explicit and `stopwords` accepts a custom list. `extract_relations`
  returns all relations by default.
- `Lexicon.tokeinze` is renamed `tokenize`. `Lexicon.extract` takes
  `nested=True` and `Lexicon.load` takes `case_sensitive`.

### Other

- `tklearn.config` is a plain dataclass; `cache_dir`, `temp_dir` and
  `assets_dir` are `Path`s derived from `base_dir`.
- Repeated `get_logger(name)` calls no longer add duplicate handlers.
