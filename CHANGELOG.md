# Changelog

## 0.5.0

A redesign that keeps the knowledge base (`tklearn.kb`) and rewrites the
metrics and the training loop. Pin `textkit-learn<0.5` if you need the
modules listed under Removed.

### Removed

- `tklearn.embeddings`, `tklearn.plotting` and most of `tklearn.utils`.
  They are being redesigned.
- From `tklearn.nn`: backbones, the linear and prototype classifiers,
  K-BERT, `ModelRepo`, `Encoder`, `TargetBasedLoss` and the data
  collators.
  They are being redesigned. The `History` callback is gone (`fit`
  returns the logs of each epoch), and so is `TrackingCallback`
  (MLflow).
- `tklearn.agents` (vendored smolagents). Use the `smolagents` package.
- nightjar configs, `Auto*` factories and lookup by name. Import the class
  you need and call its constructor.
- The v0 Wiktionary store (`"wiktionary:v0"`), the DuckDB `TripleStore` and
  `kb.wiktionary.helpers`.
- From `tklearn.kb`: `Lexicon`, the in-memory `TripleStore`,
  `WiktionaryProcessor`, `WiktionaryStore.push_to_hub` and the pydantic
  models in `kb.wiktionary.models`. The store matches forms itself.
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
- Calibration: `LogLoss`, `BrierScore` and `ExpectedCalibrationError`
  (top-label, `n_bins` equal-width bins) read probabilities as `y_score`,
  for binary, multiclass and multilabel targets. The log loss and the
  binary Brier score match scikit-learn; the multiclass Brier score sums
  over the classes.
- `MetricCollection.update(**inputs)` passes each metric the inputs it
  accepts, and `input_names` lists them; `result()` is now `compute()`.

### Neural networks (`tklearn.nn`)

- One `Trainer` with `fit`, `evaluate` and `predict` replaces `Trainer`,
  `Evaluator` and `Predictor`. It runs on Hugging Face Accelerate: CPU,
  CUDA or MPS, mixed precision (`mixed_precision="bf16"`), gradient
  accumulation and multi-GPU with `accelerate launch`. It takes torch
  `DataLoader`s.
- A `Trainer` holds configuration only, read-only once created. It takes
  the optimizer as a factory, e.g.
  `functools.partial(torch.optim.AdamW, lr=2e-5)`. Each `fit`, `evaluate`
  or `predict` opens a new `Session`, which prepares the model, holds
  the call's progress, and is closed and dropped when the call returns,
  so one call does not affect the next.
- Callbacks, which keep state, are given to each call, as in Keras:
  `fit(train, valid, epochs=3, callbacks=[...])`, and likewise
  `evaluate`, `predict` and `calibrate`. The caller keeps them and reads
  their state afterwards, e.g. `ModelCheckpoint.best`.
- The trainer builds each session's optimizer, scheduler and
  `Accelerator` with `create_optimizer`, `create_scheduler` and
  `create_accelerator`. A subclass overrides them to build these
  otherwise, e.g. the optimizer with parameter groups or the accelerator
  with plugins.
- `Module` has two hooks. `predict_step(batch)` returns the metric inputs
  (`y_true`, `y_pred`, `y_score`, ...) and optionally a `"loss"`.
  `training_step(batch)` defaults to that loss. It replaces
  `compute_loss` and `compute_metric_inputs`.
- `training_step` can return `{"loss": ..., **terms}`: `"loss"` is
  minimized and the other terms are logged. This replaces `LossDict`.
  `on_train_batch_end` gets them as a read-only mapping that converts
  each value to a float when it is first read, so a batch whose logs no
  callback reads does not wait for the device.
- `Trainer(metrics=...)` feeds a `MetricCollection`, and
  `evaluate(loader, metrics=...)` uses other metrics for one call. Results
  from evaluation during `fit` are logged with a `valid_` prefix, and
  `fit` returns the logs of each epoch as a list of dicts.
- Callbacks read the session they are in as `self.session` (`model`,
  `optimizer`, `epoch`, `global_step`, `history`, ...;
  `self.session.trainer` is the configuration), and setting
  `self.session.should_stop = True` stops training, on every process
  when set on one. Hooks take no trainer (`on_epoch_end(logs)`).
  `set_model`, `set_trainer` and `set_params` are gone. Hooks keep their
  Keras names (`on_train_begin`, `on_test_end`, ...).
- A callback is in one session at a time, and evaluates within its
  session with `self.session.evaluate`. `Dispatcher` calls the hooks of
  a session's callbacks, also from a custom training loop.
- Built-in callbacks follow `keras.callbacks`: `EarlyStopping`,
  `ModelCheckpoint` (`.safetensors` or `.pt` state dicts),
  `ReduceLROnPlateau`, `TerminateOnNaN`, `CSVLogger`, `ProgbarLogger`,
  `LambdaCallback` and `BackupAndRestore`.
- `BackupAndRestore(backup_dir, save_freq=..., double_checkpoint=...,
  delete_checkpoint=...)` backs up the state of `fit` (model, optimizer,
  scheduler, gradient scaler, progress and history) every epoch or every
  few optimizer steps, and when the program runs again after an
  interruption, such as a requeued SLURM job, restores it and goes on
  from the start of the epoch it was taken in, as Keras does: a backup
  taken during an epoch trains that epoch again from its first batch.
  `Session.state_dict` and `load_state_dict` hold that state.
- `OpenTelemetryCallback` traces training with the OpenTelemetry API: a
  `fit` span holding an `epoch` span per epoch (with its logs and its
  `evaluate` span) and a `steps` span every few optimizer steps (mean
  loss, learning rate, gradient norm, step time); `evaluate` and
  `predict` outside `fit` get their own spans. The application chooses
  the backend by configuring an OpenTelemetry SDK. Its spans are current
  while they run, so other callbacks and the model can add events and
  child spans; `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`
  and `TerminateOnNaN` add events for what they do (`checkpoint`,
  `early_stopping`, `restore_best_weights`, `reduce_lr`,
  `terminate_on_nan`).
- `Callback.wrapper`: a wrapper callback's `*_begin` hooks run before,
  and its `*_end` hooks after, those of the other callbacks, so that what
  it opens encloses their work; wrappers nest in the order given. Other
  callbacks keep the order given. `OpenTelemetryCallback` is a wrapper.
- `Callback.on_exception(exception)` runs when `fit`,
  `evaluate` or `predict` raises, including on `KeyboardInterrupt`, in
  place of the `*_end` hooks of the unfinished session. `ProgbarLogger`,
  `CSVLogger` and `OpenTelemetryCallback` close their bars, file and
  spans there.
- `Session.grad_norm` holds the gradient norm before clipping.
- `predict` pads tensors and arrays whose dimensions after the first
  differ between batches or processes, such as per-token outputs of
  batches padded to different lengths, to the largest: signed integers
  with -100 and the rest with 0 (False). `evaluate` sums the loss on the
  device, so it no longer waits for it on every batch.
- `tklearn.nn.models`: `SequenceClassifier` (multiclass, binary,
  multilabel or regression) and `TokenClassifier` wrap a Hugging Face
  model, by name or loaded, with a `predict_step` that returns the
  logits, `y_pred`, `y_score` (`mask` for tokens), `y_true` and the loss,
  ready for `tklearn.metrics` and `calibrate`
  (`activation=model.activation`). With `label_names`, a
  `TokenClassifier`'s `y_true` and `y_pred` are tag sequences for
  `SpanF1`. They replace the backbone and classifier modules.
- `RunningMean` (`tklearn.nn.running_mean`) averages logged values over
  batches and every process, as `fit` does for each epoch; a callback can
  average the batch logs over a window the same way. It sums on the
  device, so only `compute` waits for it; on several processes, every
  process must call `compute`.
- `tklearn.nn.calibration.calibrate(trainer, loader, activation=...)`
  fits a temperature to the `"logits"` that `predict_step` returns on
  held-out data, and returns a `CalibratedModule` whose probabilities
  and loss use the scaled logits, with the same predictions.
  `activation` is `"softmax"` for a multiclass classifier or
  `"sigmoid"` for a binary or multilabel one; it is not inferred, and
  logits and labels of the wrong shapes for it raise. It replaces
  `calibrate_model`. `fit_temperature(logits, y_true, activation=...)`
  fits one from tensors, returned as a tensor. With `per_label=True`, a
  multilabel classifier gets one temperature per label, each pulled
  toward the shared one by `shrinkage`, more strongly the fewer
  examples the label's rarer class has, so that a label with no or few
  positives does not get a temperature near 0.
- `lr_scheduler` accepts a name with `warmup` (steps or a fraction), or
  `f(optimizer, num_training_steps)`, built at the start of each `fit`.
  `get_scheduler` builds named schedules. Schedules count optimizer
  steps.

### Tracing (`tklearn.tracing`)

- New. `FileTracerProvider(root_dir, name, resource)` is an OpenTelemetry
  tracing SDK that writes plain files, for shared storage without a
  database, such as NFS used by SLURM jobs on several machines. Each span
  is written when it starts and ends, and each event when it is added,
  so a killed job loses nothing and its open spans stay visible. Each
  provider writes its own file, and runs are named after the SLURM job,
  so a requeued job continues its run. Attribute values are not limited
  to OpenTelemetry's types. `load_spans` and `load_events` read every run
  under a directory into a DataFrame, with `resource.<key>` columns to
  compare runs. It does not import torch.
- New. `tklearn runs monitor DIR` shows the progress of every run under a
  directory, at any depth, live or `--once`: the status of its latest fit
  (running, stalled, finished, stopped early or failed), epoch, step,
  training loss, the latest epoch's metrics and an ETA. It reads only
  what the runs add, so it can follow jobs on other machines over NFS;
  `--slurm` asks `sacct` whether a quiet job is still alive.
  `tklearn.tracing.monitor.RunMonitor` does the same from Python.
- New. The `tklearn` command (also `python -m tklearn`).

### Knowledge base (`tklearn.kb`)

- `KnowledgeBase("wiktionary")` is now `KnowledgeBase(WiktionaryStore())`.
  A store is a `KnowledgeStore` with lookup methods (`find_forms`,
  `get_gloss`, `get_senses`, `get_relations`, `get_embedding`);
  `KnowledgeBase` no longer has `lexicon`, `triples`, `senses`, ...
- New. `TableStore` reads a store from Parquet tables, which
  `write_tables` writes.
- `extract_relations` also returns the word's relations that hold for no
  particular sense, and relations stated from the other word's side:
  synonyms and antonyms both ways, hypernyms and hyponyms as each other's
  inverse. A relation found more than once is returned once, with the
  most senses known.
- `WiktionaryStore` (was `WiktionaryArtifactStore`) reads the v0.5 tables
  from `config.assets_dir/wiktionary-v0.5` (or `path`) and downloads the
  missing ones from the Hub dataset `textkit-learn/wiktionary-v0.5`,
  once. It never builds the data or uploads it, and only contacts the Hub
  when files are missing. Gloss embeddings are downloaded only with
  `embeddings=True`. It takes `version`, `repo_id`, `path` and
  `embeddings`; `language`, `offline` and `verbose` are gone. Caches
  from earlier versions (pickles) aren't read.
- `augment()` yields `Augmentation` objects instead of dicts, skips the
  unchanged text unless `include_original=True`, and takes `predicates`
  (default `("synonym",)`; the v0.5 data has `synonym`, `antonym`,
  `hypernym`, `hyponym` and `category`).
- New. `tklearn.kb.wiktionary.build_wiktionary(dump, path)` builds the
  `WiktionaryStore` tables from the kaikki.org dump (`DUMP_URL`). Unlike
  the v1.0 data, the tables keep relations that hold for a word in
  general (no sense), including word-level categories, store hyponyms as
  the hypernyms they imply, and skip relations of a word to itself. The
  tables are about 165 MB (v1.0: 7.7 GB) and load in about 4 s with 4 GB
  of memory (v1.0: 28 s and 21 GB).
- `extract_mentions(..., exact_form=False)` makes the lemma filter
  explicit and `stopwords` accepts a custom list. `extract_relations`
  returns all relations by default.

### Other

- `tklearn.config` is a plain dataclass; `cache_dir`, `temp_dir` and
  `assets_dir` are `Path`s derived from `base_dir`.
- Repeated `get_logger(name)` calls no longer add duplicate handlers.
- New. `tklearn.utils.flatten.flatten` flattens nested values into one
  level of keys, e.g. for a table of runs. Nested mappings and named
  tuples become dotted keys (`optim.lr`, `roc.fpr`), lists and arrays of
  values are kept whole as tuples, lists of mappings become a tuple per
  key (`layers.*.size: (4, 8)`), and an empty mapping stays a read-only,
  hashable `{}`. Path objects become absolute paths, without resolving
  symlinks. `freeze` gives the value stored for a single value.
  `unflatten` rebuilds the nested values; `flatten` raises a
  `ValueError` for values it could not rebuild, e.g. when only None
  tells apart the forms that the items of a list hold a key in.
