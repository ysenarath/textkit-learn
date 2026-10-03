# Changelog

## 0.5.0

A redesign of the public interfaces. This release is not backward compatible
with 0.4; the sections below list how to migrate.

### Removed

- `tklearn.agents` (vendored smolagents). Use the `smolagents` package.
- nightjar configs and `Auto*` factories (`AutoModel`, `AutoBackbone`,
  `AutoEmbedding`, `AutoArtifactStore`, `*Config` classes) and lookup by
  name. Import the class you need and call its constructor.
- The v0 Wiktionary store (`"wiktionary:v0"`, `WiktionaryArtifactStoreV1`) and
  the DuckDB `TripleStore`.
- `PlateauEarlyStopping` (its plateau check was never called).
- `MetricBase`, `MetricState`, `ArrayAccum` and `StepsCounter`.
- Compatibility aliases `TransformersEmbedding`, `AutoKnowledgeBaseModel` and
  `KnowledgeBaseTokenizer`; `tklearn.utils.constants`, `tklearn.utils.trie`,
  `tklearn.utils.datasets.load_dataset` and `kb.wiktionary.helpers`.
- Dependencies: `nightjar`, `omegaconf`, `diskcache`, `openai`,
  `python-dotenv`.

### Training loop (`tklearn.nn`)

Data is passed to the method that uses it instead of the constructor:

```python
trainer = Trainer(
    model,
    optimizer,
    lr_scheduler="linear",
    evaluator=Evaluator(model, metrics={"f1": F1(average="macro")}),
    callbacks=[EarlyStopping(monitor="valid_f1", patience=2)],
)
history = trainer.fit(train_loader, epochs=10, eval_dataloader=valid_loader)
results = Evaluator(model, metrics={"acc": Accuracy()}).evaluate(test_loader)
logits = Predictor(model).predict(test_loader, output_key="logits")
vectors = Encoder(model).encode(loader, return_tensors="np")
```

- `Trainer(model, dataloader, optimizer, epochs=...).train()` is now
  `Trainer(model, optimizer, ...).fit(dataloader, epochs=...)`. Evaluation
  results are logged with `eval_prefix` (default `"valid_"`).
  `clip_grad_norm` takes a float.
- `Evaluator.evaluate(dataloader, prefix="")` always returns a flat dict of
  mean losses and metric values. The `metrics="validate"/"test"` modes and
  `validate()`/`test()` are gone. `postprocessor` is now
  `compute_metric_inputs`. Loss errors propagate; pass `include_loss=False`
  for models without a loss.
- `Predictor.predict(dataloader, output_key=None)` concatenates outputs on
  the CPU and keeps their type; it no longer assumes a `"logits"` key.
- `Encoder(model, output_key="pooler_output").encode(dataloader, ...)`;
  `return_tensors=None` now requires `return_list=True`.
- `Module` hooks take only the batch: `predict_step(batch)`,
  `compute_loss(batch, output)`, `compute_metric_inputs(batch, output)`.
  `training_step(batch)` defaults to `compute_loss(batch, predict_step(batch))`.
  `validation_step`/`test_step` are removed.
- Callbacks: a callback stops training with
  `self.trainer.stop_training = True` (previously `model.stop_training`).
  Exceptions raised in callbacks now propagate instead of becoming warnings.
  `Evaluator` fires the `on_test_*` hooks and `Predictor`/`Encoder` the
  `on_predict_*` hooks; `ProgbarLogger` shows bars for both. `History` is
  exported and returned by `fit` (it is no longer set on the model).
  `ModelCheckpoint` monitors `"valid_loss"` by default and accepts only
  `.pt` and `.safetensors` paths.
- `LossLike` and `LossFunction` types are exported from `tklearn.nn.loss`.

### Metrics (`tklearn.metrics`)

- Metrics are plain objects: they declare `inputs` and implement
  `compute(**arrays)`. `MetricCollection({"f1": F1(), ...})` accumulates
  batches with `update(**inputs)` and returns a dict from `result()`.
  A metric can also be called directly: `F1()(y_true=..., y_pred=...)`.
- Missing inputs raise a `KeyError` naming them. Scalar results are Python
  floats.
- The curve points of `OptimalAUCThreshold`/`OptimalPRThreshold` use the key
  `threshold` (was `thresholds`).

### Models (`tklearn.nn.models`)

- `LinearMulticlassClassifier(backbone, num_labels, dropout=0.2)` and
  `PrototypeMulticlassClassifier(backbone)`. `backbone` is a `Backbone` or a
  Hugging Face model id.
- Backbones are `TransformerBackbone(model_name_or_path)` and
  `AdapterBackbone(model_name_or_path, adapter)`. The base class `Model` is
  renamed `BackboneModel`.
- `PrototypeCallback` and the `SequenceClassifierOutput*` classes are
  exported from `tklearn.nn.models`.
- `KBertTokenizer(model_name_or_path, knowledge_base=kb, ...)` takes
  keyword arguments and a `KnowledgeBase` object. The knowledge base is no
  longer saved, so pass it to `from_pretrained(path, knowledge_base=kb)`;
  tokenizers saved by 0.4 still load. `kbert.metrics` is renamed
  `kbert.feature_scoring`.

### Knowledge base (`tklearn.kb`)

- `KnowledgeBase("wiktionary")` is now `KnowledgeBase(WiktionaryStore())`.
  Store data is exposed as typed properties (`lexicon`, `triples`,
  `senses`, ...).
- `WiktionaryStore` (was `WiktionaryArtifactStore`) no longer creates a Hub
  repository or uploads on load. Use `offline=True` to skip the Hub and
  `push_to_hub()` to publish. Existing local caches keep working.
- `augment()` yields `Augmentation` objects (`text`, `original`, `span`,
  `replacement`, `relations`, `support`) instead of dicts, does not yield the
  unchanged text unless `include_original=True`, and takes `predicates`
  (default `("synonym",)`). The previously hard-coded `hyponym` and
  `instance` relations do not occur in the v1.0 Wiktionary data, which has
  `synonym`, `antonym`, `hypernym` and `category`.
- `extract_mentions(..., exact_form=False)` makes the lemma filter
  explicit; `stopwords` accepts a custom list.
- `extract_relations(candidate, predicates=None)` returns all relations by
  default.
- `Lexicon.extract(text, nested=True)` makes nested matches explicit;
  `Lexicon.tokeinze` is renamed `tokenize`; `Lexicon.load` accepts
  `case_sensitive`.

### Embeddings (`tklearn.embeddings`)

- `Embedding` defines `encode`, `encode_query`, `encode_document` and `dim`.
  `WordEmbedding` adds the vocabulary mapping. Use `GensimEmbedding(name)`,
  `FastTextEmbedding(name)` and `SentenceTransformerEmbedding(name)`
  instead of `AutoEmbedding`.
- `encode` returns a 1-D vector for a string and a 2-D array for a list of
  strings.

### Other

- `tklearn.config` is a plain dataclass; `cache_dir`, `temp_dir` and
  `assets_dir` are `Path`s derived from `base_dir`.
- `get_logger(name, level=None)`; repeated calls no longer add duplicate
  handlers.
- `tklearn.plotting` and `tklearn.nn.calibration` export their public
  functions.

### Fixed

- Cached word vectors (`GensimEmbedding`, `FastTextEmbedding`) were read
  without skipping the `.npy` header, so every lookup after the first run
  returned a shifted, wrong vector.
- `EarlyStopping` with `patience=0` stopped at epoch 1 even while improving.
- `PlateauEarlyStopping` and `ModelCheckpoint` crashed under numpy 2
  (`np.Inf`).
- `OptimalPRThreshold` reported F1 values as recall; `AUC(average=None)`
  failed on per-class results.
- `move_to_device` rejected batches containing numpy arrays, numbers or None.
- Named learning-rate schedules failed without `num_warmup_steps`.
- `tklearn.nn.huggingface.hub` failed to import.
