# Callbacks

Examples of `tklearn.nn.callbacks` with `tklearn.nn.Trainer`. Each one
trains a small classifier on synthetic data in seconds, then checks that
the callbacks did what they claim and prints a line per check.

| File | Shows |
|---|---|
| [`early_stopping.py`](early_stopping.py) | `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, `CSVLogger` and `ProgbarLogger` on a model that overfits |
| [`custom_callbacks.py`](custom_callbacks.py) | Writing callbacks: the hook order, a `Callback` subclass that freezes layers, stopping with `trainer.should_stop`, and `LambdaCallback` |
| [`terminate_on_nan.py`](terminate_on_nan.py) | `TerminateOnNaN` stopping on a corrupted example, with step checkpoints to fall back to |
| [`common.py`](common.py) | The data, model and checks shared by the examples |

## Running

```sh
python examples/callbacks/early_stopping.py
python examples/callbacks/custom_callbacks.py
python examples/callbacks/terminate_on_nan.py
```

Each example writes its files (checkpoints, CSV) to
`examples/outputs/callbacks/<example>/`, emptied at the start; pass
`--out DIR` to write elsewhere. A failed check raises an
`AssertionError`.

## Data and model

`common.make_splits` draws 16 Gaussian features and labels them by a
random hyperplane, then flips 20% of the labels. `Classifier` is a
two-layer perceptron (`encoder`, then `head`) with 256 hidden units. On
128 training examples it overfits after about 13 epochs at a learning
rate of 1e-3.

## Things the examples show

- **Callback order matters** when callbacks share state. In
  `early_stopping.py`, the learning rate is recorded in `on_epoch_begin`,
  so it is the rate the epoch trained with. `ReduceLROnPlateau` changes
  the rate at the end of an epoch, for the next one.
- **The interrupted epoch is still logged.** After a callback sets
  `trainer.should_stop`, training ends after the current batch. That
  epoch is evaluated and passed to `on_epoch_end` first.
- **The optimizer steps at the end of every epoch** with gradient
  accumulation, even when the last window is short: 3 batches with
  `gradient_accumulation_steps=2` take 2 steps.
- **`TerminateOnNaN` sees only the loss.** On Apple MPS, ReLU turns NaN
  into 0. A NaN input then turns the weights NaN while the loss stays
  finite, so `terminate_on_nan.py` runs on the CPU.
