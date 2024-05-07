import time

from tklearn.nn.callbacks.base import CallbackList
from tklearn.nn.callbacks.progbar_logger import ProgbarLogger

logger = ProgbarLogger()

callbacks = CallbackList([logger])

callbacks.set_params({
    "epochs": 10,
    "steps": 100,
    "pred_steps": 10,
    "metrics": ["loss"],
})

print(list(callbacks))
print(callbacks.params)

callbacks.on_train_begin()

for epoch in range(10):
    callbacks.on_epoch_begin(epoch)
    for batch in range(100):
        callbacks.on_train_batch_begin(batch)
        time.sleep(0.1)
        logs = {
            "loss": 0.5,
        }
        callbacks.on_train_batch_end(batch, logs)
    callbacks.on_predict_begin()
    for batch in range(10):
        callbacks.on_predict_batch_begin(batch)
        time.sleep(0.1)
        logs = {
            "valid_loss": 0.5,
        }
        callbacks.on_predict_batch_end(batch, logs)
    callbacks.on_predict_end()
    logs = {
        "loss": 0.5,
    }
    callbacks.on_epoch_end(epoch, logs)

callbacks.on_train_end()
