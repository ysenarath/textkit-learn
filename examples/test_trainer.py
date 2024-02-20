import time
from typing import Any, Tuple

import torch

from tklearn.nn.callbacks import ProgbarLogger, ValidationCallback
from tklearn.nn.dataset import TorchDataset
from tklearn.nn.torch import TorchTrainer


class LogicGates(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def training_step(
    trainer: TorchTrainer, batch: TorchDataset, **kwargs
) -> Tuple[Any, torch.Tensor]:
    y_hat = trainer.model(batch.data)
    loss = torch.nn.functional.binary_cross_entropy(y_hat, batch.target)
    time.sleep(0.1)
    return y_hat, loss


def configure_optimizers(trainer: TorchTrainer):
    return torch.optim.Adam(trainer.model.parameters())


model = LogicGates()

trainer = TorchTrainer(
    model=model,
    num_epochs=10,
    batch_size=4,
    device="cpu",
    callbacks=[
        ValidationCallback(),
        ProgbarLogger(),
    ],
)

trainer.register("training_step", training_step)
trainer.register("configure_optimizers", configure_optimizers)

# or, and, and not -> [operator, operand1, operand2]
# operator = 0 -> or
# operator = 1 -> and
# operator = 2 -> not
# operand1/2 = 0 -> False
# operand1/2 = 1 -> True
data = torch.tensor(
    [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
        [2, 0, 0],
        [2, 1, 0],
        [2, 0, 1],
        [2, 1, 1],
    ],
    dtype=torch.float32,
)
target = torch.tensor(
    [[0], [1], [1], [1], [1], [0], [0], [0], [1], [1], [0], [0]],
    dtype=torch.float32,
)

trainer.fit(data, target)
