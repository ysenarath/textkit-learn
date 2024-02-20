from tklearn.nn.callbacks import TrainerCallback


class ValidationCallback(TrainerCallback):
    def __init__(self) -> None:
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            msg = "logs are not provided"
            raise ValueError(msg)
        self.trainer.calculate_validation_loss()
        return super().on_epoch_end(epoch, logs)
