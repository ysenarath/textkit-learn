import torch
from torch.utils.data import DataLoader, TensorDataset

from tklearn.nn import Encoder  # Replace with the actual path to Encoder class
from tklearn.nn.base.module import Module
from tklearn.nn.utils.devices import get_device


# Dummy model for testing
class DummyModel(Module[torch.Tensor, dict]):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return {"pooler_output": self.linear(x.to(self.device))}


# Dummy dataset
def get_dummy_dataloader(batch_size=4):
    X = torch.randn(20, 10)  # 20 samples, 10 features
    dataset = TensorDataset(X)
    return DataLoader(dataset, batch_size=batch_size)


# Instantiate model, dataloader, and encoder
device = get_device()
model = DummyModel()
model.to(device)  # Move model to the appropriate device
dataloader = get_dummy_dataloader()

encoder = Encoder(
    model=model,
    dataloader=dataloader,
)

# Run encoding
encoded_output = encoder.encode()
print("Encoded output shape:", encoded_output.shape)
