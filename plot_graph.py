import torch
from torch.utils.tensorboard import SummaryWriter

def plot_graph(model, input_tensor):
    with SummaryWriter(comment='Net') as writer:
        writer.add_graph(model, input_tensor)

# Example usage
model = torch.nn.Sequential(
          torch.nn.Linear(3, 5),
          torch.nn.ReLU(),
          torch.nn.Linear(5, 1)
        )
input_tensor = torch.Tensor([1, 2, 3])
plot_graph(model, input_tensor)
