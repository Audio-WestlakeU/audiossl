
from torch  import nn
class LinearHead(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super().__init__()
        self.num_labels = num_labels
        self.norm = nn.BatchNorm1d(dim,affine=False)
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.unsqueeze(2)
        x = self.norm(x)
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)
