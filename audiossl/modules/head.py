
from torch  import nn
class LinearHead(nn.Module):
    """Linear layer
    """
    def __init__(self, dim, num_labels=1000,use_norm=True,affine=False):
        super().__init__()
        self.num_labels = num_labels
        self.use_norm=use_norm
        if use_norm:
            self.norm = nn.BatchNorm1d(dim,affine=affine)
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        if self.use_norm:
            x = x.unsqueeze(2)
            x = self.norm(x)
        x = x.view(x.size(0), -1)
        # linear layer
        return self.linear(x)
