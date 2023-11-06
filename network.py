import torch
from net.JTNet import JNet, TNet


class net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_net = JNet()
        self.mask_t_net = TNet()

    def forward(self, data):
        x_j = self.image_net(data)
        x_t = self.mask_t_net(data)
        return x_j, x_t
