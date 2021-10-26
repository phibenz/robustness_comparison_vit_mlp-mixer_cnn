import torch
import torch.nn as nn


def uap(in_planes):
    # The universal adversarial representation is represented as a 1x1 convolution
    return nn.Conv2d(in_planes, in_planes,
                    kernel_size=1, stride=1,
                    padding=0, bias=False)

class UAP(nn.Module):
    def __init__(self,
                shape=(224, 224),
                num_channels=3):
        super(UAP, self).__init__()

        self.delta = nn.Parameter(torch.zeros(size=(num_channels, *shape), requires_grad=True))

    def forward(self, x):
        delta = self.delta
        # Add uap to input
        adv_x = x + delta

        return adv_x
