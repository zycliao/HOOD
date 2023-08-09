from dataclasses import dataclass

from torch import nn


@dataclass
class Config:
    weight: float = 1.


def create(mcfg):
    return Criterion(weight=mcfg.weight)


class Criterion(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.name = 'supervise'

    def forward(self, sample):
        cloth_sample = sample['cloth']

        pred_pos = cloth_sample.pred_pos
        target_pos = cloth_sample.target_pos
        loss = (pred_pos - target_pos).abs().mean() * self.weight

        # print('loss', loss)
        return dict(loss=loss)
