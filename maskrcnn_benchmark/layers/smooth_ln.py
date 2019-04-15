#! /usr/bin/env python
# -*- coding:utf8 -*-

import torch
import numpy as np

def smooth_ln(x, smooth):
    return torch.where(
        torch.le(x, smooth),
        -torch.log(1 - x),
        ((x - smooth) / (1 - smooth)) - np.log(1 - smooth)
    )