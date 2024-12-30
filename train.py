# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import numpy as np
import random
from trainer import Trainer
from options import RMSFM2Options
import torch
import os

options = RMSFM2Options()
opts = options.parse()
torch.backends.cudnn.benchmark = True
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(1)  

if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()

'''
python train.py --gc
python train.py 
'''