'''test dftbpy'''

from dftb_torch import main
import torch as t
from torch.autograd import Variable


if __name__ == '__main__':
    para = {}
    para['readInput'] = True
    para['interpHS'] = False
    para['qatom_xlbomd'] = t.Tensor([4.3, 0.9, 0.9, 0.9, 0.9])
    main(para)
