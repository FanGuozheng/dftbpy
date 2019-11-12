#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable


test1 = torch.Tensor(3, 3)
test2 = torch.rand(3, 3)
test3 = torch.zeros(3, 3)
test23 = test2+test3
test23_ = torch.add(test2, test3)
test4 = Variable(torch.ones(3, 3), requires_grad=True)
test5 = test4.sum()
test5.grad_fn
test5.backward()
print(test4.grad)
test5.backward()
test6 = torch.arange(0, 6)
test7 = test6.view(1, 2, 3)
test8 = test7.squeeze(0)
test9 = test6.resize_(3, 3, 3)
print(test4.grad, '\n', test2, '\n', test6, test7, test9)
