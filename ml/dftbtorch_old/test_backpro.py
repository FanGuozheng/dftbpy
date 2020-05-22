#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch as t


class TwoLayerNet(t.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = t.nn.Linear(D_in, H)
        self.linear2 = t.nn.Linear(H, D_out)

    def forward(self, x):
        """
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


x = t.zeros(10, 2)
y = t.zeros(10)
y_pred = t.randn(10, requires_grad=False)
for n in range(0, 10):
    for i in range(0, 2):
        x[n, i] = i+i*t.cos(t.Tensor([n]))
        y[n] = i+i*t.cos(t.Tensor([n+0.2]))
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 10, 2, 10, 1

# Create random Tensors to hold inputs and outputs


def test_fun(inp):
    s1, s2 = inp.shape
    out = t.zeros(s1)
    test = t.zeros(10, s1, s2)
    for i in range(0, 10):
        test[i, :, :] = inp
    for i in range(0, s1):
        for j in range(0, s2):
            out[i] = inp[i] + i
            out[i] = out[i] + i * out[i] * test[3, i, j]
    return out


# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = t.nn.MSELoss(reduction='sum')
optimizer = t.optim.SGD(model.parameters(), lr=1e-4)
y_out_pred = t.zeros(10)
for tt in range(0, 1000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)
    y_out_pred = test_fun(y_pred)
    # Compute and print loss
    loss = criterion(y_out_pred, y)
    if tt % 200 == 99:
        print(tt, loss.item())
        print(y_pred, y, loss)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
