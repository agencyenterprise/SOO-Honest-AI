import torch.nn as nn
import torch
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = torch.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        for_h1 = torch.tensor([[ 1.8483e-01, -7.0044e-02, -2.1018e-02, -5.4828e-03, -5.5185e-02,
          9.8155e-04, -5.8708e-02,  8.2189e-02,  1.1120e-03,  1.6135e-02,
          5.4340e-02,  8.3427e-02,  3.1060e-03,  4.7255e-03, -2.7330e-01,
         -6.7700e-02,  4.5142e-03,  1.4767e-01,  2.2081e-03,  9.1233e-03,
          5.2594e-03, -8.0504e-02,  4.3673e-03,  1.0520e-01, -6.2893e-02,
          4.5729e-04,  2.4431e-01,  1.7773e-02, -3.8822e-02, -8.6239e-02,
          7.4611e-02,  5.6760e-04, -4.8670e-01,  1.6442e-02, -1.4110e-02,
         -8.4462e-02,  1.1486e-01,  4.1862e-02,  1.5915e-02, -2.5603e-03,
         -2.0943e-02, -3.4070e-03,  6.3524e-03, -1.9118e-03, -1.2822e-01,
         -5.7752e-02,  1.1313e-02,  2.8271e-03, -6.5929e-05,  5.5830e-05,
          2.1314e-01, -3.0183e-02, -8.6665e-02,  2.9184e-03, -8.3792e-03,
          1.8061e-02, -2.0719e-02, -1.4946e-01, -2.5718e-01,  1.5572e-01,
         -1.7626e-01, -8.3751e-02,  6.4761e-02,  8.0969e-03]])
        for_h2 =  torch.tensor([[-6.4507e-02, -9.3250e-02,  1.6960e-03, -2.0703e-01, -2.2208e-03,
         -3.0279e-03,  4.5021e-04, -7.6319e-03, -1.0909e-01, -4.5874e-03,
          3.3943e-02, -5.2208e-04, -1.5270e-01, -1.4257e-01, -2.1375e-03,
          4.8784e-02, -1.0878e-02, -6.8945e-04, -2.1997e-03,  1.6732e-01,
          1.0258e-01,  2.5472e-05,  5.9712e-02, -4.4964e-03,  4.1212e-03,
         -1.0171e-02, -3.0829e-03,  1.0267e-03, -3.0180e-03,  4.9280e-03,
          5.6452e-03, -1.5366e-03, -2.5416e-03,  4.7989e-04, -7.5882e-02,
          2.0498e-04,  1.6012e-03, -2.0317e-01, -1.7729e-02, -3.7721e-02,
         -3.9190e-01, -2.0997e-03,  6.2307e-02,  2.5994e-03,  3.6904e-03,
         -1.3495e-01, -7.8623e-04, -8.2088e-01,  7.2570e-03, -7.5923e-02,
          2.1660e-02,  7.6357e-04,  1.5432e-02,  2.1245e-04, -6.7201e-02,
         -3.6046e-03,  9.2550e-02, -1.4953e-01,  6.4396e-03, -2.1453e-03,
          1.6539e-03,  6.7940e-05,  1.0117e-03, -6.9122e-03]])
        h1 = self.nonlin(self.fc1(self.in_fn(X))) 
        h2 = self.nonlin(self.fc2(h1)) 
        out = self.out_fn(self.fc3(h2))
        return h1, h2, out
