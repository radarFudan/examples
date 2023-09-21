import scipy

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P



class ExpLambda(nn.Module):
    def forward(self, X):
        return torch.matrix_exp(X)  # Return an exponential parametrized matrix

    def right_inverse(self, A):
        return torch.real(torch.from_numpy(scipy.linalg.logm(A))).to(torch.float32)


class StableRNN(nn.Module):
    def __init__(self,
                 activation,
                 hid_dim,
                 input_dim,
                 output_dim,
                 dt=0.1,
                ):
        """_summary_

        Args:
            activation (_type_): _description_
            hid_dim (_type_): _description_
            input_dim (_type_): _description_
            output_dim (_type_): _description_

            dt (float, optional): _description_. Defaults to 0.1.
        """

        super().__init__()
        if activation == "linear":
            self.activation = None
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "hardtanh":
            self.activation = torch.nn.functional.hardtanh

        # U, W, c
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
        self.U = nn.Linear(input_dim, hid_dim, bias=True, dtype=torch.float32)

        # Need unsafe parameterization to get the low rank W
        self.W = nn.Linear(hid_dim, hid_dim, bias=False, dtype=torch.float32)

        self.c = nn.Linear(hid_dim, output_dim, bias=True, dtype=torch.float32)

        P.register_parametrization(self.W, "weight", ExpLambda())

        self.dt = dt
        self.hid_dim = hid_dim

    def forward(self, x, h=None):
        
        #src = [batch size, input len, input dim]
        x = torch.transpose(x, 0, 1)
        length = x.shape[1]

        # What is the meaning of the first 1, 1 in hidden
        # Does it mean the first 1 will be broadcast into batches?
        hidden = []
        if h is None:
            hidden.append(torch.zeros(1, 1, self.hid_dim, dtype=x.dtype, device=x.device))
        else:
            h = torch.transpose(h, 0, 1)
# print(h.shape)
            hidden.append(h[:,0,:])
# print(hidden[0].shape)

# print("recur", self.U.weight.shape)
        x = self.U(x)
# print("x", x.shape)

        # Residual RNN
        for i in range(length):
            if self.activation is None:
# if i<2:
# print(i)
# print(hidden[i].shape)
# print(x[:,i,:].shape)
# print(self.W(hidden[i]).shape)
                h_next = hidden[i] + self.dt * (x[:,i,:]-self.W(hidden[i]))
# print("h_next", h_next.shape)
            else:
# if i<2:
# print(i)
# print(hidden[i].shape)
# print(x[:,i,:].shape)
# print(self.W(hidden[i]).shape)
                tmp = self.dt * self.activation(x[:,i,:]-self.W(hidden[i]))
# print("tmp", tmp.shape)

                h_next = hidden[i] + self.dt * self.activation(x[:,i,:]-self.W(hidden[i]))
# print("h_next", h_next.shape)
            hidden.append(h_next)
        hidden = torch.cat(hidden[1:], dim=0)

        out = self.c(hidden)
# print("out", out.shape)
# print("hid", h_next.unsqueeze(1))
        return out, h_next.unsqueeze(1)


        
