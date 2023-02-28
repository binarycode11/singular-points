# build a simple equivariant model using a SequentialModule
import e2cnn
import torch
from e2cnn.nn import *

class KeyNet(torch.nn.Module):
    def __init__(self):
        super(KeyNet, self).__init__()
        s = e2cnn.gspaces.Rot2dOnR2(8)
        self.c_in = e2cnn.nn.FieldType(s, [s.trivial_repr]*1)
        c_hid = e2cnn.nn.FieldType(s, [s.regular_repr]*3)
        c_out = e2cnn.nn.FieldType(s, [s.regular_repr]*1)

        self.block1 = SequentialModule(
            R2Conv(self.c_in, c_hid, 3, bias=False),
            InnerBatchNorm(c_hid),
            ReLU(c_hid, inplace=True),
            PointwiseMaxPool(c_hid, kernel_size=3, stride=1, padding=1),
            R2Conv(c_hid, c_out, 3, bias=False),
            InnerBatchNorm(c_out),
            ELU(c_out, inplace=True),
            GroupPooling(c_out)
        )

    def forward(self, input_data):
        x = e2cnn.nn.GeometricTensor(input_data,
                                        self.c_in)
        x = self.block1(x)
        return x




# check that the two models are equivalent

net = KeyNet()

net.eval()

a1 = torch.zeros(1,1,12,12)
k1 = net(a1)
print(k1)

a2 = torch.ones(1,1,12,12)
k2 = net(a2)
print(k2)

batch = torch.cat((a1,a2))
k3 = net(batch)
print(k3)