import torch.nn as nn
import torch.nn.functional as F


class Net9(nn.Module):
    def __init__(self):
        super(Net9, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv3_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv4_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.up_1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.up_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.up_3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.down_1 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.down_2 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.down_3 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.activation = nn.PReLU()
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)


    def forward(self, MAXIM, HiCPlus, HiCARN, HiCNN):
        MAXIM_p = self.conv1_1(MAXIM)
        MAXIM_p1 = F.relu(MAXIM_p)
        MAXIM_p2 = self.conv1_2(MAXIM_p1)
        MAXIM_p2 = F.relu(MAXIM_p2)
        HiCPlus_p = self.conv2_1(HiCPlus)
        HiCPlus_p1 = F.relu(HiCPlus_p)
        HiCARN_p = self.conv3_1(HiCARN)
        HiCARN_p1 = F.relu(HiCARN_p)
        HiCNN_p = self.conv4_1(HiCNN)
        HiCNN_p1 = F.relu(HiCNN_p)
        Diff_1 = self.activation(self.up_1(MAXIM_p2)) - HiCPlus_p1
        Fuse_1 = MAXIM_p2 + self.activation(self.down_1(Diff_1))
        Diff_2 = self.activation(self.up_2(Fuse_1)) - HiCARN_p1
        Fuse_2 = Fuse_1 + self.activation(self.down_2(Diff_2))
        Diff_3 = self.activation(self.up_3(Fuse_2)) - HiCNN_p1
        Fuse_3 = Fuse_2 + self.activation(self.down_3(Diff_3))
        out = F.relu(self.conv5(Fuse_3))

        return out