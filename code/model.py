'''
Wenrui Liu
2024-4-16

pix2pix Model for picture coloring
'''
import torch
import torch.nn as nn

class CNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, use_batchNorm=False, use_maxPool=True, down=True):
        super().__init__()
        self.down = down
        if down:
            self.conv = nn.Conv2d(in_channel, out_channel, 3,1,1)
        else:
            self.conv = nn.ConvTranspose2d(in_channel, out_channel, 4, 2, 1)
        self.use_batchNorm = use_batchNorm
        if use_batchNorm:
            self.batchNorm = nn.BatchNorm2d(out_channel)
        else:
            self.batchNorm = None
        self.relu = nn.ReLU()
        self.use_maxpool = use_maxPool
        if use_maxPool:
            self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        else:
            self.maxpool = None
    
    def forward(self, x):
        x = self.conv(x)
        if self.use_batchNorm:
            x = self.batchNorm(x)
        x = self.relu(x)
        if self.use_maxpool:
            x = self.maxpool(x)
        return x

# U-net generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # input:batch*1*32*32
        self.down1 = CNNLayer(1, 64, use_batchNorm=False, use_maxPool=True, down=True)
        
        # batch*64*16*16
        self.down2 = CNNLayer(64, 128, use_batchNorm=True, use_maxPool=True, down=True)

        # batch*128*8*8
        self.down3 = CNNLayer(128, 256, use_batchNorm=True, use_maxPool=True, down=True)

        # batch*256*4*4
        self.down4 = CNNLayer(256, 512, use_batchNorm=True, use_maxPool=True, down=True)

        # batch*512*2*2
        self.bottom = CNNLayer(512, 512, use_batchNorm=False, use_maxPool=False, down=True)

        # batch*512*2*2
        self.up4 = CNNLayer(512*2, 256, use_batchNorm=True, use_maxPool=False, down=False)

        # batch*256*4*4
        self.up3 = CNNLayer(256*2, 128, use_batchNorm=True, use_maxPool=False, down=False)

        # batch*128*8*8
        self.up2 = CNNLayer(128*2, 64, use_batchNorm=True, use_maxPool=False, down=False)

        # batch*64*16*16
        self.up1 = CNNLayer(64*2, 4, use_batchNorm=False, use_maxPool=False, down=False)

        # batch*3*32*32
        self.end = nn.Conv2d(4, 3, 1, 1, 0)

    def forward(self, x):
        xd1 = self.down1(x)
        xd2 = self.down2(xd1)
        xd3 = self.down3(xd2)
        xd4 = self.down4(xd3)
        xb = self.bottom(xd4)
        xu4 = self.up4(torch.cat([xb, xd4], 1))
        xu3 = self.up3(torch.cat([xu4, xd3], 1))
        xu2 = self.up2(torch.cat([xu3, xd2], 1))
        xu1 = self.up1(torch.cat([xu2, xd1], 1))
        res = self.end(xu1)

        return res


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # batch*(3+1)*32*32
        self.conv1 = CNNLayer(3+1, 32, use_batchNorm=True, use_maxPool=True)
        # batch*64*16*16
        self.conv2 = CNNLayer(32, 64, use_batchNorm=True,use_maxPool=True)
        # batch*128*8*8
        # self.conv3 = CNNLayer(128, 256, use_batchNorm=True,use_maxPool=True)
        # batch*256*4*4
        # self.conv4 = CNNLayer(256, 512, use_batchNorm=True,use_maxPool=False)
        # batch*512*4*4
        self.conv5 = nn.Conv2d(64, 1, 3, 1, 1)
        # batch*1*4*4


    def forward(self, x, y):
        x = torch.cat([x,y],dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        x = self.conv5(x)
        return x


if __name__ == "__main__":
    x = torch.randn((1, 1, 32, 32))
    y = torch.randn((1, 3, 32, 32))
    z = torch.randn((1, 1, 32, 32))
    model = Generator()
    z=  model(z)
    dis = Discriminator()
    x = dis(x,y)
    print(z.size(),x.size())

