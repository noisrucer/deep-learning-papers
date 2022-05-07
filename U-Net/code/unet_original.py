import torch
import torch.nn as nn
from utils import crop_img, concat_imgs_crop

def double_conv(in_channel, out_channel):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=0),
        nn.ReLU(inplace=True)
    )

    return conv


def trans_conv(in_channel, out_channel):
    return nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)


class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=10):
        super(UNet, self).__init__()
        self.n_classes = n_classes

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder - Contracting Path
        self.down_conv_1 = double_conv(in_channels, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        # Decoder - Expansive Path
        self.up_conv_1 = double_conv(1024, 512)
        self.up_conv_2 = double_conv(512, 256)
        self.up_conv_3 = double_conv(256, 128)
        self.up_conv_4 = double_conv(128, 64)

        # Final Layer
        self.final_layer = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, img):
        '''
        img: (B, C, H, W)
        '''

        # Encoder - Contracting Path
        d1 = self.down_conv_1(img) # concat
        d2 = self.down_conv_2(self.max_pool(d1)) # concat
        d3 = self.down_conv_3(self.max_pool(d2)) # concat
        d4 = self.down_conv_4(self.max_pool(d3)) # concat
        d5 = self.down_conv_5(self.max_pool(d4))

        # Decoder - Expansive Path
        u4 = trans_conv(1024, 512)(d5)
        u4 = concat_imgs_crop(d4, u4)

        u3 = trans_conv(512, 256)(self.up_conv_1(u4))
        u3 = concat_imgs_crop(d3, u3)

        u2 = trans_conv(256, 128)(self.up_conv_2(u3))
        u2 = concat_imgs_crop(d2, u2)

        u1 = trans_conv(128, 64)(self.up_conv_3(u2))
        u1 = concat_imgs_crop(d1, u1)

        out = self.up_conv_4(u1)

        # Final Layer
        out = self.final_layer(out)
        print(out.shape)

        return out



if __name__ == '__main__':
    img = torch.rand(1, 3, 572, 572)
    model = UNet(in_channels=3, n_classes=10)
    model(img)
