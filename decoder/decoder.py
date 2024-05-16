import torch
import torch.nn as nn
from diffusers.models.autoencoders.vae import Decoder
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchview import draw_graph
class DEMDecoder2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 up_block_types,
                 block_out_channels,
                 layers_per_block,
                 norm_num_groups,
                 act_fn,
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 1)
        if self.training:
            self.decoder = Decoder(
                in_channels,
                out_channels,
                up_block_types,
                block_out_channels,
                layers_per_block,
                norm_num_groups,
                act_fn
            ).train()
        else:
            self.decoder = Decoder(
                in_channels,
                out_channels,
                up_block_types,
                block_out_channels,
                layers_per_block,
                norm_num_groups,
                act_fn
            )
        # 128->256
    def forward(self, x):
        x = self.conv(x)
        x = self.decoder(x)
        return x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_num_groups, act_fn):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(num_channels=out_channels, num_groups=norm_num_groups, eps=1e-6)
        self.act_fn1 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(num_channels=out_channels, num_groups=norm_num_groups, eps=1e-6)
        self.act_fn2 = nn.SiLU()
        self.residual = nn.Conv2d(in_channels, out_channels, 1)
        self.act_fn3 = nn.SiLU()

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act_fn1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act_fn2(x)
        x = x + residual
        x = self.act_fn3(x)
        return x

class DEMDecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 up_block_types,
                 block_out_channels,
                 layers_per_block,
                 norm_num_groups,
                 act_fn,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers_per_block = layers_per_block

        # image_down
        # 512*512->256*256
        self.down_block1 = self.down_block(3, self.in_channels)
        # 256*256->128*128
        self.down_block2 = self.down_block(self.in_channels, self.in_channels)
        # 128*128->64*64
        self.down_block3 = self.down_block(self.in_channels, self.in_channels)

        self.post_quant_conv = nn.Conv2d(self.in_channels, self.in_channels, 1)
        self.mid_blocks = self.mid_block(self.in_channels, self.in_channels, self.in_channels, act_fn)
        self.mid_blocks_ = self.mid_block(3, self.in_channels, self.in_channels, act_fn)

        # size 64->128 channels 16->block_out_channels[0]
        self.fit_blocks_1 = self.fit_block(self.in_channels*2, block_out_channels[0])
        self.up_blocks_1 = self.up_block(block_out_channels[0], block_out_channels[0], norm_num_groups)

        # size 128->256 channels block_out_channels[0]->block_out_channels[1]
        self.fit_blocks_2 = self.fit_block(block_out_channels[0]+self.in_channels, block_out_channels[1])
        self.up_blocks_2 = self.up_block(block_out_channels[1], block_out_channels[1], norm_num_groups)

        # size 256->512 channels block_out_channels[1]->block_out_channels[2]
        self.fit_blocks_3 = self.fit_block(block_out_channels[1]+self.in_channels, block_out_channels[2])
        self.up_blocks_3 = self.up_block(block_out_channels[2], block_out_channels[2], norm_num_groups)

        # size 512->512 channels block_out_channels[2]->block_out_channels[3]
        self.fit_blocks_4 = self.fit_block(block_out_channels[2]+self.in_channels, block_out_channels[3])
        self.out_blocks = nn.ModuleList(
            [
                ResNetBlock(block_out_channels[3], block_out_channels[3]//2, norm_num_groups, act_fn),
                ResNetBlock(block_out_channels[3]//2, block_out_channels[3]//2, norm_num_groups, act_fn),
                nn.Conv2d(block_out_channels[3]//2, block_out_channels[3]//2, 3, 1, 1),
                nn.GroupNorm(num_channels=block_out_channels[3]//2, num_groups=norm_num_groups, eps=1e-6),
                nn.SiLU(),
                nn.Conv2d(block_out_channels[3]//2, self.out_channels, 3,1,1),
                nn.ReLU()
            ]
        )
        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def fit_block(self, in_channels, out_channels,norm_num_groups=32):
        return nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.SiLU(),
                ResNetBlock(out_channels, out_channels, norm_num_groups, "silu"),
            ]
        )
    def up_block(self, in_channels, out_channels, norm_num_groups):
        return nn.ModuleList(
            [
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
                nn.GroupNorm(num_channels=out_channels, num_groups=norm_num_groups, eps=1e-6),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.GroupNorm(num_channels=out_channels, num_groups=norm_num_groups, eps=1e-6),
                nn.SiLU(),
            ]
        )
    def mid_block(self, in_channels, out_channels, norm_num_groups, act_fn):
        return nn.ModuleList(
            [
                ResNetBlock(in_channels, out_channels, norm_num_groups, act_fn),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.GroupNorm(num_channels=out_channels, num_groups=norm_num_groups, eps=1e-6),
                nn.SiLU(),
            ]
        )
    def down_block(self, in_channels, out_channels):
        return nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.SiLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.SiLU(),
            ]
        )

    def forward(self, x,image):
        x = self.post_quant_conv(x)
        # image_down
        # 515->256
        image_256 = image
        for layer in self.down_block1:
            image_256 = layer(image_256)
        for layer in self.mid_blocks:
            image_256 = layer(image_256)
        # 256->128
        image_128 = image_256
        for layer in self.down_block2:
            image_128 = layer(image_128)
        for layer in self.mid_blocks:
            image_128 = layer(image_128)
        # 128->64
        image_64 = image_128
        for layer in self.down_block3:
            image_64 = layer(image_64)
        for layer in self.mid_blocks:
            image_64 = layer(image_64)

        for layer in self.mid_blocks_:
            image = layer(image)

        # size 64->128 channels 16->block_out_channels[0]
        x = torch.cat([x, image_64], dim=1)
        for layer in self.fit_blocks_1:
            x = layer(x)
        for layer in self.up_blocks_1:
            x = layer(x)

        # size 128->256 channels block_out_channels[0]->block_out_channels[1]
        x = torch.cat([x, image_128], dim=1)
        for layer in self.fit_blocks_2:
            x = layer(x)
        for layer in self.up_blocks_2:
            x = layer(x)

        # size 256->512 channels block_out_channels[1]->block_out_channels[2]
        x = torch.cat([x, image_256], dim=1)
        for layer in self.fit_blocks_3:
            x = layer(x)
        for layer in self.up_blocks_3:
            x = layer(x)

        # size 512->512 channels block_out_channels[2]->block_out_channels[3]
        x = torch.cat([x, image], dim=1)
        for layer in self.fit_blocks_4:
            x = layer(x)
        for layer in self.out_blocks:
            x = layer(x)


        return x

class DEMDecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 up_block_types,
                 block_out_channels,
                 layers_per_block,
                 norm_num_groups,
                 act_fn,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers_per_block = layers_per_block

        # image_down
        # 512*512->256*256
        self.down_block1 = self.down_block(3, self.in_channels)
        # 256*256->128*128
        self.down_block2 = self.down_block(self.in_channels, self.in_channels)
        # 128*128->64*64
        self.down_block3 = self.down_block(self.in_channels, self.in_channels)

        self.post_quant_conv = nn.Conv2d(self.in_channels, self.in_channels, 1)
        self.mid_blocks = self.mid_block(self.in_channels, self.in_channels, self.in_channels, act_fn)
        self.mid_blocks_ = self.mid_block(3, self.in_channels, self.in_channels, act_fn)

        # size 64->128 channels 16->block_out_channels[0]
        self.fit_blocks_1 = self.fit_block(self.in_channels*2, block_out_channels[0])
        self.up_blocks_1 = self.up_block(block_out_channels[0], block_out_channels[0], norm_num_groups)

        # size 128->256 channels block_out_channels[0]->block_out_channels[1]
        self.fit_blocks_2 = self.fit_block(block_out_channels[0]+self.in_channels, block_out_channels[1])
        self.up_blocks_2 = self.up_block(block_out_channels[1], block_out_channels[1], norm_num_groups)

        # size 256->512 channels block_out_channels[1]->block_out_channels[2]
        self.fit_blocks_3 = self.fit_block(block_out_channels[1]+self.in_channels, block_out_channels[2])
        self.up_blocks_3 = self.up_block(block_out_channels[2], block_out_channels[2], norm_num_groups)

        # size 512->512 channels block_out_channels[2]->block_out_channels[3]
        self.fit_blocks_4 = self.fit_block(block_out_channels[2]+self.in_channels, block_out_channels[3])
        self.out_blocks = nn.ModuleList(
            [
                ResNetBlock(block_out_channels[3], block_out_channels[3]//2, norm_num_groups, act_fn),
                ResNetBlock(block_out_channels[3]//2, block_out_channels[3]//2, norm_num_groups, act_fn),
                nn.Conv2d(block_out_channels[3]//2, block_out_channels[3]//2, 3, 1, 1),
                nn.GroupNorm(num_channels=block_out_channels[3]//2, num_groups=norm_num_groups, eps=1e-6),
                nn.SiLU(),
                nn.Conv2d(block_out_channels[3]//2, self.out_channels, 3,1,1),
                nn.ReLU()
            ]
        )
        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def fit_block(self, in_channels, out_channels,norm_num_groups=32):
        return nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.SiLU(),
                ResNetBlock(out_channels, out_channels, norm_num_groups, "silu"),
            ]
        )
    def up_block(self, in_channels, out_channels, norm_num_groups):
        return nn.ModuleList(
            [
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
                nn.GroupNorm(num_channels=out_channels, num_groups=norm_num_groups, eps=1e-6),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.GroupNorm(num_channels=out_channels, num_groups=norm_num_groups, eps=1e-6),
                nn.SiLU(),
            ]
        )
    def mid_block(self, in_channels, out_channels, norm_num_groups, act_fn):
        return nn.ModuleList(
            [
                ResNetBlock(in_channels, out_channels, norm_num_groups, act_fn),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.GroupNorm(num_channels=out_channels, num_groups=norm_num_groups, eps=1e-6),
                nn.SiLU(),
            ]
        )
    def down_block(self, in_channels, out_channels):
        return nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.SiLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.SiLU(),
            ]
        )

    def forward(self, x,image):
        x = self.post_quant_conv(x)
        # image_down
        # 515->256
        image_256 = image
        for layer in self.down_block1:
            image_256 = layer(image_256)
        for layer in self.mid_blocks:
            image_256 = layer(image_256)
        # 256->128
        image_128 = image_256
        for layer in self.down_block2:
            image_128 = layer(image_128)
        for layer in self.mid_blocks:
            image_128 = layer(image_128)
        # 128->64
        image_64 = image_128
        for layer in self.down_block3:
            image_64 = layer(image_64)
        for layer in self.mid_blocks:
            image_64 = layer(image_64)

        for layer in self.mid_blocks_:
            image = layer(image)

        # size 64->128 channels 16->block_out_channels[0]
        x = torch.cat([x, image_64], dim=1)
        for layer in self.fit_blocks_1:
            x = layer(x)
        for layer in self.up_blocks_1:
            x = layer(x)

        # size 128->256 channels block_out_channels[0]->block_out_channels[1]
        x = torch.cat([x, image_128], dim=1)
        for layer in self.fit_blocks_2:
            x = layer(x)
        for layer in self.up_blocks_2:
            x = layer(x)

        # size 256->512 channels block_out_channels[1]->block_out_channels[2]
        x = torch.cat([x, image_256], dim=1)
        for layer in self.fit_blocks_3:
            x = layer(x)
        for layer in self.up_blocks_3:
            x = layer(x)

        # size 512->512 channels block_out_channels[2]->block_out_channels[3]
        x = torch.cat([x, image], dim=1)
        for layer in self.fit_blocks_4:
            x = layer(x)
        for layer in self.out_blocks:
            x = layer(x)


        return x

class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, norm_num_groups):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_channels=in_channels, num_groups=norm_num_groups, eps=1e-6)
        self.act_fn1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        self.norm2 = nn.GroupNorm(num_channels=out_channels, num_groups=norm_num_groups, eps=1e-6)
        self.act_fn2 = nn.SiLU()
        self.dropout = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        self.residual = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        residual = self.residual(x)
        x = self.norm1(x)
        x = self.act_fn1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act_fn2(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x + residual
        x = x.div(2.0)
        return x

class Upsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.interpolate = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
    def forward(self, x):
        x = self.interpolate(x)
        x = self.conv(x)

        return x
class Downsample2D(nn.Module):
    def __init__(self, in_channels, out_channels, norm_num_groups):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm = nn.GroupNorm(num_channels=out_channels, num_groups=norm_num_groups, eps=1e-6)
        self.act_fn = nn.SiLU()
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act_fn(x)
        x = self.pool(x)
        return x


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, norm_num_groups):
        super().__init__()

        self.resnet1 = ResnetBlock2D(in_channels, out_channels, norm_num_groups)
        self.resnet2 = ResnetBlock2D(out_channels, out_channels, norm_num_groups)
        self.upsample = Upsample2D(out_channels, out_channels)

    def forward(self, x):
        x = self.resnet1(x)
        x = self.resnet2(x)
        x = self.upsample(x)
        return x

class FitBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, norm_num_groups):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.act_fn = nn.SiLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.act_fn(x)
        return x

class MidBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, norm_num_groups):
        super().__init__()
        self.resnet1 = ResnetBlock2D(in_channels, out_channels//2, norm_num_groups)
        self.resnet2 = ResnetBlock2D(out_channels//2, out_channels, norm_num_groups)
        self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
    def forward(self, x):
        x = self.resnet1(x)
        x = self.resnet2(x)
        x = self.conv(x)
        return x


class ConditionalDecoder(nn.Module):
    def __init__(self, in_channels, out_channels,  block_out_channels,  norm_num_groups):
        super().__init__()


        # image 512 256 128 64    channel 3
        # dem   64  128 256 512   channel 4  512  512

        # image
        self.in_conv_image = nn.Conv2d(3,block_out_channels[2],1)
        # image_size 512->256  channel 128->256
        self.down_block_1 = Downsample2D(block_out_channels[2], block_out_channels[1], norm_num_groups)
        # image_size 256->128  channel 256->512
        self.down_block_2 = Downsample2D(block_out_channels[1], block_out_channels[0], norm_num_groups)
        # image_size 128->64   channel 512->4
        self.down_block_3 = Downsample2D(block_out_channels[0], in_channels, in_channels)

        # dem
        self.in_conv_dem = nn.Conv2d(in_channels, in_channels, 1)
        # channel 4+4->512           size 64->128
        self.fit_block_1 = FitBlock2D(in_channels*2, block_out_channels[0], norm_num_groups)
        self.mid_block_1 = MidBlock2D(block_out_channels[0], block_out_channels[0], norm_num_groups)
        self.upsample_block_1 = UpDecoderBlock2D(block_out_channels[0], block_out_channels[0], norm_num_groups)

        # channel 512+512->256       size 128->256
        self.fit_block_2 = FitBlock2D(block_out_channels[0]*2,block_out_channels[1],norm_num_groups)
        self.upsample_block_2 = UpDecoderBlock2D(block_out_channels[1], block_out_channels[1], norm_num_groups)

        # channel 256+256->128       size 256->512
        self.fit_block_3 = FitBlock2D(block_out_channels[1]*2,block_out_channels[2],norm_num_groups)
        self.upsample_block_3 = UpDecoderBlock2D(block_out_channels[2], block_out_channels[2], norm_num_groups)

        # channel 128+128->64        size 512->512
        self.fit_block_4 = FitBlock2D(block_out_channels[2]*2,block_out_channels[2],norm_num_groups)
        self.mid_block_2 = MidBlock2D(block_out_channels[2], block_out_channels[2]//2, norm_num_groups)

        self.out_conv = nn.Conv2d(block_out_channels[2]//2, out_channels, 3, 1, 1)

    def forward(self, x,image):
        image = self.in_conv_image(image)
        image_256 = self.down_block_1(image)
        image_128 = self.down_block_2(image_256)
        image_64 = self.down_block_3(image_128)

        x = self.in_conv_dem(x)
        # 4
        x = torch.cat([x, image_64], dim=1)
        x = self.fit_block_1(x)
        x = self.mid_block_1(x)
        x = self.upsample_block_1(x)

        # out_block[0]
        x = torch.cat([x, image_128], dim=1)
        x = self.fit_block_2(x)
        x = self.upsample_block_2(x)

        # out_block[1]
        x = torch.cat([x, image_256], dim=1)
        x = self.fit_block_3(x)
        x = self.upsample_block_3(x)

        # out_block[2]
        x = torch.cat([x, image], dim=1)
        x = self.fit_block_4(x)
        x = self.mid_block_2(x)
        x = self.out_conv(x)


        return x











if __name__ == '__main__':
    in_channels = 4
    out_channels = 1
    up_block_types = [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D"
    ]
    block_out_channels = [
        512,
        256,
        128
    ]
    layers_per_block = 1
    norm_num_groups = 32
    act_fn = "silu"

    x = torch.randn(1, in_channels, 64, 64).cuda()
    y = torch.randn(1, 3, 512, 512).cuda()
    model = ConditionalDecoder(in_channels,
                       out_channels,
                       block_out_channels,
                        norm_num_groups).cuda()

    # out = model(x,y)
    summary(model, [(in_channels, 64, 64), (3, 512, 512)])
    model_graph = draw_graph(model, input_data=[x,y],depth=1, expand_nested=True, save_graph=True, filename="torchview3", directory=r".")
    model_graph.visual_graph

    writer = SummaryWriter('log') #建立一个保存数据用的东西
    writer.add_graph(model, input_to_model=[x,y])
