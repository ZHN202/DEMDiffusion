import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from diffusers import AutoencoderKL
import torch
from PIL import Image
from torchvision import transforms
from diffusers import UNet2DModel

unet = UNet2DModel(
    in_channels=8,
    out_channels=4,
    sample_size=64,
    attention_head_dim=8,
    block_out_channels=[
        320,
        640,
        1280,
        1280
    ],
    center_input_sample=False,
    down_block_types=[
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D"
    ],
    downsample_padding=1,
    flip_sin_to_cos=True,
    freq_shift=0,
    layers_per_block=2,
    mid_block_scale_factor=1,
    norm_eps=1e-05,
    norm_num_groups=32,
    up_block_types=[
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D"
    ],

)

print(unet)
input = torch.randn(1, 8, 64, 64)
timestep = torch.ones(1)
output = unet(input, timestep)[0]
print(output.shape)
