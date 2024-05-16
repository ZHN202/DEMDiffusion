import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# Bootstrapped from:
# https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import argparse
import math
import os

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    DDIMScheduler,
    UNet2DModel, AutoencoderKL,
)
from diffusers.optimization import get_scheduler

from tqdm.auto import tqdm


from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from pathlib import Path

import random
from torch.utils.tensorboard import SummaryWriter

prompt = [
    "generate DEM, according to the given RGB image"
]


def _randomset(lis):
    ret = []
    for i in range(len(lis)):
        if random.random() < 0.5:
            ret.append(lis[i])
    return ret


def _shuffle(lis):
    return random.sample(lis, len(lis))


class DreamBoothTiDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
            self,
            instance_image_data_root,
            instance_DEM_data_root,
            output_dir,
            size=64,
    ):
        self.size = size

        self.instance_image_data_root = Path(instance_image_data_root)
        self.instance_DEM_data_root = Path(instance_DEM_data_root)
        if not self.instance_image_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        if not self.instance_DEM_data_root.exists():
            raise ValueError("Instance DEM root doesn't exists.")
        self.instance_images_path_ = list(Path(instance_image_data_root).iterdir())
        self.instance_DEM_path_ = list(Path(instance_DEM_data_root).iterdir())

        # 去重
        instance_images_path_ = [i.name[:-4] for i in self.instance_images_path_]
        instance_dem_path_ = [i.name[:-4] for i in self.instance_DEM_path_]

        ex_ = [i.name[:-5] for i in list(Path(output_dir).iterdir())]
        self.instance_images_path = [self.instance_images_path_[i] for i in range(len(instance_images_path_)) if
                                     instance_images_path_[i] not in ex_]
        self.instance_DEM_path = [self.instance_DEM_path_[i] for i in range(len(instance_dem_path_)) if
                                  instance_dem_path_[i] not in ex_]
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.DEM_transforms = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images]
        )
        if instance_image.mode!='RGB':
            instance_image = instance_image.convert('RGB')
        instance_DEM = Image.open(
            self.instance_DEM_data_root.joinpath(
                self.instance_images_path[index % self.num_instance_images].name[:-4] + '.tif')
        )
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_DEM"] = self.DEM_transforms(instance_DEM).sub(2.5).div(7.5)  # 最大值最小值缩放至-1~1区间
        temp_tensor = example["instance_DEM"]
        example["instance_DEM"] = torch.cat((example["instance_DEM"], temp_tensor), dim=0)
        example["instance_DEM"] = torch.cat((example["instance_DEM"], temp_tensor), dim=0)
        example["file_name"] = self.instance_images_path[index % self.num_instance_images].name[:-4]
        return example


logger = get_logger(__name__)




def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_unet_path",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        required=True,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--instance_image_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_DEM_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance DEM.",
    )
    parser.add_argument(
        "--stochastic_attribute",
        type=str,
        default=None,
        required=False,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--use_xformers", action="store_true", help="Whether or not to use xformers"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()


    return args


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def unfreeze_params(params):
    for param in params:
        param.requires_grad = True


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
    ).to(device)
    if args.pretrained_unet_path is not None:
        print("Loading pretrained UNet")
        unet.load_state_dict(torch.load(args.pretrained_unet_path).state_dict())
        print("Loaded pretrained UNet")
    vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
        )
    vae = vae.to(device)



    noise_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    dataset = DreamBoothTiDataset(
        instance_image_data_root=args.instance_image_data_dir,
        instance_DEM_data_root=args.instance_DEM_data_dir,
        output_dir=args.output_dir,
        size=512,
    )

    def collate_fn(examples):

        pixel_values = [example["instance_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        batch = {

            "pixel_values": pixel_values,
            "DEM_values": torch.stack([example["instance_DEM"] for example in examples]),
            "file_name": [example["file_name"] for example in examples]
        }
        return batch

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
    )


    weight_dtype = torch.float32

    noise_scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = noise_scheduler.timesteps

    print("***** Running Saving *****")



    sub_sub_progress_bar = tqdm(range(0, int(len(dataloader))))
    sub_sub_progress_bar.set_description('Test')

    # test
    with torch.no_grad():

        sub_sub_progress_bar.refresh()
        sub_sub_progress_bar.reset()

        for step, batch in enumerate(dataloader):
            if step % 3!=0:
                sub_sub_progress_bar.update(1)
                continue
            for i in range(5):

                # dem = batch["DEM_values"].to(device, dtype=weight_dtype)
                image = batch["pixel_values"].to(device, dtype=weight_dtype)
                latents_image = vae.encode(image).latent_dist.sample()
                # torch.save(latents_image, os.path.join(args.output_dir, batch["file_name"][0] +'_image.pt'))

                # latents_dem  = vae.encode(dem).latent_dist.sample()
                # torch.save(latents_dem, os.path.join(args.output_dir, batch["file_name"][0] +'_dem.pt'))

                latents_image = latents_image * 0.18215

                # Sample noise that we'll add to the latents
                noisy = torch.randn_like(latents_image).to(device, dtype=weight_dtype)

                for timestep in timesteps:
                    bsz = latents_image.shape[0]
                    # Sample a random timestep for each image
                    t = torch.ones(
                        (bsz,),
                        device=device,
                    ).mul(timestep)

                    t = t.long()

                    unet_input_latents = torch.cat((noisy, latents_image), dim=1)

                    # Predict the noise residual
                    model_pred_noisy = unet(unet_input_latents, t).sample


                    noisy = \
                        noise_scheduler.step(model_pred_noisy, t, noisy, return_dict=False)[0]

                latent_result_without_image = noisy/0.18215

                # # Save the latent tensor
                # if i==0:
                #     torch.save(latents_dem, os.path.join(args.output_dir, batch["file_name"][0] +'_dem.pt'))
                torch.save(latent_result_without_image, os.path.join(args.output_dir, batch["file_name"][0]+'_without_image_'+str(i) + '.pt'))
            sub_sub_progress_bar.update(1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
