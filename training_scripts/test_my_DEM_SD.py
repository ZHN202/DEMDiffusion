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
from colorize_dem import colorize_depth_maps
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DModel,
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
            size=512,
    ):
        self.size = size

        self.instance_image_data_root = Path(instance_image_data_root)
        self.instance_DEM_data_root = Path(instance_DEM_data_root)
        if not self.instance_image_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        if not self.instance_DEM_data_root.exists():
            raise ValueError("Instance DEM root doesn't exists.")
        self.instance_images_path = list(Path(instance_image_data_root).iterdir())
        self.instance_DEM_path = list(Path(instance_DEM_data_root).iterdir())
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
        "--pretrained_unet_path",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_DEM_test_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance DEM.",
    )
    parser.add_argument(
        "--instance_image_test_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance DEM.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
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
    logging_dir = Path(args.output_dir, args.logging_dir)

    writer = SummaryWriter(log_dir=args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    ).to(device)

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
        print("Loading pretrained unet")
        unet.load_state_dict(torch.load(args.pretrained_unet_path).state_dict())
        print("Loaded pretrained unet")
    vae.requires_grad_(False)
    unet.requires_grad_(False)


    noise_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    test_dataset = DreamBoothTiDataset(
        instance_image_data_root=args.instance_image_test_data_dir,
        instance_DEM_data_root=args.instance_DEM_test_data_dir,
        size=args.resolution,
    )

    def collate_fn(examples):

        pixel_values = [example["instance_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        batch = {

            "pixel_values": pixel_values,
            "DEM_values": torch.stack([example["instance_DEM"] for example in examples])
        }
        return batch
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
    )





    # Train!

    noise_scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = noise_scheduler.timesteps

    print("***** Running trainin *****")

    sub_sub_progress_bar = tqdm(range(0, int(len(test_dataloader))))
    sub_sub_progress_bar.set_description('Test')


    with torch.no_grad():
        unet.eval()
        dem_loss = []
        sub_sub_progress_bar.refresh()
        sub_sub_progress_bar.reset()

        for step, batch in enumerate(test_dataloader):
            image = batch["pixel_values"].to(device=device)
            dem = batch["DEM_values"].to(device=device)

            # Convert images to latent space
            latents_image = vae.encode(
                image
            ).latent_dist.sample()
            latents_image = latents_image * 0.18215
            # Sample noise that we'll add to the latents
            noisy_latents = torch.randn_like(latents_image)

            for timestep in timesteps:

                bsz = latents_image.shape[0]
                # Sample a random timestep for each image
                t = torch.ones(
                    (bsz,),
                    device=latents_image.device,
                ).mul(timestep)

                t = t.long()

                unet_input_latents = torch.cat((noisy_latents, latents_image), dim=1)

                # Predict the noise residual
                model_pred_noisy = unet(unet_input_latents, t).sample

                noisy_latents = \
                    noise_scheduler.step(model_pred_noisy, t, noisy_latents, return_dict=False)[0]
                if step==0:
                    dem_pred_ = vae.decode(noisy_latents/0.18215)[0].mean(dim=1, keepdim=True)

                    colored_dem = colorize_depth_maps(dem_pred_.cpu(),-1,1).squeeze()
                    writer.add_image("Test/Gen_DEM_pro", colored_dem, 1000-timestep)

            dem_pred = vae.decode(noisy_latents/0.18215)[0].mean(dim=1, keepdim=True)
            dem_pred = torch.clip(dem_pred, -1, 1)


            loss_dem_pred = F.smooth_l1_loss(dem_pred.float().mul(7.5),
                                       dem.mean(dim=1, keepdim=True).float().mul(7.5),beta=0,
                                       reduction="mean")

            sub_sub_progress_bar.update(1)
            logs = {
                "gen_dem_loss": loss_dem_pred.mean().detach().item(),
            }
            sub_sub_progress_bar.set_postfix(**logs)
            colored_dem_pred = colorize_depth_maps(dem_pred.cpu(), -1, 1).squeeze()
            colored_dem = colorize_depth_maps(dem.mean(dim=1, keepdim=True).cpu(),-1,1).squeeze()
            writer.add_image("Test/Image", image[0].cpu(), step)
            writer.add_image("Test/Pred_DEM", colored_dem_pred, step)
            writer.add_image("Test/DEM", colored_dem, step)

            writer.add_scalars("Test_Loss",
                           logs,
                           step)  # Get the target for loss depending on the prediction type



if __name__ == "__main__":
    args = parse_args()
    main(args)
