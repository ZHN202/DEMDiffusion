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
    AutoencoderKL,
    DDPMScheduler,
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


def save_progress(text_encoder, placeholder_token_id, accelerator, args, save_path):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[placeholder_token_id]
    )
    print("Current Learned Embeddings: ", learned_embeds[:4])
    print("saved to ", save_path)
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)


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
        "--min_steps",
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
        "--pretrained_unet_path",
        type=str,
        default=None,
        required=False,
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
        "--output_format",
        type=str,
        choices=["pt", "safe", "both"],
        default="both",
        help="The output format of the model predicitions and checkpoints.",
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
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
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

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate
                * args.gradient_accumulation_steps
                * args.train_batch_size
        )

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
        lr=args.learning_rate,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    train_dataset = DreamBoothTiDataset(
        instance_image_data_root=args.instance_image_data_dir,
        instance_DEM_data_root=args.instance_DEM_data_dir,
        size=args.resolution,
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

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )


    weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    # vae.to(accelerator.device, dtype=weight_dtype)
    #
    # unet.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)



    # Train!
    total_batch_size = (
            args.train_batch_size
            * args.gradient_accumulation_steps
    )

    noise_scheduler.set_timesteps(args.min_steps, device=device)

    print("***** Running trainin *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )

    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps)
    )
    progress_bar.set_description("Steps")

    sub_sub_progress_bar = tqdm(range(0, int(len(test_dataloader))))
    sub_sub_progress_bar.set_description('Test')

    global_step = 0
    last_save = 0

    for epoch in range(args.num_train_epochs):
        unet.train()
        vae.eval()

        for step, batch in enumerate(train_dataloader):

            image = batch["pixel_values"].to(device=device)
            dem = batch["DEM_values"].to(device=device)

            # Convert images to latent space
            latents_image = vae.encode(
                image
            ).latent_dist.sample()

            # 尺度因子确保扩散模型运行的初始潜空间(不同编码器)具有近似的单位方差
            latents_image = latents_image * 0.18215

            # Convert DEM to latent space
            latents_dem = vae.encode(
                dem
            ).latent_dist.sample()
            latents_dem = latents_dem * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents_dem)
            bsz = latents_dem.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents_image.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents_dem, noise, timesteps)

            unet_input_latents = torch.cat((noisy_latents, latents_image), dim=1)

            model_pred_noisy = unet(unet_input_latents, timesteps).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents_dem, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            latent_loss = F.mse_loss(model_pred_noisy.float(), target.float(), reduction="mean")
            loss = latent_loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            global_step += 1
            progress_bar.update(1)

            # Checks if the accelerator has performed an optimization step behind the scenes

            if args.save_steps and global_step - last_save >= args.save_steps:
                torch.save(unet, os.path.join(args.output_dir, str(global_step)+"_unet_weight.pt"))

                last_save = global_step

            logs = {
                "loss": loss.mean().detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            writer.add_scalars("Loss", {"loss": loss.mean().detach().item(),
                                        }, global_step)
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        with torch.no_grad():
            unet.eval()
            dem_loss = []
            sub_sub_progress_bar.refresh()
            sub_sub_progress_bar.reset()

            for step, batch in enumerate(test_dataloader):
                if step%500!=0:
                    continue

                image = batch["pixel_values"].to(device=device)
                dem = batch["DEM_values"].to(device=device)

                # Convert images to latent space
                latents_image = vae.encode(
                    image
                ).latent_dist.sample()
                latents_image = latents_image * 0.18215
                # Sample noise that we'll add to the latents
                noisy_latents = torch.randn_like(latents_image)

                for timestep in range(noise_scheduler.config.num_train_timesteps - 1, 0,
                                      -int(noise_scheduler.config.num_train_timesteps / args.min_steps)):
                    bsz = latents_image.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.ones(
                        (bsz,),
                        device=latents_image.device,
                    ).mul(timestep)

                    timesteps = timesteps.long()

                    unet_input_latents = torch.cat((noisy_latents, latents_image), dim=1)

                    # Predict the noise residual
                    model_pred_noisy = unet(unet_input_latents, timesteps).sample

                    noisy_latents = \
                        noise_scheduler.step(model_pred_noisy, timesteps, noisy_latents, return_dict=False)[0]

                dem_pred = vae.decode(noisy_latents/0.18215)[0]
                loss_dem_pred = F.mse_loss(dem_pred.float().mul(7.5),
                                           dem.float().mul(7.5),
                                           reduction="mean")
                dem_loss.append(loss_dem_pred.mean().detach().item())
                sub_sub_progress_bar.update(1)
                logs = {
                    "gen_dem_loss": loss_dem_pred.mean().detach().item(),
                }
                sub_sub_progress_bar.set_postfix(**logs)

            writer.add_scalars("Test_Loss",
                               {"gen_loss": sum(dem_loss) / len(dem_loss)},
                               global_step)  # Get the target for loss depending on the prediction type



if __name__ == "__main__":
    args = parse_args()
    main(args)
