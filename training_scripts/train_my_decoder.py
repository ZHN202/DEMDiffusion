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
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from decoder.decoder import DEMDecoder, ConditionalDecoder
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from colorize_dem import colorize_depth_maps
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
            instance_latent_data_root,
            instance_DEM_data_root,
            instance_image_data_root,
            size=512,
    ):
        self.size = size

        self.instance_latent_data_root = Path(instance_latent_data_root)
        self.instance_DEM_data_root = Path(instance_DEM_data_root)
        self.instance_image_data_root = Path(instance_image_data_root)
        if not self.instance_latent_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        if not self.instance_DEM_data_root.exists():
            raise ValueError("Instance DEM root doesn't exists.")
        self.instance_latent_path_ = list(Path(instance_latent_data_root).iterdir())
        self.instance_DEM_path = list(Path(instance_DEM_data_root).iterdir())

        self.instance_latent_path = [x for x in self.instance_latent_path_ if x.name[-4]=='m' or x.name[-6]=='e']
        print(len(self.instance_latent_path))

        self.num_instance_latent = len(self.instance_latent_path)

        self._length = self.num_instance_latent
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
        instance_latent = torch.load(
            self.instance_latent_path[index % self.num_instance_latent],map_location='cpu'
        )
        if self.instance_latent_path[index % self.num_instance_latent].name[-4]=='m':
            file_name = self.instance_latent_path[index % self.num_instance_latent].name[:-7]
        elif self.instance_latent_path[index % self.num_instance_latent].name[-6]=='e':
            file_name = self.instance_latent_path[index % self.num_instance_latent].name[:-19]
        else:
            file_name = self.instance_latent_path[index % self.num_instance_latent].name[:-3]
        instance_DEM = Image.open(
            self.instance_DEM_data_root.joinpath(
                file_name + '.tif')
        )
        instance_image = Image.open(
            self.instance_image_data_root.joinpath(
                file_name + '.png')
        )
        if instance_image.mode!='RGB':
            instance_image = instance_image.convert('RGB')
        example["instance_image"] = self.image_transforms(instance_image)
        example["instance_latent"] = instance_latent
        example["instance_DEM"] = self.DEM_transforms(instance_DEM).add(5).div(15)  # 最大值最小值缩放至0~1区间

        return example


logger = get_logger(__name__)



def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--instance_latent_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--pretrained_decoder_path",
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
        "--instance_image_data_dir",
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
        "--instance_latent_test_data_dir",
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

    device = torch.device( "cuda:0" if torch.cuda.is_available() else  "cpu")

    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # decoder = DEMDecoder2(
    #     in_channels=8,
    #     out_channels=1,
    #     up_block_types=[
    #         "UpDecoderBlock2D",
    #         "UpDecoderBlock2D",
    #         "UpDecoderBlock2D",
    #         "UpDecoderBlock2D"
    #     ],
    #     block_out_channels=[
    #         128,
    #         256,
    #         512,
    #         512
    #     ],
    #     layers_per_block=2,
    #     norm_num_groups=32,
    #     act_fn="silu",
    # ).to(device)


    # decoder = DEMDecoder(
    #     in_channels=4,
    #     out_channels=1,
    #     up_block_types=[
    #         "UpDecoderBlock2D",
    #         "UpDecoderBlock2D",
    #         "UpDecoderBlock2D",
    #         "UpDecoderBlock2D"
    #     ],
    #     block_out_channels=[
    #         512,
    #         512,
    #         256,
    #         128
    #     ],
    #     layers_per_block=1,
    #     norm_num_groups=32,
    #     act_fn="silu",
    # ).to(device)

    decoder = ConditionalDecoder(
        in_channels=4,
        out_channels=1,
        block_out_channels=[
            512,
            256,
            128
        ],
        norm_num_groups=32,
    ).to(device)


    if args.pretrained_decoder_path is not None:
        print("Loading pretrained decoder")
        decoder.load_state_dict(torch.load(args.pretrained_decoder_path).state_dict())
        print("Pretrained decoder loaded")

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate
                * args.gradient_accumulation_steps
                * args.train_batch_size
        )

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
        lr=args.learning_rate,
    )


    train_dataset = DreamBoothTiDataset(
        instance_latent_data_root=args.instance_latent_data_dir,
        instance_DEM_data_root=args.instance_DEM_data_dir,
        instance_image_data_root=args.instance_image_data_dir,
        size=args.resolution,
    )
    test_dataset = DreamBoothTiDataset(
        instance_latent_data_root=args.instance_latent_test_data_dir,
        instance_DEM_data_root=args.instance_DEM_test_data_dir,
        instance_image_data_root=args.instance_image_test_data_dir,
        size=args.resolution,
    )

    def collate_fn(examples):

        latent_values = [example["instance_latent"][0] for example in examples]

        latent_values = torch.stack(latent_values)
        latent_values = latent_values.to(memory_format=torch.contiguous_format).float()

        batch = {
            "image_values": torch.stack([example["instance_image"] for example in examples]),
            "latent_values": latent_values,
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
        batch_size=args.train_batch_size,
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
        num_cycles=5,
    )


    weight_dtype = torch.float32

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # Train!
    total_batch_size = (
            args.train_batch_size
            * args.gradient_accumulation_steps
    )

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
        decoder.train()

        for step, batch in enumerate(train_dataloader):
            progress_bar.update(1)

            latent = batch["latent_values"].to(device)
            target = batch["DEM_values"].to(device)
            image = batch["image_values"].to(device)


            pred_dem = decoder(latent,image)
            pred_dem = torch.clip(pred_dem, 0, 1)

            # loss = F.smooth_l1_loss(pred_dem.float(), target.float(),beta=0.1, reduction="mean")#.requires_grad_(True)
            loss = F.huber_loss(pred_dem.float().mul(15),target.float().mul(15),reduction="mean",delta=0.5)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            global_step += 1


            # Checks if the accelerator has performed an optimization step behind the scenes

            if args.save_steps and global_step - last_save >= args.save_steps:
                torch.save(decoder, os.path.join(args.output_dir, str(global_step)+"_decoder_weight.pt"))
                last_save = global_step

            logs = {
                    "loss": loss.mean().detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
            writer.add_scalars("Train", logs, global_step)
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        torch.save(decoder, os.path.join(args.output_dir, str(global_step)+"_decoder_weight.pt"))
        with torch.no_grad():
            decoder.eval()
            dem_loss = []
            sub_sub_progress_bar.refresh()
            sub_sub_progress_bar.reset()

            for step, batch in enumerate(test_dataloader):
                latent = batch["latent_values"].to(device)
                target = batch["DEM_values"].to(device)
                image = batch["image_values"].to(device)
                pred_dem = decoder(latent,image)
                pred_dem = torch.clip(pred_dem,0,1)

                loss_dem_pred = F.smooth_l1_loss(pred_dem.float().mul(15),
                                           target.to(dtype=weight_dtype).float().mul(15),
                                                 beta=0,
                                           reduction="mean")
                dem_loss.append(loss_dem_pred.mean().detach().item())
                sub_sub_progress_bar.update(1)
                logs = {
                    "gen_dem_loss": loss_dem_pred.mean().detach().item(),
                }
                sub_sub_progress_bar.set_postfix(**logs)

                colored_dem_pred = colorize_depth_maps(pred_dem[0].cpu(), 0, 1).squeeze()
                colored_dem = colorize_depth_maps(target[0].cpu(),0,1).squeeze()
                writer.add_image("Test/Image", image[0].cpu(), step)
                writer.add_image("Test/Pred_DEM", colored_dem_pred, step)
                writer.add_image("Test/DEM", colored_dem, step)
            writer.add_scalars("Test_Loss",
                               {"gen_loss": sum(dem_loss) / len(dem_loss)},
                               global_step)  # Get the target for loss depending on the prediction type




if __name__ == "__main__":
    args = parse_args()
    main(args)
