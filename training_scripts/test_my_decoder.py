
import sys
import os

import cv2

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import error
import argparse
import os
import rasterio
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
import torch.utils.checkpoint
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from decoder.decoder import DEMDecoder, ConditionalDecoder
from tqdm.auto import tqdm
from colorize_dem import colorize_depth_maps
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

        self.instance_latent_path = [x for x in self.instance_latent_path_ if x.name[-6]=='e']
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
        )[0]
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
        example["file_name"] = file_name
        example["instance_image"] = self.image_transforms(instance_image)
        example["instance_latent"] = instance_latent
        example["instance_DEM"] = self.DEM_transforms(instance_DEM).add(5).div(15)  # 最大值最小值缩放至0~1区间

        return example


logger = get_logger(__name__)



def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_decoder_path",
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

    vae_decoder = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5",subfolder='vae').to(device)


    if args.pretrained_decoder_path is not None:
        print("Loading pretrained decoder")
        decoder.load_state_dict(torch.load(args.pretrained_decoder_path).state_dict())
        print("Pretrained decoder loaded")


    test_dataset = DreamBoothTiDataset(
        instance_latent_data_root=args.instance_latent_test_data_dir,
        instance_DEM_data_root=args.instance_DEM_test_data_dir,
        instance_image_data_root=args.instance_image_test_data_dir,
        size=args.resolution,
    )

    def collate_fn(examples):

        latent_values = [example["instance_latent"] for example in examples]

        latent_values = torch.stack(latent_values)
        latent_values = latent_values.to(memory_format=torch.contiguous_format).float()

        batch = {
            "image_values": torch.stack([example["instance_image"] for example in examples]),
            "file_names": [example["file_name"] for example in examples],  # "file_name
            "latent_values": latent_values,
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



    weight_dtype = torch.float32


    print("***** Running testing *****")


    # Only show the progress bar once on each machine.

    sub_sub_progress_bar = tqdm(range(0, int(len(test_dataloader))))
    sub_sub_progress_bar.set_description('Test')

    global_step = 0
    last_save = 0

    # with open(os.path.join(args.output_dir, "prompt.txt"), "w") as f:

    with torch.no_grad():
        decoder.eval()
        vae_decoder.eval()
        mars_mae_cd = []
        mars_mae_vae = []
        moon_mae_cd = []
        moon_mae_vae = []
        mars_rmse_cd = []
        mars_rmse_vae = []
        moon_rmse_cd = []
        moon_rmse_vae = []
        mars_r2_cd = []
        mars_r2_vae = []
        moon_r2_cd = []
        moon_r2_vae = []
        mars_delta1_cd = []
        mars_delta1_vae = []
        moon_delta1_cd = []
        moon_delta1_vae = []

        for step, batch in enumerate(test_dataloader):
            latent = batch["latent_values"].to(device)
            target = batch["DEM_values"].to(device)
            image = batch["image_values"].to(device)

            # Conditional Decoder
            pred_dem = decoder(latent,image)
            pred_dem = torch.clip(pred_dem,0,1)

            # VAE Decoder
            pred_dem_vae = vae_decoder.decode(latent)[0].mean(dim=1, keepdim=True)
            pred_dem_vae = torch.clip(pred_dem_vae, -1, 1).add(1).div(2)


            if batch["file_names"][0][0] == 'm':
                mae = error.mae(pred_dem.float().mul(15), target.float().mul(15))
                mae_vae = error.mae(pred_dem_vae.float().mul(15), target.float().mul(15))
                rmse = error.rmse(pred_dem.float().mul(15), target.float().mul(15))
                rmse_vae = error.rmse(pred_dem_vae.float().mul(15), target.float().mul(15))
                r2 = error.r2_score(pred_dem.float().mul(15), target.float().mul(15))
                r2_vae = error.r2_score(pred_dem_vae.float().mul(15), target.float().mul(15))
                detla1_cd = error.delta1_accuracy(pred_dem.float().mul(15), target.float().mul(15))
                detla1_vae = error.delta1_accuracy(pred_dem_vae.float().mul(15), target.float().mul(15))
                mars_delta1_cd.append(detla1_cd)
                mars_delta1_vae.append(detla1_vae)

                mars_mae_cd.append(mae)
                mars_mae_vae.append(mae_vae)
                mars_rmse_cd.append(rmse)
                mars_rmse_vae.append(rmse_vae)
                if r2>0:
                    mars_r2_cd.append(r2)
                if r2_vae>0:
                    mars_r2_vae.append(r2_vae)
            else:
                mae = error.mae(pred_dem.float().mul(15), target.float().mul(15))
                mae_vae = error.mae(pred_dem_vae.float().mul(15), target.float().mul(15))
                rmse = error.rmse(pred_dem.float().mul(15), target.float().mul(15))
                rmse_vae = error.rmse(pred_dem_vae.float().mul(15), target.float().mul(15))
                r2 = error.r2_score(pred_dem.float().mul(15), target.float().mul(15))
                r2_vae = error.r2_score(pred_dem_vae.float().mul(15), target.float().mul(15))
                detla1_cd = error.delta1_accuracy(pred_dem.float().mul(15), target.float().mul(15))
                detla1_vae = error.delta1_accuracy(pred_dem_vae.float().mul(15), target.float().mul(15))
                moon_delta1_cd.append(detla1_cd)
                moon_delta1_vae.append(detla1_vae)
                moon_mae_cd.append(mae)
                moon_mae_vae.append(mae_vae)
                moon_rmse_cd.append(rmse)
                moon_rmse_vae.append(rmse_vae)
                if r2>0:
                    moon_r2_cd.append(r2)
                if r2_vae>0:
                    moon_r2_vae.append(r2_vae)




            sub_sub_progress_bar.update(1)
            logs = {
                "Conditional Decoder MAE": mae.mean().detach().item() ,
                # 保留两位小鼠
                "VAE Decoder MAE": mae_vae.mean().detach().item(),
                "Conditional Decoder RMSE": rmse.mean().detach().item(),
                "VAE Decoder RMSE": rmse_vae.mean().detach().item(),
                "Conditional Decoder R2": r2,
                "VAE Decoder R2": r2_vae,
                "Conditional Decoder Delta1": detla1_cd,
                "VAE Decoder Delta1": detla1_vae
            }
            sub_sub_progress_bar.set_postfix(**logs)


            colored_dem_pred = colorize_depth_maps(pred_dem[0].cpu(), 0, 1).squeeze()
            colored_dem_pred_vae = colorize_depth_maps(pred_dem_vae[0].cpu(), 0, 1).squeeze()
            colored_dem = colorize_depth_maps(target[0].cpu(), 0, 1).squeeze()
            writer.add_image("Test/2_Conditional Decoder Pred DEM", colored_dem_pred, step)
            writer.add_image("Test/3_VAE Decoder Pred DEM", colored_dem_pred_vae, step)
            writer.add_image("Test/1_DEM", colored_dem, step)
            writer.add_image("Test/4_Image", image[0].cpu(), step)

            writer.add_scalars("Test_Loss",
                           logs,
                           step)  # Get the target for loss depending on the prediction type
            # 保存预测的DEM
            pred_dem = pred_dem[0].mul(15).sub(5).cpu().numpy()
            pred_dem_vae = pred_dem_vae[0].mul(15).sub(5).cpu().numpy()
            # 保存为tif格式
            file_name = batch["file_names"][0]
            with rasterio.open(os.path.join(args.output_dir, file_name +"_"+ str(mae.detach().item())[:5]+"_pred_cd.tif"),
                               'w',width=512,height=512,driver='GTiff',count=1,dtype="float32",
                               nodata=0.0,crs= rasterio.crs.CRS.from_epsg(4326),transform=rasterio.transform.from_origin(0,0,0.125,0.125)

                               ) as dst:
                dst.write(pred_dem)

            with rasterio.open(os.path.join(args.output_dir, file_name + "_"+ str(mae_vae.detach().item())[:5]+"_pred_vae.tif"),
                               'w',width=512,height=512,driver='GTiff',count=1,dtype="float32",
                                nodata=0.0,crs= rasterio.crs.CRS.from_epsg(4326),transform=rasterio.transform.from_origin(0,0,0.125,0.125)
                               ) as dst:
                dst.write(pred_dem_vae)




        if len(mars_mae_cd)>0:
            mean_delta1_cd_mars = sum(mars_delta1_cd) / len(mars_delta1_cd)
            mean_delta1_vae_mars = sum(mars_delta1_vae) / len(mars_delta1_vae)
            mean_mae_cd_mars = sum(mars_mae_cd) / len(mars_mae_cd)
            mean_mae_vae_mars = sum(mars_mae_vae) / len(mars_mae_vae)
            mean_rmse_cd_mars = sum(mars_rmse_cd) / len(mars_rmse_cd)
            mean_rmse_vae_mars = sum(mars_rmse_vae) / len(mars_rmse_vae)
            mean_r2_cd_mars = sum(mars_r2_cd) / len(mars_r2_cd)
            mean_r2_vae_mars = sum(mars_r2_vae) / len(mars_r2_vae)
            writer.add_text("Test/5_Mean MAE CD Mars", str(mean_mae_cd_mars), global_step
                            )
            writer.add_text("Test/6_Mean MAE VAE Mars", str(mean_mae_vae_mars), global_step
                            )
            writer.add_text("Test/7_Mean RMSE CD Mars", str(mean_rmse_cd_mars), global_step
                            )
            writer.add_text("Test/8_Mean RMSE VAE Mars", str(mean_rmse_vae_mars), global_step
                            )
            writer.add_text("Test/9_Mean R2 CD Mars", str(mean_r2_cd_mars), global_step
                            )
            writer.add_text("Test/10_Mean R2 VAE Mars", str(mean_r2_vae_mars), global_step
                            )
            writer.add_text("Test/17_Mean Delta1 CD Mars", str(mean_delta1_cd_mars), global_step
                            )
            writer.add_text("Test/18_Mean Delta1 VAE Mars", str(mean_delta1_vae_mars), global_step
                            )

        if len(moon_mae_cd)>0:
            mean_mae_cd_moon = sum(moon_mae_cd) / len(moon_mae_cd)
            mean_mae_vae_moon = sum(moon_mae_vae) / len(moon_mae_vae)
            mean_rmse_cd_moon = sum(moon_rmse_cd) / len(moon_rmse_cd)
            mean_rmse_vae_moon = sum(moon_rmse_vae) / len(moon_rmse_vae)
            mean_r2_cd_moon = sum(moon_r2_cd) / len(moon_r2_cd)
            mean_r2_vae_moon = sum(moon_r2_vae) / len(moon_r2_vae)
            mean_delta1_cd_moon = sum(moon_delta1_cd) / len(moon_delta1_cd)
            mean_delta1_vae_moon = sum(moon_delta1_vae) / len(moon_delta1_vae)

            writer.add_text("Test/11_Mean MAE CD Moon", str(mean_mae_cd_moon), global_step
                            )
            writer.add_text("Test/12_Mean MAE VAE Moon", str(mean_mae_vae_moon), global_step
                            )
            writer.add_text("Test/13_Mean RMSE CD Moon", str(mean_rmse_cd_moon), global_step
                            )
            writer.add_text("Test/14_Mean RMSE VAE Moon", str(mean_rmse_vae_moon), global_step
                            )
            writer.add_text("Test/15_Mean R2 CD Moon", str(mean_r2_cd_moon), global_step
                            )
            writer.add_text("Test/16_Mean R2 VAE Moon", str(mean_r2_vae_moon), global_step
                            )

            writer.add_text("Test/19_Mean Delta1 CD Moon", str(mean_delta1_cd_moon), global_step
                            )
            writer.add_text("Test/20_Mean Delta1 VAE Moon", str(mean_delta1_vae_moon), global_step
                            )






if __name__ == "__main__":
    args = parse_args()
    main(args)
