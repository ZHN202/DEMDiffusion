#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_TEST_DEM_DIR="/home/vipuser/data/mix_testB"
export INSTANCE_TEST_IMAGE_DIR="/home/vipuser/data/mix_testA"
export UNET_MODEL_PATH="/home/vipuser/code/lora-master/training_scripts/output_sd_mix_3/31680_unet_weight.pt"
export OUTPUT_DIR="./test_sd_4"

accelerate launch test_my_DEM_SD.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_DEM_test_data_dir=$INSTANCE_TEST_DEM_DIR \
  --instance_image_test_data_dir=$INSTANCE_TEST_IMAGE_DIR \
  --output_dir=$OUTPUT_DIR \
  --pretrained_unet_path=$UNET_MODEL_PATH \
  --num_inference_steps=50