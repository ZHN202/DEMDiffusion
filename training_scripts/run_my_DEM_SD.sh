#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_IMAGE_DIR="/home/vipuser/data/mix_trainA"
export INSTANCE_DEM_DIR="/home/vipuser/data/mix_trainB"
export INSTANCE_TEST_DEM_DIR="/home/vipuser/data/mix_testB"
export INSTANCE_TEST_IMAGE_DIR="/home/vipuser/data/mix_testA"
export UNET_MODEL_PATH="/home/vipuser/code/lora-master/training_scripts/output_sd_mix_2/65000_unet_weight.pt"
export OUTPUT_DIR="./output_sd_mix_3"

accelerate launch train_my_DEM_SD.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --pretrained_unet_path=$UNET_MODEL_PATH \
  --instance_image_data_dir=$INSTANCE_IMAGE_DIR \
  --instance_DEM_data_dir=$INSTANCE_DEM_DIR \
  --instance_DEM_test_data_dir=$INSTANCE_TEST_DEM_DIR \
  --instance_image_test_data_dir=$INSTANCE_TEST_IMAGE_DIR \
  --output_dir=$OUTPUT_DIR \
  --min_steps=50 \
  --output_format="safe" \
  --resolution=512 \
  --train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-5 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=500 \
  --max_train_steps=1000000 \
  --save_steps=15840