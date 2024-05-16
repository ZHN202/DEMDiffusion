#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_LATENT_DIR="/home/vipuser/code/lora-master/training_scripts/tensors_sd_3_step_50"
export INSTANCE_DEM_DIR="/home/vipuser/data/mix_trainB"
export INSTANCE_IMAGE_DIR="/home/vipuser/data/mix_trainA"
export INSTANCE_TEST_IMAGE_DIR="/home/vipuser/data/mix_testA"
export INSTANCE_TEST_DEM_DIR="/home/vipuser/data/mix_testB"
export INSTANCE_TEST_LATENT_DIR="/home/vipuser/code/lora-master/training_scripts/tensors_sd_3_step_50_test"
export DECODER_MODEL_PATH="/home/vipuser/code/lora-master/training_scripts/output_de_without_5/2980_decoder_weight.pt"
export OUTPUT_DIR="./output_de_without_7"
#  --pretrained_decoder_path=$DECODER_MODEL_PATH \
accelerate launch train_my_decoder.py \
  --instance_latent_data_dir=$INSTANCE_LATENT_DIR \
  --instance_DEM_data_dir=$INSTANCE_DEM_DIR \
  --instance_DEM_test_data_dir=$INSTANCE_TEST_DEM_DIR \
  --instance_latent_test_data_dir=$INSTANCE_TEST_LATENT_DIR \
  --instance_image_data_dir=$INSTANCE_IMAGE_DIR \
  --instance_image_test_data_dir=$INSTANCE_TEST_IMAGE_DIR \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-5 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=500 \
  --max_train_steps=20000 \
  --save_steps=2500