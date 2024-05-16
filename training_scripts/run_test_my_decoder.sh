#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_TEST_IMAGE_DIR="/home/vipuser/data/testA"
export INSTANCE_TEST_DEM_DIR="/home/vipuser/data/testB"
export INSTANCE_TEST_LATENT_DIR="/home/vipuser/code/lora-master/training_scripts/tensors_test_100_moon"
export DECODER_MODEL_PATH="/home/vipuser/code/lora-master/training_scripts/output_de_without_6/4590_decoder_weight.pt"
export OUTPUT_DIR="./test_moon"
#  --pretrained_decoder_path=$DECODER_MODEL_PATH \
accelerate launch test_my_decoder.py \
  --pretrained_decoder_path=$DECODER_MODEL_PATH \
  --instance_DEM_test_data_dir=$INSTANCE_TEST_DEM_DIR \
  --instance_latent_test_data_dir=$INSTANCE_TEST_LATENT_DIR \
  --instance_image_test_data_dir=$INSTANCE_TEST_IMAGE_DIR \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \

