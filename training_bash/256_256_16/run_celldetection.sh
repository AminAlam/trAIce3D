# !/bin/bash
NO_GPUS=$1
NO_NODES=$2
NODE_RANK=$3
MASTER_ADDR=$4
MASTER_PORT=$5

export NEPTUNE_API_TOKEN=$7
neptune_project="CellSegmentation/BranchSegmentation"

# model parameters
image_input_size=256_256_16
encoder_patch_size=8_8_2
encoder_emb_dim=128 # should be image_input_size[0]//8 * image_input_size[1]//8 * image_input_size[2]//8
encoder_depth=32
encoder_mlp_ratio=4
skip_connection_linear_proj_dim=512
residual_block_num_conv_layers=1
encoder_transformer_num_heads=8
decoder_up_sampling_dim=128
model_name="UNet_PromptDriven"
freeze_encoder=False
freeze_decoder=False
freeze_skip_connection_block=False
freeze_prompt_encoder=False

# fine-tuning parameters
encoder_load_path=None
decoder_load_path=None
prompt_encoder_load_path=None
optimizer_load_path=None
skip_connection_block_load_path=None
scaler_load_path=None
epoch_start=0

# training parameters
batch_size=56
num_epochs=5000
learning_rate_start=1e-3
learning_rate_end=1e-6
lr_factor=0.95
patience_epochs=2
accumulation_steps=2
L1_weight=5e-6
L2_weight=1e-2

# paths and data
model_save_path="./models"
datasets_path_train="./data"
transform_bool=True
train_test_ratio=0.9
load2ram=False

# save and test parameters
smaple_test_image_freq=2
test_result_save_freq=10
save_model_freq=2

# device 
device="cuda"

# create model save path
if [ ! -d $model_save_path ]; then
    mkdir -p $model_save_path
fi

# run training
torchrun --nproc_per_node=$NO_GPUS --nnodes=$NO_NODES --node_rank=$NODE_RANK --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
train.py --image_input_size $image_input_size \
--encoder_patch_size $encoder_patch_size \
--encoder_emb_dim $encoder_emb_dim \
--encoder_depth $encoder_depth \
--encoder_mlp_ratio $encoder_mlp_ratio \
--encoder_transformer_num_heads $encoder_transformer_num_heads \
--skip_connection_linear_proj_dim $skip_connection_linear_proj_dim \
--residual_block_num_conv_layers $residual_block_num_conv_layers \
--decoder_up_sampling_dim $decoder_up_sampling_dim \
--model_name $model_name \
--epoch_start $epoch_start \
--batch_size $batch_size \
--num_epochs $num_epochs \
--learning_rate_start $learning_rate_start \
--learning_rate_end $learning_rate_end \
--lr_factor $lr_factor \
--patience_epochs $patience_epochs \
--accumulation_steps $accumulation_steps \
--model_save_path $model_save_path \
--datasets_path_train $datasets_path_train \
--transform_bool $transform_bool \
--train_test_ratio $train_test_ratio \
--load2ram $load2ram \
--smaple_test_image_freq $smaple_test_image_freq \
--test_result_save_freq $test_result_save_freq \
--save_model_freq $save_model_freq \
--device $device \
--neptune_project $neptune_project \
--l1_weight $L1_weight \
--l2_weight $L2_weight \
train_cell_segmentation \
--encoder_load_path $encoder_load_path \
--decoder_load_path $decoder_load_path \
--optimizer_load_path $optimizer_load_path \
--skip_connection_block_load_path $skip_connection_block_load_path \
--scaler_load_path $scaler_load_path \
--freeze_encoder $freeze_encoder \
--freeze_skip_connection_block $freeze_skip_connection_block \
--freeze_decoder $freeze_decoder \
--freeze_prompt_encoder $freeze_prompt_encoder