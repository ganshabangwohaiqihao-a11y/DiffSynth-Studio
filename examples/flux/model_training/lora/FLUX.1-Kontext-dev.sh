cd /share/home/202230550120/DiffSynth-Studio

accelerate launch examples/flux/model_training/train.py \
  --dataset_base_path /share/home/202230550120/DiffSynth-Studio/data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2511 \
  --dataset_metadata_path /share/home/202230550120/DiffSynth-Studio/data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2511/metadata1.csv \
  --data_file_keys "image,edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 20\
  --model_paths '["/share/home/202230550120/models/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors", "/share/home/202230550120/models/FLUX.1-dev/text_encoder/model.safetensors", ["/share/home/202230550120/models/FLUX.1-dev/text_encoder_2/model-00001-of-00002.safetensors", "/share/home/202230550120/models/FLUX.1-dev/text_encoder_2/model-00002-of-00002.safetensors"], "/share/home/202230550120/models/FLUX.1-dev/ae.safetensors"]' \
  --tokenizer_1_path /share/home/202230550120/models/FLUX.1-dev/tokenizer/ \
  --tokenizer_2_path /share/home/202230550120/models/FLUX.1-dev/tokenizer_2/ \
  --learning_rate 1e-4 \
  --num_epochs 10\
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train12/FLUX.1-Kontext-dev_lora" \
  --tensorboard_log_dir "./models/train12/FLUX.1-Kontext-dev_lora/tensorboard" \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank 32 \
  --align_to_opensource_format \
  --extra_inputs "edit_image" \
  --use_gradient_checkpointing