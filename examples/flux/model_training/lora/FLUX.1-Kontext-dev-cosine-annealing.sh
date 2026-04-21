cd /share/home/202230550120/DiffSynth-Studio

accelerate launch examples/flux/model_training/train.py \
  --dataset_base_path /share/home/202230550120/diffusers/output \
  --dataset_metadata_path /share/home/202230550120/diffusers/output/metadata_kontext_train_500items_without_furn.csv\
  --data_file_keys "image,kontext_images" \
  --max_pixels 1048576 \
  --dataset_repeat 2\
  --model_paths '["/share/home/202230550120/models/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors", "/share/home/202230550120/models/FLUX.1-dev/text_encoder/model.safetensors", ["/share/home/202230550120/models/FLUX.1-dev/text_encoder_2/model-00001-of-00002.safetensors", "/share/home/202230550120/models/FLUX.1-dev/text_encoder_2/model-00002-of-00002.safetensors"], "/share/home/202230550120/models/FLUX.1-dev/ae.safetensors"]' \
  --tokenizer_1_path /share/home/202230550120/models/FLUX.1-dev/tokenizer/ \
  --tokenizer_2_path /share/home/202230550120/models/FLUX.1-dev/tokenizer_2/ \
  --learning_rate 1e-4 \
  --num_epochs 10\
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train8.3.26.21:50/FLUX.1-Kontext-dev_lora_cosine_annealing_without_furn" \
  --tensorboard_log_dir "./models/train8.3.26.21:50/FLUX.1-Kontext-dev_lora_cosine_annealing_without_furn/tensorboard" \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank 32 \
  --align_to_opensource_format \
  --extra_inputs "kontext_images" \
  --use_gradient_checkpointing \
  --lr_scheduler_type "cosine" \
  --lr_scheduler_kwargs '{"T_max": 16870, "eta_min": 1e-5}'