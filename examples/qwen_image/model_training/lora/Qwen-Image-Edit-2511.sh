cd /share/home/202230550120/DiffSynth-Studio

accelerate launch examples/qwen_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2511 \
  --dataset_metadata_path /share/home/202230550120/DiffSynth-Studio/data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2511/metadata_paired_train.json \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 20 \
  --model_paths '["/share/home/202230550120/models/Qwen/Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model.safetensors.index.json","/share/home/202230550120/models/Qwen/Qwen-Image-Edit-2511/text_encoder/model.safetensors.index.json","/share/home/202230550120/models/Qwen/Qwen-Image-Edit-2511/vae/diffusion_pytorch_model.safetensors"]' \
  --tokenizer_path "/share/home/202230550120/models/Qwen/Qwen-Image-Edit-2511/tokenizer" \
  --processor_path "/share/home/202230550120/models/Qwen/Qwen-Image-Edit-2511/processor" \
  --learning_rate 1e-4 \
  --num_epochs 20 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/Qwen-Image-train_2026.04.08.20:02/Qwen-Image-Edit-2511_lora" \
  --tensorboard_log_dir "./models/Qwen-Image-train_2026.04.08.20:02/Qwen-Image-Edit-2511_lora/tensorboard" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --zero_cond_t # This is a special parameter introduced by Qwen-Image-Edit-2511. Please enable it for this model.
