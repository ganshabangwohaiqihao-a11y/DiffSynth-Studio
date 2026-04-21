import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from modelscope import dataset_snapshot_download
from PIL import Image
import torch
from datetime import datetime


pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path="/share/home/202230550120/models/Qwen/Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model.safetensors.index.json"),
        ModelConfig(path="/share/home/202230550120/models/Qwen/Qwen-Image-Edit-2511/text_encoder/model.safetensors.index.json"),
        ModelConfig(path="/share/home/202230550120/models/Qwen/Qwen-Image-Edit-2511/vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(path="/share/home/202230550120/models/Qwen/Qwen-Image-Edit-2511/tokenizer/"),
    processor_config=ModelConfig(path="/share/home/202230550120/models/Qwen/Qwen-Image-Edit-2511/processor"),
)

# ============= LoRA 配置 =============
# 指定 LoRA 检查点路径（从训练脚本输出目录中选择）
lora_checkpoint_path = "/share/home/202230550120/DiffSynth-Studio/models/Qwen-Image-train1/Qwen-Image-Edit-2511_lora/epoch-9.safetensors"

# 加载 LoRA 权重
if os.path.exists(lora_checkpoint_path):
    print(f"加载 LoRA 权重: {lora_checkpoint_path}")
    lora_config = ModelConfig(path=lora_checkpoint_path)
    # 将 LoRA 融合到 dit（Diffusion Transformer）中
    pipe.load_lora(pipe.dit, lora_config, alpha=1.0)
    print("✓ LoRA 权重加载完成")
else:
    print(f"⚠ LoRA 检查点不存在: {lora_checkpoint_path}")
    print("将使用基础模型进行推理")

prompt = "我要按照下面要求对我当前的户型图进行拆改：1. 拆除主卧（25.19㎡）内部的隔墙，将原有的两个小空间合并为一个大开间； 2. 拆除次卧（9.57㎡）与客餐厅之间的非承重隔墙，改用透明玻璃隔断或半开放柜体； 3. 封闭原卫生间朝向走道的门洞，在主卧内部新开门洞，实现主卧套房化； 4. 在客餐厅大空间中心区域，围绕结构柱增设一组中岛台面，并对玄关入口处增设 L 型遮挡隔断。"
edit_image = [
    Image.open("/share/home/202230550120/extracted_images_col6/edited_image_3022236.png").convert("RGB"),
]
image = pipe(
    prompt,
    edit_image=edit_image,
    seed=1,
    num_inference_steps=40,
    height=800,
    width=848,
    edit_image_auto_resize=True,
    zero_cond_t=True, # This is a special parameter introduced by Qwen-Image-Edit-2511
)
# 生成带时间戳的文件名
# 输出目录（将推理图像保存在 infer_img 子目录）
output_dir = "/share/home/202230550120/DiffSynth-Studio/examples/qwen_image/model_inference/infer_img"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = os.path.join(output_dir, f"image_out_{timestamp}.jpg")
image.save(output_filename)
print(f"✓ 推理结果已保存: {output_filename}")

# Qwen-Image-Edit-2511 is a multi-image editing model.
# Please use a list to input `edit_image`, even if the input contains only one image.
# edit_image = [Image.open("image.jpg")]
# Please do not input the image directly.
# edit_image = Image.open("image.jpg")
