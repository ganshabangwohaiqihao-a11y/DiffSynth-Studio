import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path="/share/home/202230550120/models/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors"),
        ModelConfig(path="/share/home/202230550120/models/FLUX.1-dev/text_encoder/model.safetensors"),
        ModelConfig(path=[
            "/share/home/202230550120/models/FLUX.1-dev/text_encoder_2/model-00001-of-00002.safetensors",
            "/share/home/202230550120/models/FLUX.1-dev/text_encoder_2/model-00002-of-00002.safetensors"
        ]),
        ModelConfig(path="/share/home/202230550120/models/FLUX.1-dev/ae.safetensors"),
    ],
    tokenizer_1_config=ModelConfig(path="/share/home/202230550120/models/FLUX.1-dev/tokenizer/"),
    tokenizer_2_config=ModelConfig(path="/share/home/202230550120/models/FLUX.1-dev/tokenizer_2/"),
)

# 冷加载 LoRA 权重（直接融合到模型）
lora_config = ModelConfig(path="/share/home/202230550120/DiffSynth-Studio/models/train3/FLUX.1-Kontext-dev_lora_cosine_annealing/epoch-4.safetensors")
# lora_config = ModelConfig(path="/share/home/202230550120/DiffSynth-Studio/models/train8.3.26.21:50/FLUX.1-Kontext-dev_lora_cosine_annealing_without_furn/epoch-4.safetensors")
# lora_config = ModelConfig(path="/share/home/202230550120/DiffSynth-Studio/models/train7.3.26.21:48/FLUX.1-Kontext-dev_lora_learning_rate_constant/epoch-4.safetensors")


pipe.load_lora(pipe.dit, lora_config, alpha=1)

# 创建输出目录（使用时间戳标记）
# 从 LoRA 路径提取训练号（如 train6）
lora_path = lora_config.path
lora_trainnum = lora_path.split('/models/')[-1].split('/')[0]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_base_dir = f"inference_results_{timestamp}_{lora_trainnum}"
os.makedirs(output_base_dir, exist_ok=True)

inference_tasks = [
    # {
    #     "kontext_image_path": "/share/home/202230550120/diffusers/output/viz_samples/1540780_885846_客卫_no_furn.png",
    #     "gt_image_path": "/share/home/202230550120/diffusers/output/viz_samples/1540780_885846_客卫_with_furn.png",
    #     "prompt": "2D semantic segmentation map, top-down floorplan, simple solid color blocks. room: bathroom. feature: The bathroom emphasizes functionality and hygiene, with emphasis on dry and wet separation in order to maintain cleanliness.",
    #     "name": "task_1540780_885846_客卫"
    # },
    # {
    #     "kontext_image_path": "/share/home/202230550120/diffusers/output/viz_samples/1872162_105361_次卧_no_furn.png",
    #     "gt_image_path": "/share/home/202230550120/diffusers/output/viz_samples/1872162_105361_次卧_with_furn.png",
    #     "prompt": "2D semantic segmentation map, top-down floorplan, simple solid color blocks. room: bedroom. ",
    #     "name": "task_1872162_105361_次卧"
    # },
    # {
    #     "kontext_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2430829_13299_主卫_no_furn.png",
    #     "gt_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2430829_13299_主卫_with_furn.png",
    #     "prompt": "2D semantic segmentation map, top-down floorplan, simple solid color blocks. room: bathroom. feature: There's an independent bath in the master's bedroom.",
    #     "name": "task_2430829_13299_主卫"
    # },
    # {
    #     "kontext_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2430829_331035_客餐厅_no_furn.png",
    #     "gt_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2430829_331035_客餐厅_with_furn.png",
    #     "prompt": "2D semantic segmentation map, top-down floorplan, simple solid color blocks. room: living_dining_room. feature: Restaurants are designed in an integrated and open manner, with loose wires and a green recreational area on the balcony.",
    #     "name": "task_2430829_331035_客餐厅"
    # },
    # {
    #     "kontext_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2430829_331297_儿童房_no_furn.png",
    #     "gt_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2430829_331297_儿童房_with_furn.png",
    #     "prompt": "2D semantic segmentation map, top-down floorplan, simple solid color blocks. room: bedroom. feature: One of them is a children's room.",
    #     "name": "task_2430829_331297_儿童房"
    # },
    # {
    #     "kontext_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2430829_248139_厨房_no_furn.png",
    #     "gt_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2430829_248139_厨房_with_furn.png",
    #     "prompt": "2D semantic segmentation map, top-down floorplan, simple solid color blocks. room: kitchen. feature: There's a kitchen and a public health room in the centre.",
    #     "name": "task_2430829_248139_厨房"
    # },
    # {
    #     "kontext_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2430829_215865_阳台_no_furn.png",
    #     "gt_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2430829_215865_阳台_with_furn.png",
    #     "prompt": "2D semantic segmentation map, top-down floorplan, simple solid color blocks. room: balcony. feature: And connect to the large balcony below the left; the balcony keeps a green recreational area.",
    #     "name": "task_2430829_215865_阳台"
    # },
    # {
    #     "kontext_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2440029_17409_次卧_no_furn.png",
    #     "gt_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2440029_17409_次卧_with_furn.png",
    #     "prompt": "2D semantic segmentation map, top-down floorplan, simple solid color blocks. room: bedroom. feature: The sub-residential is required to reconcile residence with the functions of the study/electric competition.",
    #     "name": "task_2440029_17409_次卧"
    # },
    # {
    #     "kontext_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2428975_7040_客餐厅_no_furn.png",
    #     "gt_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2428975_7040_客餐厅_with_furn.png",
    #     "prompt": "2D semantic segmentation map, top-down floorplan, simple solid color blocks. room: living dining room. feature: The household is designed for two compact rooms. The left is a spacious guest restaurant integrated area with an open layout connecting the north balcony/scene window; the bottom is the gate and the L kitchen.",
    #     "name": "task_2428975_7040_客餐厅"
    # },
    # {
    #     "kontext_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2430829_13299_主卫_no_furn.png",
    #     "gt_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2430829_13299_主卫_with_furn.png",
    #     "prompt": "2D semantic segmentation map, top-down floorplan, simple solid color blocks. room: bathroom.",
    #     "name": "task_2430829_13299_主卫"
    # },
    # {
    #     "kontext_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2435174_111221_客餐厅_no_furn.png",
    #     "gt_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2435174_111221_客餐厅_with_furn.png",
    #     "prompt": "2D semantic segmentation map, top-down floorplan, simple solid color blocks. room: living dining room. feature: The home is a compact, two-room, two-room, first-guard structure.",
    #     "name": "task_2435174_111221_客餐厅"
    # },
    # {
    #     "kontext_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2462429_8073402_主卧_no_furn.png",
    #     "gt_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2462429_8073402_主卧_with_furn.png",
    #     "prompt": "2D semantic segmentation map, top-down floorplan, simple solid color blocks. room: bedroom. feature: The main bedroom is a spacious, well-lit room with a large window.",
    #     "name": "task_2462429_8073402_主卧"
    # },
    # {
    #     "kontext_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2462429_8073491_次卧_no_furn.png",
    #     "gt_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2462429_8073491_次卧_with_furn.png",
    #     "prompt": "2D semantic segmentation map, top-down floorplan, simple solid color blocks. room: bedroom. feature: Three bedrooms need to be kept, the main bedroom needs to be equipped with an independent bathroom with a bathtub, and one of the next bedrooms can be used as a children's room.",
    #     "name": "task_2462429_8073491_次卧"
    # },
    # {
    #     "kontext_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2462928_137571_主卧_no_furn.png",
    #     "gt_image_path": "/share/home/202230550120/diffusers/output_add/viz_samples/2462928_137571_主卧_with_furn.png",
    #     "prompt": "2D semantic segmentation map, top-down floorplan, simple solid color blocks. room: bedroom. feature: There's a dry-wet-separated bathroom on the west side and the main bedroom.",
    #     "name": "task_2462928_137571_主卧"
    # }

    {
        "kontext_image_path": "/share/home/202230550120/image.png",
        "gt_image_path": "/share/home/202230550120/image (1).png",
        "prompt": "1. 拆除主卧（25.19㎡）内部的隔墙，将原有的两个小空间合并为一个大开间； 2. 拆除次卧（9.57㎡）与客餐厅之间的非承重隔墙，改用透明玻璃隔断或半开放柜体； 3. 封闭原卫生间朝向走道的门洞，在主卧内部新开门洞，实现主卧套房化； 4. 在客餐厅大空间中心区域，围绕结构柱增设一组中岛台面，并对玄关入口处增设 L 型遮挡隔断。",
        "name": ""
    }
]

# 循环处理每个推理任务
for idx, task in enumerate(inference_tasks):
    print(f"\n处理任务 {idx + 1}/{len(inference_tasks)}: {task['name']}")
    
    # 加载条件图像（无家具）
    kontext_image = Image.open(task["kontext_image_path"])
    print(f"加载条件图像: {task['kontext_image_path']}")
    print(f"条件图像尺寸: {kontext_image.size}")
    
    # 加载 GT 图像（有家具）
    gt_image = Image.open(task["gt_image_path"])
    print(f"加载 GT 图像: {task['gt_image_path']}")
    print(f"GT 图像尺寸: {gt_image.size}")
    
    # 推理
    print(f"开始推理，提示词: {task['prompt']}")
    generated_image = pipe(
        prompt=task["prompt"],
        kontext_images=kontext_image,
        embedded_guidance=2.5,
        seed=42 + idx,  # 每个任务使用不同的 seed
    )
    print(f"推理完成，生成图像尺寸: {generated_image.size}")
    
    # 调整 GT 图像到与生成图像相同的大小
    gt_resized = gt_image.resize(generated_image.size)
    
    # 拼接：[GT有家具 | 间隙 | 生成结果]，并在顶部添加标签
    gap_width = 20  # 图片之间的间隙宽度（像素）
    label_height = 40  # 标签区域高度
    
    # 创建容纳标签和图片的完整图像
    total_width = gt_resized.width + gap_width + generated_image.width
    total_height = generated_image.height + label_height
    combined_image = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    
    # 拼接：GT（左）| 间隙 | 生成结果（右）
    combined_image.paste(gt_resized, (0, label_height))
    combined_image.paste(generated_image, (gt_resized.width + gap_width, label_height))
    
    # 添加标签文字
    draw = ImageDraw.Draw(combined_image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()
    
    # GT标签（左侧）
    gt_label = "Ground Truth"
    gt_bbox = draw.textbbox((0, 0), gt_label, font=font)
    gt_text_width = gt_bbox[2] - gt_bbox[0]
    gt_x = (gt_resized.width - gt_text_width) // 2
    draw.text((gt_x, 10), gt_label, fill=(0, 0, 0), font=font)
    
    # Prediction标签（右侧）
    pred_label = "Prediction"
    pred_bbox = draw.textbbox((0, 0), pred_label, font=font)
    pred_text_width = pred_bbox[2] - pred_bbox[0]
    pred_x = gt_resized.width + gap_width + (generated_image.width - pred_text_width) // 2
    draw.text((pred_x, 10), pred_label, fill=(0, 0, 0), font=font)
    
    # 保存结果
    output_path = f"{output_base_dir}/{task['name']}_combined.jpg"
    combined_image.save(output_path)
    print(f"拼接结果已保存: {output_path}")
    print(f"拼接格式: [生成结果 | GT有家具]")
    
    # 分别保存各个部分
    generated_path = f"{output_base_dir}/{task['name']}_00_generated.jpg"
    gt_path = f"{output_base_dir}/{task['name']}_01_gt_with_furn.jpg"
    generated_image.save(generated_path)
    gt_resized.save(gt_path)
    print(f"单图已保存:")
    print(f"  - 生成结果: {generated_path}")
    print(f"  - GT（有家具）: {gt_path}")

print(f"\n所有推理任务完成！结果已保存到 {output_base_dir}/ 目录")
