import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from PIL import Image


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

# image_1 = pipe(
#     prompt="a beautiful Asian long-haired female college student.",
#     embedded_guidance=2.5,
#     seed=1,
# )
# image_1.save("image_1.jpg")

image_2 = pipe(
    prompt="1.Demolish the internal partition walls of the master bedroom (25.19㎡) to merge the two smaller spaces into one large open-plan area.2.Remove the non-load-bearing wall between the secondary bedroom (9.57㎡) and the living-dining area, replacing it with a transparent glass partition or a semi-open cabinet system.3.Seal off the original bathroom door facing the corridor and create a new doorway within the master bedroom to form an en-suite layout.4.In the central area of the living-dining space, add an island unit around the structural column, and install an L-shaped partition at the entrance to provide visual screening",
    kontext_images=[
        Image.open("/share/home/202230550120/image.png"),
    ],
    embedded_guidance=2.5,
    seed=2,
)
image_2.save("image_2.jpg")

# image_3 = pipe(
#     prompt="let her smile.",
#     kontext_images=image_1,
#     embedded_guidance=2.5,
#     seed=3,
# )
# image_3.save("image_3.jpg")

# image_4 = pipe(
#     prompt="let the girl play basketball.",
#     kontext_images=image_1,
#     embedded_guidance=2.5,
#     seed=4,
# )
# image_4.save("image_4.jpg")

# image_5 = pipe(
#     prompt="move the girl to a park, let her sit on a chair.",
#     kontext_images=image_1,
#     embedded_guidance=2.5,
#     seed=5,
# )
# image_5.save("image_5.jpg")