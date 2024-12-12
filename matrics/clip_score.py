import torch
from torchmetrics.multimodal.clip_score import CLIPScore
from app.sana_pipeline import SanaPipeline

device = torch.device("cuda:3")

generator = torch.Generator(device=device).manual_seed(42)
sana_600M_fp32_config = "/home/siyuanyu/lightsvd/Sana/configs/sana_config/512ms/Sana_600M_img512.yaml"
sana_600M_checkpoint_path = "/home/siyuanyu/HF-cache/hub/models--Efficient-Large-Model--Sana_600M_512px/snapshots/46f36e52d7d1376b3c948a12c6bf97a9b0c1eb39/checkpoints/Sana_600M_512px_MultiLing.pth"
sana = SanaPipeline(sana_600M_fp32_config, device=device)
sana.from_pretrained(
    sana_600M_checkpoint_path
)

prompts = [
    "a photo of an astronaut riding a horse on mars",
    "A high tech solarpunk utopia in the Amazon rainforest",
    "A pikachu fine dining with a view to the Eiffel Tower",
    "A mecha robot in a favela in expressionist style",
    "an insect robot preparing a delicious meal",
    "A small cabin on top of a snowy mountain in the style of Disney, artstation",
]

images = sana(
    prompt=prompts,
    height=512,
    width=512,
    guidance_scale=5.0,
    pag_guidance_scale=2.0,
    num_inference_steps=18,
    generator=generator,
)
print(f"{images.shape=}")

# 使用 make_grid 创建图像网格
images_int = ((images + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
print(f"{images_int.shape=}, {images_int.dtype=}")
print(f"{images_int.min()=}, {images_int.max()=}")

metric = CLIPScore(model_name_or_path="/home/siyuanyu/HF-cache/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41").to(device)

score = metric(images_int, prompts)
score.detach().round()
print(f"{score=}")
