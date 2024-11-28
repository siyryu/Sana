import torch
from app.sana_pipeline import SanaPipeline
from torchvision.utils import save_image

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(42)

sana = SanaPipeline("/home/siyuanyu/lightsvd/Sana/configs/sana_config/512ms/Sana_1600M_img512.yaml")
sana.from_pretrained(
    "/home/siyuanyu/HF-cache/hub/models--Efficient-Large-Model--Sana_1600M_512px/snapshots/b17b080d9f3b6c4fb71ba7b0d8384d445d500dd1/checkpoints/Sana_1600M_512px.pth"
)
prompt = 'a cyberpunk cat with a neon sign that says "Sana"'

image = sana(
    prompt=prompt,
    height=512,
    width=512,
    guidance_scale=5.0,
    pag_guidance_scale=2.0,
    num_inference_steps=18,
    generator=generator,
)

save_image(image, "output/sana_without_flashatten.png", nrow=1, normalize=True, value_range=(-1, 1))
