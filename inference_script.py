import torch
from app.sana_pipeline import SanaPipeline
from torchvision.utils import save_image
import time

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(42)
prompt = 'a cyberpunk cat with a neon sign that says "Sana"'

sana_600M_checkpoint_path = "/home/siyuanyu/HF-cache/hub/models--Efficient-Large-Model--Sana_600M_512px/snapshots/46f36e52d7d1376b3c948a12c6bf97a9b0c1eb39/checkpoints/Sana_600M_512px_MultiLing.pth"
sana_1600M_checkpoint_path = "/home/siyuanyu/HF-cache/hub/models--Efficient-Large-Model--Sana_1600M_512px/snapshots/b17b080d9f3b6c4fb71ba7b0d8384d445d500dd1/checkpoints/Sana_1600M_512px.pth"

sana_600M_config = "/home/siyuanyu/lightsvd/Sana/configs/sana_config/512ms/Sana_600M_img512.yaml"
# Inference time: 1.81 seconds on A5000
sana_1600M_fp16_config = "/home/siyuanyu/lightsvd/Sana/configs/sana_config/512ms/Sana_1600M_img512.yaml"
# Inference time: 2.20 seconds on A5000
sana_600M_fp32_config = "/home/siyuanyu/lightsvd/Sana/configs/sana_config/512ms/Sana_600M_img512_fp32.yaml"
# Inference time: 2.12 seconds on A5000

sana = SanaPipeline(sana_600M_fp32_config)
sana.from_pretrained(
    sana_600M_checkpoint_path
)

image = sana(
    prompt=prompt,
    height=512,
    width=512,
    guidance_scale=5.0,
    pag_guidance_scale=2.0,
    num_inference_steps=18,
    generator=generator,
)

save_image(image, "sana_out_600M_fp32.png", nrow=1, normalize=True, value_range=(-1, 1))
