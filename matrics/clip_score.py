import torch
from torchmetrics.multimodal.clip_score import CLIPScore
from app.sana_pipeline import SanaPipeline
from matrics.mjhq_dataset import MJHQ30KDataModule
import torch.nn.functional as F
from tqdm import tqdm

import os
os.environ["DPM_TQDM"] = "True"

device = torch.device("cuda:3")

dataset_meta_path = "/home/siyuanyu/HF-cache/hub/datasets--playgroundai--MJHQ-30K/snapshots/15b0a659e066e763d0e9a6cd8f00e25f8af5e084/meta_data.json"
dataset_path = (
    "/home/siyuanyu/HF-cache/hub/datasets--playgroundai--MJHQ-30K/snapshots/15b0a659e066e763d0e9a6cd8f00e25f8af5e084/"
)


generator = torch.Generator(device=device).manual_seed(42)
sana_600M_fp32_config = "/home/siyuanyu/lightsvd/Sana/configs/sana_config/512ms/Sana_600M_img512.yaml"
sana_600M_checkpoint_path = "/home/siyuanyu/HF-cache/hub/models--Efficient-Large-Model--Sana_600M_512px/snapshots/46f36e52d7d1376b3c948a12c6bf97a9b0c1eb39/checkpoints/Sana_600M_512px_MultiLing.pth"
sana = SanaPipeline(sana_600M_fp32_config, device=device)
sana.from_pretrained(sana_600M_checkpoint_path)

# Setup MJHQ dataset
data_module = MJHQ30KDataModule(dataset_path, dataset_meta_path, batch_size=4)  # Small batch size for memory
data_module.setup()
dataloader = data_module.full_dataloader()

# Initialize CLIP score metric
metric = CLIPScore(
    model_name_or_path="/home/siyuanyu/HF-cache/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"
).to(device)

total_score = 0
num_batches = 0

# Process batches
pbar = tqdm(dataloader, desc="Calculating CLIP scores")
for batch in pbar:
    images = batch["image"]  # Shape: [B, C, H, W]
    prompts = batch["prompt"]  # List of prompts

    # Generate images with Sana
    generated_images = sana(
        prompt=prompts,
        height=512,
        width=512,
        guidance_scale=5.0,
        pag_guidance_scale=2.0,
        num_inference_steps=18,
        generator=generator,
    )

    # Convert to uint8 format expected by CLIP score
    images_int = ((generated_images + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)

    # Calculate CLIP score
    score = metric(images_int, prompts)
    total_score += score.item()
    num_batches += 1

    current_avg = total_score / num_batches
    pbar.set_postfix({"CLIP Score": f"{current_avg:.4f}"})
