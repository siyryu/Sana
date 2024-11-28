import torch
from app.sana_pipeline import SanaPipeline
import coremltools as ct

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(42)

sana = SanaPipeline("/home/siyuanyu/lightsvd/Sana/configs/sana_config/512ms/Sana_1600M_img512_fp32.yaml")
sana.from_pretrained(
    "/home/siyuanyu/HF-cache/hub/models--Efficient-Large-Model--Sana_1600M_512px/snapshots/b17b080d9f3b6c4fb71ba7b0d8384d445d500dd1/checkpoints/Sana_1600M_512px.pth"
)
prompt = 'a cyberpunk cat with a neon sign that says "Sana"'

ae = sana.vae

example_input = torch.randn(1, 32, 16, 16).to(device)


class VAEDecodeWrapper(torch.nn.Module):
    def __init__(self, ae):
        super().__init__()
        self.ae = ae

    def forward(self, latent):
        return self.ae.decode(latent)


vae_decode_wrapper = VAEDecodeWrapper(ae)
with torch.no_grad():
    traced_model = torch.jit.trace(
        vae_decode_wrapper,
        example_input.detach() / ae.cfg.scaling_factor,
    )

mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="input", shape=example_input.shape)],
    minimum_deployment_target=ct.target.iOS18,
)

mlmodel.save("/home/siyuanyu/lightsvd/outputs/sana_dcae_decode.mlpackage")
