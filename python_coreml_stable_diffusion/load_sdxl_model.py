from diffusers import StableDiffusionXLPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)


# for i, item in enumerate([item for item in dir(pipeline) if not item.startswith('_')]):
#     print("{} : {}".format(i, item))

print(type(pipeline.unet))