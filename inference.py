from diffusers import DiffusionPipeline
import torch

model_id = "./model"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompts = [
    "sks yorkie on the moon with a spacesuit on looking at the camera"
]

for idx, prompt in enumerate(prompts):
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    image.save(f"ai_bella{idx+start+1}.png")