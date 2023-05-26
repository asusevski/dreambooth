import torch
import torch.backends.cudnn as cudnn
from torchvision.transforms import ToPILImage
import numpy as np
import hashlib
import os

from tqdm import tqdm

from diffusers import StableDiffusionPipeline

# Generate class images
class_images_dir = "class_images"
os.makedirs(class_images_dir, exist_ok=True)

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    cudnn.benchmark = True

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

class_prompt = "a photo of Yorkie"
num_imgs_to_generate = 300

with torch.no_grad():
    for i in tqdm(range(num_imgs_to_generate)):
        image = pipe(class_prompt).images[0]

        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
        image_filename = os.path.join(class_images_dir, f"{hash_image}.jpg")
        image.save(image_filename)
