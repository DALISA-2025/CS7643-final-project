import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from diffusers import StableDiffusionInstructPix2PixPipeline

class FractalGenerator:
    @staticmethod
    def generate_mandelbrot(height, width, x_min, x_max, y_min, y_max, max_iter):
        x, y = np.linspace(x_min, x_max, width), np.linspace(y_min, y_max, height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        Z = np.zeros_like(C)
        fractal = np.zeros(C.shape, dtype=float)
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + C[mask]
            fractal[mask] += 1
        return fractal / max_iter

    @staticmethod
    def generate(save_dir, num_images=100, img_size=(32, 32)):
        os.makedirs(save_dir, exist_ok=True)
        x_start, x_end = -2.0, 1.0
        y_start, y_end = -1.5, 1.5
        
        for i in range(num_images):
            zoom = np.random.uniform(0.5, 2.0)
            center_x = np.random.uniform(-0.7, 0.0)
            center_y = np.random.uniform(-0.5, 0.5)
            range_x = (x_end - x_start) / zoom
            range_y = (y_end - y_start) / zoom
            
            fractal = FractalGenerator.generate_mandelbrot(
                img_size[0], img_size[1], 
                center_x - range_x/2, center_x + range_x/2,
                center_y - range_y/2, center_y + range_y/2, 
                max_iter=100
            )
            
            plt.figure(figsize=(1, 1), dpi=img_size[0])
            cmap = np.random.choice(['inferno', 'magma', 'plasma', 'viridis', 'twilight', 'ocean'])
            plt.imshow(fractal, cmap=cmap)
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f'fractal_{i:03d}.png'), bbox_inches='tight', pad_inches=0)
            plt.close()

class DiffusionGenerator:
    def __init__(self, model_id="timbrooks/instruct-pix2pix", device=None):
        if device is None:
            if torch.cuda.is_available(): self.device = "cuda"
            elif torch.backends.mps.is_available(): self.device = "mps"
            else: self.device = "cpu"
        else:
            self.device = device
            
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=dtype, safety_checker=None)
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        
        self.prompts = [
            "Autumn", "Snowy", "Watercolor art", "Sunset", "Rainbow", "Mosaic", "Sunny",
            "Origami", "Starry Night", "Pointillism", "Cyberpunk", "Impressionism"
        ]
        
    def generate_variation(self, image, prompt, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7.0):
        full_prompt = f"A transformed version of image into {prompt}"
        original_size = image.size
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        if image.mode != "RGB": image = image.convert("RGB")
        
        result = self.pipe(
            prompt=full_prompt, image=image, num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale, guidance_scale=guidance_scale
        ).images[0]
        
        return result.resize(original_size, Image.Resampling.LANCZOS)

class DiffuseMixTransform:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def get_mask(self, height, width):
        mask = torch.zeros((1, height, width), dtype=torch.float32)
        if np.random.choice(['vertical', 'horizontal']) == 'vertical':
            mask[:, :, :np.random.randint(0, width)] = 1.0
        else:
            mask[:, :np.random.randint(0, height), :] = 1.0
        return mask

    def __call__(self, original_img, generated_img, fractal_img, lam=0.15):
        if isinstance(original_img, Image.Image): original_img = self.to_tensor(original_img)
        if isinstance(generated_img, Image.Image): generated_img = self.to_tensor(generated_img)
        if isinstance(fractal_img, Image.Image): fractal_img = self.to_tensor(fractal_img)
        
        _, h, w = original_img.shape
        if generated_img.shape != original_img.shape: generated_img = transforms.functional.resize(generated_img, (h, w))
        if fractal_img.shape != original_img.shape: fractal_img = transforms.functional.resize(fractal_img, (h, w))

        mask = self.get_mask(h, w)
        hybrid_img = mask * original_img + (1 - mask) * generated_img
        return (1 - lam) * hybrid_img + lam * fractal_img

class DiffuseMixTensorDataset(Dataset):
    def __init__(self, pt_file_path, transform=None):
        self.transform = transform
        data = torch.load(pt_file_path)
        self.images = data['images']
        self.labels = data['labels']
        
    def __len__(self): return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform: image = self.transform(image)
        return image, label
