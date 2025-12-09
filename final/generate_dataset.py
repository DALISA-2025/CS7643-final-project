import os
import torch
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from final.diffusemix_core import FractalGenerator, DiffusionGenerator, DiffuseMixTransform

def generate_dataset(output_dir='./data/cifar10_diffusemix', fractal_dir='./data/fractals'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate fractals if needed
    if not os.path.exists(fractal_dir) or not os.listdir(fractal_dir):
        FractalGenerator.generate(fractal_dir)
    
    cifar_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True)
    diffusion_gen = DiffusionGenerator()
    mixer = DiffuseMixTransform()
    
    fractal_files = [f for f in os.listdir(fractal_dir) if f.endswith('.png')]
    
    all_images = []
    all_labels = []
    
    for i in tqdm(range(len(cifar_dataset))):
        try:
            original_img, label = cifar_dataset[i]
            prompt = np.random.choice(diffusion_gen.prompts)
            diffused_img = diffusion_gen.generate_variation(original_img, prompt)
            
            fractal_name = np.random.choice(fractal_files)
            fractal_img = Image.open(os.path.join(fractal_dir, fractal_name)).convert('RGB')
            
            mixed_tensor = mixer(original_img, diffused_img, fractal_img)
            
            all_images.append(mixed_tensor)
            all_labels.append(label)
            
            if i % 50 == 0:
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                elif torch.backends.mps.is_available(): torch.mps.empty_cache()
                
        except Exception:
            continue
            
    final_images = torch.stack(all_images)
    final_labels = torch.tensor(all_labels)
    
    torch.save({'images': final_images, 'labels': final_labels}, os.path.join(output_dir, 'diffusemix_data.pt'))
    print("Done")

if __name__ == "__main__":
    generate_dataset()
