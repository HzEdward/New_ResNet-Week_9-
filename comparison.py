import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import functional as F

def overlay_mask(image_path, mask_path, alpha=0.5, colormap='viridis'):
    """
    Overlay a segmentation mask on the original image using a specific colormap.

    Parameters:
    - image: The original image (Tensor or numpy array).
    - mask: The segmentation mask (Tensor or numpy array).
    - alpha: Transparency level of the overlay.
    - colormap: Colormap to apply to the mask.

    Returns:
    - Composite image with the mask overlaid on the original image.
    """
    print(f"image_path:{image_path}")
    print(f"mask_path:{mask_path}")
    # Load the original image and mask
    image = plt.imread(image_path)
    mask = plt.imread(mask_path, 0)  # The second parameter ensures it's loaded as grayscale

    # Convert tensors to numpy arrays if necessary
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().cpu().numpy()

    # Normalize the mask to ensure it's in the correct range for the colormap
    mask_normalized = mask.astype(np.float32) / mask.max()

    # Apply a colormap to the normalized mask
    # This returns an RGBA image, we take only the RGB channels
    colored_mask = plt.get_cmap(colormap)(mask_normalized)[:, :, :3]

    # Overlay the mask on the image by blending them according to the alpha parameter
    composite_image = (1 - alpha) * image + alpha * colored_mask

    return composite_image

if __name__ == "__main__":
    # Define paths for the original image and mask
    original_image_path = "7_blacklist_pair/image_Video24_frame000730_rotated.png"
    mask_path = "7_blacklist_pair/label_Video24_frame000730_rotated.png"
    
    composite_image = overlay_mask(original_image_path, mask_path, alpha=0.7, colormap='viridis')

    # Specify the output directory and filename
    output_folder = "output_images"
    output_filename = "composite_Video12_frame009750.png"
    output_path = os.path.join(output_folder, output_filename)

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the composite image
    plt.imsave(output_path, composite_image)

    print(f"Composite image saved to {output_path}")


