import os
import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import functional as F

def overlay_mask(image, mask, alpha=0.5):
    """
    Overlay a segmentation mask on the original image.

    Parameters:
    - image: The original image (Tensor or numpy array).
    - mask: The segmentation mask (Tensor or numpy array).
    - alpha: Transparency level of the overlay.

    Returns:
    - Composite image with the mask overlaid on the original image.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().cpu().numpy()

    # Apply a colormap to the mask
    colored_mask = plt.get_cmap('jet')(mask)[:, :, :3]  # Discard the alpha channel

    # Overlay the mask on the image
    composite_image = (1 - alpha) * image + alpha * colored_mask

    return composite_image


if __name__ == "__main__":
    # Load your original image and mask here
    original_image_path = "7_blacklist_pair/image_Video24_frame000730_rotated.png"
    mask_path = "7_blacklist_pair/label_Video24_frame000730_rotated.png"
    print(f"the height and width of original_image_path:{plt.imread(original_image_path).shape}")
    print(f"the height and width of mask_path:{plt.imread(mask_path).shape}")
    sys.exit(1)
    original_image = plt.imread(original_image_path)
    mask = plt.imread(mask_path)

    composite_image = overlay_mask(original_image, mask, alpha=0.7)

    # Define the folder and filename for the saved image
    output_folder = "output_images"
    output_filename = "label_Video12_frame009750.png"
    output_path = f"{output_folder}/{output_filename}"

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the composite image
    plt.imsave(output_path, composite_image)

    print(f"Composite image saved to {output_path}")
