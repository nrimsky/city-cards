from PIL import Image
import numpy as np


def preprocess_png_to_white_background(
    input_path: str, output_path: str = None, threshold: int = 230
) -> str:
    try:
        img = Image.open(input_path).convert("RGBA")
        data = np.array(img)
        # Separate RGB and alpha channels
        rgb = data[:, :, :3]
        alpha = data[:, :, 3]
        # Find pixels where all RGB values are above threshold
        # This creates a mask of pixels that should become white
        light_pixels = np.all(rgb >= threshold, axis=2)
        # Set those pixels to pure white
        data[light_pixels] = [255, 255, 255, 255]
        # Also handle semi-transparent light pixels
        # If alpha < 255 and the color is light, make it fully white
        semi_transparent_light = (alpha < 255) & np.all(rgb >= threshold - 30, axis=2)
        data[semi_transparent_light] = [255, 255, 255, 255]
        # Create new image from modified data
        new_img = Image.fromarray(data, "RGBA")
        if output_path is None:
            output_path = input_path
        new_img.save(output_path, "PNG")
        print(f"Processed image saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error processing image {input_path}: {e}")
        return input_path


def crop_white_borders(
    input_path: str,
    output_path: str = None,
    max_padding: int = 10,
    white_threshold: int = 230,
) -> str:
    try:
        img = Image.open(input_path).convert("RGBA")
        data = np.array(img)
        height, width = data.shape[:2]
        rgb = data[:, :, :3]
        is_white = np.all(rgb >= white_threshold, axis=2)
        top_crop = 0
        for row in range(height):
            if not np.all(is_white[row, :]):
                top_crop = max(0, row - max_padding)
                break
        bottom_crop = height
        for row in range(height - 1, -1, -1):
            if not np.all(is_white[row, :]):
                bottom_crop = min(height, row + 1 + max_padding)
                break
        left_crop = 0
        for col in range(width):
            if not np.all(is_white[:, col]):
                left_crop = max(0, col - max_padding)
                break
        right_crop = width
        for col in range(width - 1, -1, -1):
            if not np.all(is_white[:, col]):
                right_crop = min(width, col + 1 + max_padding)
                break
        if top_crop >= bottom_crop or left_crop >= right_crop:
            print(
                f"Warning: Image appears to be entirely white or crop boundaries invalid. Skipping crop."
            )
            if output_path is None:
                output_path = input_path
            img.save(output_path, "PNG")
            return output_path
        cropped_data = data[top_crop:bottom_crop, left_crop:right_crop]
        cropped_img = Image.fromarray(cropped_data, "RGBA")

        # Save the cropped image
        if output_path is None:
            output_path = input_path
        cropped_img.save(output_path, "PNG")
        return output_path

    except Exception as e:
        print(f"Error cropping image {input_path}: {e}")
        return input_path
