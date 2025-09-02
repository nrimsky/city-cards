from google import genai
from PIL import Image
from io import BytesIO
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=API_KEY)

TEMPLATE = "Generate a representative scene from the city {city}, {state}, minimalist, aquarelle, plain bright exactly white background, no text."


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
    white_threshold: int = 250,
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


def make_image(city: str, state: str, dir: str = "images"):
    image_save_path = f"{dir}/image_{city.lower().replace(' ', '_')}_{state.lower().replace(' ', '_')}.png"
    if os.path.exists(image_save_path):
        return image_save_path
    response = client.models.generate_content(
        model="gemini-2.5-flash-image-preview",
        contents=[TEMPLATE.format(city=city, state=state)],
    )
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = Image.open(BytesIO(part.inline_data.data))
            image.save(image_save_path)
    preprocess_png_to_white_background(image_save_path)
    crop_white_borders(image_save_path)
    return image_save_path


if __name__ == "__main__":
    city = input("city >> ")
    state = input("state >> ")
    make_image(city, state, "custom")
