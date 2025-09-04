from google import genai
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv

from utils import crop_white_borders, preprocess_png_to_white_background

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=API_KEY)

TEMPLATE = "Generate a representative scene from the country {country}, minimalist, aquarelle, plain bright exactly white background, no text."


def make_image(country: str, dir: str = "country_images"):
    image_save_path = f"{dir}/image_{country.lower().replace(' ', '_')}.png"
    if os.path.exists(image_save_path):
        return image_save_path
    response = client.models.generate_content(
        model="gemini-2.5-flash-image-preview",
        contents=[TEMPLATE.format(country=country)],
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
    country = input("country >> ")
    make_image(country=country, dir="custom")
