from google import genai
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv

from img_utils import crop_white_borders, preprocess_png_to_white_background

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=API_KEY)

TEMPLATE = "Generate a representative scene from the city {city}, {state}, minimalist, aquarelle, plain bright exactly white background, no text."


def make_image(city: str, state: str, dir: str = "city_images"):
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
