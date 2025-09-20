import xml.etree.ElementTree as ET
import random
import os
import csv
from generate_world_city_images import make_image
from helpers import (
    CARD_HEIGHT,
    CARD_WIDTH,
    embed_image_as_base64,
    get_image_dimensions,
    get_text_width,
    load_card_template,
    text_to_svg_group,
)
from world_map import create_mini_map_group


def create_city_card(
    city: str,
    country: str,
    code: str,
    image_path: str,
) -> str:
    template_content = load_card_template()
    closing_tag_pos = template_content.rfind("</svg>")
    if closing_tag_pos == -1:
        print("Warning: Could not find closing SVG tag in template")
        closing_tag_pos = len(template_content)

    additional_content = []

    # Add image with preserved aspect ratio
    if image_path:
        # First, get the image's original dimensions
        original_dims = get_image_dimensions(image_path)

        if original_dims:
            original_width, original_height = original_dims

            # Define the bounding box for the image on the card
            BOUNDING_BOX_WIDTH = 130
            BOUNDING_BOX_HEIGHT = 130

            # Calculate scaling ratios for width and height
            ratio_w = BOUNDING_BOX_WIDTH / original_width
            ratio_h = BOUNDING_BOX_HEIGHT / original_height
            # Use the smaller ratio to ensure the image fits entirely
            scale_ratio = min(ratio_w, ratio_h)

            # Calculate the new dimensions that preserve the aspect ratio
            img_width = original_width * scale_ratio
            img_height = original_height * scale_ratio

            # Center the resized image in the same area as before
            # The desired center point is (90, 125)
            img_x = (CARD_WIDTH / 2) - (img_width / 2)
            img_y = 125 - (img_height / 2)

            # Now, embed the image data as base64
            image_data = embed_image_as_base64(image_path)
            if image_data:
                # The width and height now perfectly match the image's aspect ratio
                additional_content.append(
                    f'<g id="city-image"><image x="{img_x:.2f}" y="{img_y:.2f}" width="{img_width:.2f}" height="{img_height:.2f}" href="{image_data}"/></g>'
                )

    # Text to Path Conversion
    city_info_group = ET.Element("g", {"id": "city-info"})

    max_city_width = 140
    city_font_size = 18.0

    initial_width = get_text_width(
        text=city, font_family="Sans", font_size=city_font_size, font_weight="bold"
    )

    if initial_width > max_city_width:
        city_font_size = (max_city_width / initial_width) * city_font_size
        print(
            f"  - Resizing city '{city}' font to {city_font_size:.2f}pt to fit 140px width."
        )

    city_paths = text_to_svg_group(
        text=city,
        x=CARD_WIDTH // 2,
        y=40,
        font_family="Sans",
        font_size=city_font_size,
        font_weight="bold",
        text_anchor="middle",
        fill="#2B2B2B",
    )
    if city_paths is not None:
        city_info_group.append(city_paths)

    max_country_width = 140
    country_font_size = 12.0

    initial_width = get_text_width(
        text=country,
        font_family="Sans",
        font_size=country_font_size,
        font_weight="normal",
    )

    if initial_width > max_country_width:
        country_font_size = (max_country_width / initial_width) * country_font_size
        print(
            f"  - Resizing country '{country}' font to {country_font_size:.2f}pt to fit 140px width."
        )

    country_paths = text_to_svg_group(
        text=country,
        x=CARD_WIDTH // 2,
        y=58,
        font_family="Sans",
        font_size=country_font_size,
        font_weight="normal",
        text_anchor="middle",
        fill="#666",
    )
    if country_paths is not None:
        city_info_group.append(country_paths)

    additional_content.append(f'\n{ET.tostring(city_info_group, encoding="unicode")}')

    map_group = create_mini_map_group(
        country=country, card_height=CARD_HEIGHT, city=city
    )
    additional_content.append(f'\n{ET.tostring(map_group, encoding="unicode")}')

    code_group = ET.Element("g", {"id": "city-code"})
    code_paths = text_to_svg_group(
        text=code,
        x=CARD_WIDTH - 36,
        y=CARD_HEIGHT - 24,
        font_family="Mono",
        font_size=15,
        font_weight="bold",
        text_anchor="middle",
        fill="#2BA6DE",
    )
    code_group.append(code_paths)

    additional_content.append(f'\n{ET.tostring(code_group, encoding="unicode")}')

    final_svg = (
        template_content[:closing_tag_pos]
        + "\n".join(additional_content)
        + "\n"
        + template_content[closing_tag_pos:]
    )
    return final_svg


def generate_card_from_city(
    city: str,
    country: str,
    code: str,
    image_path: str = None,
    output_dir: str = "world_city_cards",
    skip_if_exists: bool = True,
):
    filename = (
        f"{city.lower().replace(' ', '_')}_{country.lower().replace(' ', '_')}.svg"
    )
    filepath = os.path.join(output_dir, filename)
    if skip_if_exists and os.path.exists(filepath):
        print(f"Card for {city}, {country} already exists. Skipping...")
        return
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating card for {city}, {country}...")
    # Generate card
    svg_content = create_city_card(city, country, code=code, image_path=image_path)
    # Save to file
    with open(filepath, "w") as f:
        f.write(svg_content)
    print(f"  ✓ Saved to {filepath}")


def main():
    data = csv.DictReader(open("world_cities.csv"))
    codes_seen = set()
    for i, elem in enumerate(data):
        # if i > 0:
        #     break
        try:
            name = elem["Name"]
            country = elem["Country"]
            code = random.randint(0, 999)
            while code in codes_seen:
                code = random.randint(0, 999)
            codes_seen.add(code)
            # convert to DDD string
            code = f"{code:03d}"
        except Exception as e:
            print(f"Error reading row {elem}: {e}")
            continue
        img_path = make_image(city=name, country=country)
        generate_card_from_city(
            city=name, country=country, code=code, image_path=img_path
        )


if __name__ == "__main__":
    main()
