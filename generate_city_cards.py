import xml.etree.ElementTree as ET
import hashlib
import os
import csv
from generate_city_images import make_image

from helpers import (
    CARD_HEIGHT,
    CARD_WIDTH,
    embed_image_as_base64,
    get_codes,
    get_image_dimensions,
    get_text_width,
    load_card_template,
    text_to_svg_group,
)
from us_map import create_mini_map_group

MAP_WIDTH = 80
MAP_HEIGHT = 50


def generate_city_code(city: str, state: str) -> str:
    """
    Generate a unique 3-letter code for a city/state combination.
    Args:
        city: City name
        state: State name or abbreviation
    Returns:
        3-letter code
    """
    # Create a hash of the city and state
    combined = f"{city.upper()}{state.upper()}"
    hash_obj = hashlib.md5(combined.encode())
    hash_hex = hash_obj.hexdigest()
    # Convert hash to uppercase letters
    code = ""
    for i in range(0, 6, 2):
        # Convert pairs of hex digits to letters A-Z
        num = int(hash_hex[i : i + 2], 16) % 26
        code += chr(65 + num)
    return code[:3]


def create_city_card(
    city: str,
    state: str,
    code: str = None,
    image_path: str = None,
) -> str:
    template_content = load_card_template()
    closing_tag_pos = template_content.rfind("</svg>")
    if closing_tag_pos == -1:
        print("Warning: Could not find closing SVG tag in template")
        closing_tag_pos = len(template_content)

    additional_content = []

    # --- UPDATED: Add image with preserved aspect ratio ---
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

    # --- Text to Path Conversion ---
    # (The rest of the function remains the same as before)
    city_info_group = ET.Element("g", {"id": "city-info"})

    MAX_CITY_WIDTH = 140
    city_font_size = 18.0

    initial_width = get_text_width(
        text=city, font_family="Sans", font_size=city_font_size, font_weight="bold"
    )

    if initial_width > MAX_CITY_WIDTH:
        city_font_size = (MAX_CITY_WIDTH / initial_width) * city_font_size
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

    state_paths = text_to_svg_group(
        text=state.upper(),
        x=CARD_WIDTH // 2,
        y=58,
        font_family="Sans",
        font_size=14,
        font_weight="normal",
        text_anchor="middle",
        fill="#666",
    )
    if state_paths is not None:
        city_info_group.append(state_paths)

    additional_content.append(f'\n{ET.tostring(city_info_group, encoding="unicode")}')

    map_x, map_y = 18, CARD_HEIGHT - MAP_HEIGHT - 18
    map_group = create_mini_map_group(
        state, map_x, map_y, map_width=MAP_WIDTH, map_height=MAP_HEIGHT, city_name=city
    )
    additional_content.append(f'\n{ET.tostring(map_group, encoding="unicode")}')

    if code is None:
        code = generate_city_code(city, state)

    code_group = ET.Element("g", {"id": "city-code"})
    code_paths = text_to_svg_group(
        text=code,
        x=CARD_WIDTH - 36,
        y=CARD_HEIGHT - 23,
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
    state: str,
    code: str = None,
    image_path: str = None,
    output_dir: str = "city_cards",
    skip_if_exists: bool = True,
):
    filename = f"{city.lower().replace(' ', '_')}_{state.lower().replace(' ', '_')}.svg"
    filepath = os.path.join(output_dir, filename)
    if skip_if_exists and os.path.exists(filepath):
        print(f"Card for {city}, {state} already exists. Skipping...")
        return
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating card for {city}, {state}...")
    # Generate card
    svg_content = create_city_card(city, state, code=code, image_path=image_path)
    # Save to file
    with open(filepath, "w") as f:
        f.write(svg_content)
    print(f"  ✓ Saved to {filepath}")


def main():
    data = csv.DictReader(open("cities.csv"))
    data = list(data)
    n = len(data)
    print(f"Loaded {n} cities from CSV.")
    codes = get_codes(n)
    for elem in data:
        try:
            name = elem["Name"]
            state = elem["State"]
            code = codes.pop()
        except Exception as e:
            print(f"Error reading row {elem}: {e}")
            continue
        img_path = make_image(city=name, state=state)
        generate_card_from_city(city=name, state=state, code=code, image_path=img_path)


if __name__ == "__main__":
    main()
