import xml.etree.ElementTree as ET
import os
import csv
from generate_country_images import make_image
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
from world_map import create_mini_map_group


def create_country_card(
    country: str,
    code: str,
    image_path: str,
) -> str:
    """
    Create a playing card for a country with text converted to paths.
    """
    template_content = load_card_template()
    closing_tag_pos = template_content.rfind("</svg>")
    if closing_tag_pos == -1:
        print("Warning: Could not find closing SVG tag in template")
        closing_tag_pos = len(template_content)

    additional_content = []

    # Add image with preserved aspect ratio
    if os.path.exists(image_path):
        original_dims = get_image_dimensions(image_path)
        if original_dims:
            original_width, original_height = original_dims
            BOUNDING_BOX_WIDTH = 130
            BOUNDING_BOX_HEIGHT = 130
            ratio_w = BOUNDING_BOX_WIDTH / original_width
            ratio_h = BOUNDING_BOX_HEIGHT / original_height
            scale_ratio = min(ratio_w, ratio_h)
            img_width = original_width * scale_ratio
            img_height = original_height * scale_ratio
            img_x = (CARD_WIDTH / 2) - (img_width / 2)
            img_y = 120 - (img_height / 2)
            image_data = embed_image_as_base64(image_path)
            if image_data:
                additional_content.append(
                    f'<g id="country-image"><image x="{img_x:.2f}" y="{img_y:.2f}" width="{img_width:.2f}" height="{img_height:.2f}" href="{image_data}"/></g>'
                )

    # --- Text to Path Conversion ---
    country_info_group = ET.Element("g", {"id": "country-info"})
    MAX_COUNTRY_WIDTH = 140
    country_font_size = 18.0
    initial_width = get_text_width(
        text=country,
        font_family="Sans",
        font_size=country_font_size,
        font_weight="bold",
    )
    if initial_width > MAX_COUNTRY_WIDTH:
        country_font_size *= MAX_COUNTRY_WIDTH / initial_width
        print(
            f"  - Resizing country '{country}' font to {country_font_size:.2f}pt to fit."
        )

    country_paths = text_to_svg_group(
        text=country,
        x=CARD_WIDTH // 2,
        y=40,
        font_family="Sans",
        font_size=country_font_size,
        font_weight="bold",
        text_anchor="middle",
        fill="#2B2B2B",
    )
    if country_paths is not None:
        country_info_group.append(country_paths)

    additional_content.append(
        f'\n{ET.tostring(country_info_group, encoding="unicode")}'
    )

    # Add mini map
    map_group = create_mini_map_group(country=country, card_height=CARD_HEIGHT)
    additional_content.append(f'\n{ET.tostring(map_group, encoding="unicode")}')

    # Add code in bottom right
    code_group = ET.Element("g", {"id": "country-code"})
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


def generate_card_from_country(
    country: str,
    code: str,
    image_path: str,
    output_dir: str = "country_cards",
    skip_if_exists: bool = True,
):
    """Generate and save a card for a country."""
    filename = f"{country.lower().replace(' ', '_')}.svg"
    filepath = os.path.join(output_dir, filename)
    if skip_if_exists and os.path.exists(filepath):
        print(f"Card for {country} already exists. Skipping...")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating card for {country}...")

    svg_content = create_country_card(
        country,
        code=code,
        image_path=image_path,
    )

    with open(filepath, "w") as f:
        f.write(svg_content)
    print(f"  ✓ Saved to {filepath}")


def main():
    """Main function to generate country cards from CSV file."""
    csv_path = "countries.csv"
    try:
        with open(csv_path, "r") as f:
            data = list(csv.DictReader(f))
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file {csv_path} not found.")

    n = len(data)
    codes = get_codes(n)

    for elem in data:
        try:
            name = elem["Name"]
            code = codes.pop()
        except Exception as e:
            print(f"Error reading row {elem}: {e}")
            continue

        img_path = make_image(country=name)
        generate_card_from_country(country=name, code=code, image_path=img_path)


if __name__ == "__main__":
    main()
