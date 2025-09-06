import xml.etree.ElementTree as ET
import os
import csv
from generate_state_images import make_image

from helpers import (
    CARD_HEIGHT,
    CARD_WIDTH,
    embed_image_as_base64,
    get_image_dimensions,
    get_text_width,
    load_card_template,
    text_to_svg_group,
)
from us_map import create_mini_map_group

MAP_WIDTH = 80
MAP_HEIGHT = 50


def create_state_card(
    state_name: str,
    state_abbr: str,
    image_path: str = None,
) -> str:
    """
    Create a playing card for a US state with text converted to paths.
    Args:
        state_name: Full state name
        state_abbr: State abbreviation
        image_path: Path to state image
        template_path: Path to card template
    Returns:
        SVG string for the card
    """
    template_content = load_card_template()
    closing_tag_pos = template_content.rfind("</svg>")
    if closing_tag_pos == -1:
        print("Warning: Could not find closing SVG tag in template")
        closing_tag_pos = len(template_content)

    additional_content = []

    # --- Add image with preserved aspect ratio ---
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
            # The desired center point is (90, 130)
            img_x = (CARD_WIDTH / 2) - (img_width / 2)
            img_y = 130 - (img_height / 2)

            # Now, embed the image data as base64
            image_data = embed_image_as_base64(image_path)
            if image_data:
                # The width and height now perfectly match the image's aspect ratio
                additional_content.append(
                    f'<g id="state-image"><image x="{img_x:.2f}" y="{img_y:.2f}" width="{img_width:.2f}" height="{img_height:.2f}" href="{image_data}"/></g>'
                )

    # --- State Info Group ---
    state_info_group = ET.Element("g", {"id": "state-info"})

    # State name (title)
    MAX_STATE_WIDTH = 140
    state_font_size = 18.0

    initial_width = get_text_width(
        text=state_name,
        font_family="Sans",
        font_size=state_font_size,
        font_weight="bold",
    )

    if initial_width > MAX_STATE_WIDTH:
        state_font_size = (MAX_STATE_WIDTH / initial_width) * state_font_size
        print(
            f"  - Resizing state '{state_name}' font to {state_font_size:.2f}pt to fit 140px width."
        )

    state_paths = text_to_svg_group(
        text=state_name,
        x=CARD_WIDTH // 2,
        y=40,
        font_family="Sans",
        font_size=state_font_size,
        font_weight="bold",
        text_anchor="middle",
        fill="#2B2B2B",
    )
    if state_paths is not None:
        state_info_group.append(state_paths)

    # State abbreviation (subtitle)
    abbr_paths = text_to_svg_group(
        text=state_abbr.upper(),
        x=CARD_WIDTH // 2,
        y=58,
        font_family="Sans",
        font_size=14,
        font_weight="normal",
        text_anchor="middle",
        fill="#666",
    )
    if abbr_paths is not None:
        state_info_group.append(abbr_paths)

    additional_content.append(f'\n{ET.tostring(state_info_group, encoding="unicode")}')

    # --- Mini Map with State Highlighted ---
    # Center the map horizontally: (CARD_WIDTH - MAP_WIDTH) / 2
    map_x = (CARD_WIDTH - MAP_WIDTH) // 2
    map_y = CARD_HEIGHT - MAP_HEIGHT - 20
    map_group = create_mini_map_group(state_name, map_x, map_y, MAP_WIDTH, MAP_HEIGHT)
    additional_content.append(f'\n{ET.tostring(map_group, encoding="unicode")}')

    # --- State Code Circle REMOVED ---
    # The three-letter code element has been removed per requirements

    # Combine all elements
    final_svg = (
        template_content[:closing_tag_pos]
        + "\n".join(additional_content)
        + "\n"
        + template_content[closing_tag_pos:]
    )
    return final_svg


def generate_state_card_file(
    state_name: str,
    state_abbr: str,
    image_path: str = None,
    output_dir: str = "state_cards",
    skip_if_exists: bool = True,
):
    """
    Generate and save a state card to file.
    """
    filename = f"{state_name.lower().replace(' ', '_')}.svg"
    filepath = os.path.join(output_dir, filename)

    if skip_if_exists and os.path.exists(filepath):
        print(f"Card for {state_name} already exists. Skipping...")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating card for {state_name} ({state_abbr})...")

    # Generate card
    svg_content = create_state_card(
        state_name,
        state_abbr,
        image_path=image_path,
    )

    # Save to file
    with open(filepath, "w") as f:
        f.write(svg_content)
    print(f"  ✓ Saved to {filepath}")


def main():
    csv_file = "states.csv"
    with open(csv_file, "r") as f:
        data = csv.DictReader(f)
        for i, elem in enumerate(data):
            # if i > 1:
            #     break
            # Read state information from CSV
            state_name = elem.get("Name", "").strip()
            state_abbr = elem.get("Abbreviation", "").strip()

            if not state_name or not state_abbr:
                print(f"Warning: Skipping row {i+1} - missing Name or Abbreviation")
                continue

            # Generate image using make_image from generate_state_images
            img_path = make_image(state=state_name)

            # Generate the card
            generate_state_card_file(
                state_name=state_name,
                state_abbr=state_abbr,
                image_path=img_path,
                output_dir="state_cards",
                skip_if_exists=True,
            )


if __name__ == "__main__":
    main()
