python -m generate_state_cards
python -m generate_city_cards
python -m generate_country_cards
python -m generate_world_city_cards

python -m svg2png_batch ./state_cards ./state_cards_png --width 750
python -m svg2png_batch ./city_cards ./city_cards_png --width 750
python -m svg2png_batch ./country_cards ./country_cards_png --width 750
python -m svg2png_batch ./world_city_cards ./world_city_cards_png --width 750