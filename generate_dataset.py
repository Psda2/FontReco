import os
import string
import itertools
from PIL import Image, ImageDraw, ImageFont
import random
import matplotlib.font_manager as fm

# ======================================
# Configuration
# ======================================

OUTPUT_DIR = 'generated_font_dataset'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Characters to use for generation (A-Z, a-z, 0-9)
CHARACTERS = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)

# Font sizes to use for each image (these are the font sizes, not the image size)
FONT_SIZES = [32, 40, 54]

# Number of two-letter combinations for testing
TEST_COMBINATION_COUNT = 1000  # Choose a fixed number of two-letter combinations for testing

# ======================================
# Helper Functions
# ======================================

def get_installed_fonts():
    """
    Retrieve a list of installed font paths on the system.
    """
    return fm.findSystemFonts(fontpaths=None, fontext='ttf')

def render_text(text, font_path, font_size, image_size=(128, 128)):
    """
    Renders text as an image using the specified font size within a fixed image size.
    """
    image = Image.new('L', image_size, color=255)  # Grayscale image with white background
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"Error loading font: {e}")
        font = ImageFont.load_default()

    # Using draw.textbbox to calculate the size of the text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)

    draw.text(position, text, fill=0, font=font)  # Black text
    return image

def save_image(image, folder, text, font_size):
    """
    Saves the image to the specified folder with a file name derived from the text and font size.
    """
    os.makedirs(folder, exist_ok=True)
    file_name = f"{text}_font{font_size}.png"
    image.save(os.path.join(folder, file_name))

# ======================================
# Generate Dataset
# ======================================

def generate_single_letter_images(font_path, font_name):
    """
    Generate images for each single character for training.
    """
    for char in CHARACTERS:
        for font_size in FONT_SIZES:
            image = render_text(char, font_path, font_size)
            save_image(image, os.path.join(OUTPUT_DIR, 'train', 'letters', font_name), char, font_size)

def generate_two_letter_combinations(font_path, font_name):
    """
    Generate images for each combination of two characters for both training and testing.
    """
    combinations = list(itertools.product(CHARACTERS, repeat=2))
    random.shuffle(combinations)  # Shuffle to randomize

    test_combinations = combinations[:TEST_COMBINATION_COUNT]
    train_combinations = combinations[TEST_COMBINATION_COUNT:]

    # Generate training images
    for combo in train_combinations:
        text = ''.join(combo)
        for font_size in FONT_SIZES:
            image = render_text(text, font_path, font_size)
            save_image(image, os.path.join(OUTPUT_DIR, 'train', 'combinations', font_name), text, font_size)

    # Generate testing images
    for combo in test_combinations:
        text = ''.join(combo)
        for font_size in FONT_SIZES:
            image = render_text(text, font_path, font_size)
            save_image(image, os.path.join(OUTPUT_DIR, 'test', 'combinations', font_name), text, font_size)

# ======================================
# Main
# ======================================

if __name__ == "__main__":
    # Get installed fonts
    installed_fonts = get_installed_fonts()
    if not installed_fonts:
        print("No fonts found on the system.")
        exit(1)

    while True:
        print("\nAvailable Fonts:")
        for i, font in enumerate(installed_fonts):
            print(f"{i + 1}: {font}")

        # Ask the user to select a font by index
        while True:
            try:
                font_index = int(input("Select a font by number (or 0 to exit): ")) - 1
                if font_index == -1:  # User wants to exit
                    print("Exiting the program.")
                    exit(0)
                if font_index < 0 or font_index >= len(installed_fonts):
                    raise IndexError
                break
            except (ValueError, IndexError):
                print("Invalid selection. Please enter a valid number corresponding to the available fonts.")

        font_path = installed_fonts[font_index]  # Get the selected font path

        # Extract font name for subfolder naming
        font_name = os.path.splitext(os.path.basename(font_path))[0]  # Get font name without extension

        # Generate single-letter images (for training only)
        print("Generating single-letter images...")
        generate_single_letter_images(font_path, font_name)

        # Generate two-letter combination images (for both training and testing)
        print("Generating two-letter combination images...")
        generate_two_letter_combinations(font_path, font_name)

        print(f"Dataset generation complete for font: {font_name}.")
