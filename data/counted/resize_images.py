from pathlib import Path
from PIL import Image

# Ordner, in dem dieses Script liegt
SCRIPT_DIR = Path(__file__).resolve().parent

# Eingabeordner: aktueller counted-Ordner
INPUT_DIR = SCRIPT_DIR

# Ausgabeordner: neben counted
OUTPUT_BASE_DIR = SCRIPT_DIR.parent / "benchmark_scaled"

# Zielbreiten
TARGET_WIDTHS = [5100, 4096, 2048, 1024, 512, 256]

# Originalauflösung
ORIGINAL_WIDTH = 5100
ORIGINAL_HEIGHT = 7016

# Unterstützte Bildformate
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

JPEG_QUALITY = 95


def calculate_height(target_width: int) -> int:
    return round(target_width * ORIGINAL_HEIGHT / ORIGINAL_WIDTH)


def resize_image(input_file: Path, output_file: Path, size: tuple[int, int]) -> None:
    with Image.open(input_file) as img:
        img = img.convert("RGB")
        resized = img.resize(size, Image.Resampling.LANCZOS)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        resized.save(
            output_file,
            quality=JPEG_QUALITY,
            optimize=True
        )


def main():
    image_files = [
        file for file in INPUT_DIR.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"Eingabeordner: {INPUT_DIR}")
    print(f"Ausgabeordner: {OUTPUT_BASE_DIR}")
    print(f"Gefundene Bilder: {len(image_files)}")

    for target_width in TARGET_WIDTHS:
        target_height = calculate_height(target_width)
        target_size = (target_width, target_height)

        resolution_folder = OUTPUT_BASE_DIR / f"{target_width}x{target_height}"

        print(f"\nErzeuge Auflösung: {target_width}x{target_height}")

        for image_file in image_files:
            output_file = resolution_folder / image_file.name

            try:
                resize_image(image_file, output_file, target_size)
                print(f"  OK: {image_file.name}")

            except Exception as e:
                print(f"  Fehler bei {image_file.name}: {e}")


if __name__ == "__main__":
    main()