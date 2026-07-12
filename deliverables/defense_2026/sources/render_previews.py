import argparse
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw


def render(pdf_path: Path, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(pdf_path)
    page_images = []

    for index, page in enumerate(document):
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
        page_path = output_dir / f"{prefix}-{index + 1:02d}.png"
        pixmap.save(page_path)
        image = Image.open(page_path).convert("RGB")
        page_images.append(image)

    for sheet_index in range(0, len(page_images), 4):
        group = page_images[sheet_index:sheet_index + 4]
        thumb_w = 760 if group[0].width > group[0].height else 520
        thumb_h = round(group[0].height * thumb_w / group[0].width)
        gap = 28
        label_h = 34
        canvas = Image.new("RGB", (thumb_w * 2 + gap * 3, (thumb_h + label_h) * 2 + gap * 3), "#D9DEE0")
        draw = ImageDraw.Draw(canvas)
        for offset, image in enumerate(group):
            row, col = divmod(offset, 2)
            x = gap + col * (thumb_w + gap)
            y = gap + row * (thumb_h + label_h + gap)
            thumb = image.copy()
            thumb.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (x, y + label_h))
            draw.text((x, y + 6), f"Page {sheet_index + offset + 1}", fill="#202426")
        contact_path = output_dir / f"{prefix}-contact-{sheet_index // 4 + 1:02d}.png"
        canvas.save(contact_path)

    print(f"{pdf_path.name}: {len(page_images)} pages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("prefix")
    args = parser.parse_args()
    render(args.pdf, args.output_dir, args.prefix)
