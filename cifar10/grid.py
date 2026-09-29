"""Display the first 64 generated samples in filename order."""

from pathlib import Path

import click
from PIL import Image, ImageOps


@click.command()
@click.option(
    "--images", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
def main(images, out):
    paths = sorted(images.rglob("*.png"))[:64]
    if len(paths) < 64:
        raise click.ClickException("The grid requires at least 64 images")
    grid = Image.new("RGB", (8 * 32, 8 * 32), "white")
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            if image.size != (32, 32):
                raise click.ClickException("Expected 32×32 images")
            grid.paste(image.convert("RGB"), ((index % 8) * 32, (index // 8) * 32))
    out.parent.mkdir(parents=True, exist_ok=True)
    ImageOps.contain(grid, (1024, 1024)).save(out)


if __name__ == "__main__":
    main()
