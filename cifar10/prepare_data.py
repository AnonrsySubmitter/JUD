# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from NVIDIA EDM (https://github.com/NVlabs/edm), CC BY-NC-SA 4.0; see cifar10/LICENSE-NVIDIA.txt.

"""Convert the official CIFAR-10 Python archive to the EDM image format."""

import io
import json
import pickle
import tarfile
import zipfile

import click
from PIL import Image
from tqdm import tqdm


@click.command()
@click.option("--source", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--dest", required=True, type=click.Path(dir_okay=False))
def main(source, dest):
    """Use all 50,000 training images from cifar-10-python.tar.gz."""
    from pathlib import Path

    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    with (
        tarfile.open(source, "r:gz") as archive,
        zipfile.ZipFile(dest, "x", compression=zipfile.ZIP_STORED) as output,
    ):
        index = 0
        for batch in tqdm(range(1, 6), desc="Training batches"):
            with archive.extractfile(f"cifar-10-batches-py/data_batch_{batch}") as file:
                data = pickle.load(file, encoding="latin1")
            images = data["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            if len(images) != 10000:
                raise click.ClickException("Expected 10,000 images per training batch")
            for image in images:
                name = f"{index:08d}"
                buffer = io.BytesIO()
                Image.fromarray(image).save(buffer, format="PNG", compress_level=0, optimize=False)
                output.writestr(f"{name[:5]}/img{name}.png", buffer.getvalue())
                index += 1
        output.writestr("dataset.json", json.dumps({"labels": None}))


if __name__ == "__main__":
    main()
