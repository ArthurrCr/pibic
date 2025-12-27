"""Module for downloading CloudSEN12 dataset from Hugging Face Hub."""

import os
from typing import List

from huggingface_hub import hf_hub_download


REPO_ID = "tacofoundation/CloudSEN12"
REPO_TYPE = "dataset"

DATASET_PARTS = {
    "L1C": ["cloudsen12-l1c.0000.part.taco", "cloudsen12-l1c.0004.part.taco"],
    "L2A": ["cloudsen12-l2a.0000.part.taco", "cloudsen12-l2a.0004.part.taco"],
    "extra": ["cloudsen12-extra.0000.part.taco"],
}


def download_cloudsen12(
    local_dir: str = "./data/dados",
    dataset_type: str = "",
) -> List[str]:
    """
    Download CloudSEN12+ dataset parts to the specified directory.

    Downloads only parts 0 and 4 (high-quality labels) to reduce download size.
    Creates the target directory if it does not exist.

    Args:
        local_dir: Directory path where files will be saved.
        dataset_type: Type of dataset to download. Must be one of:
            - "L1C": Level-1C products
            - "L2A": Level-2A products
            - "extra": Extra dataset files

    Returns:
        List of file paths for the downloaded files.

    Raises:
        ValueError: If dataset_type is not valid.
    """
    if dataset_type not in DATASET_PARTS:
        valid_types = ", ".join(DATASET_PARTS.keys())
        raise ValueError(
            f"Invalid dataset_type '{dataset_type}'. Must be one of: {valid_types}"
        )

    os.makedirs(local_dir, exist_ok=True)

    downloaded_files = []
    filenames = DATASET_PARTS[dataset_type]

    for filename in filenames:
        file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type=REPO_TYPE,
            local_dir=local_dir,
        )
        downloaded_files.append(file_path)

    return downloaded_files