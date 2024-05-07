import os
from pathlib import Path
from typing import Optional

import torch

from .convert_hf_checkpoint import convert_hf_checkpoint
from .lightning_core_imports import RequirementCache
from .litgpt_utils import gated_repo_catcher, find_weight_files

_SAFETENSORS_AVAILABLE = RequirementCache("safetensors")
_HF_TRANSFER_AVAILABLE = RequirementCache("hf_transfer")


def download_from_hub(
    repo_id: Optional[str] = None,
    access_token: Optional[str] = os.getenv("HF_TOKEN"),
    tokenizer_only: bool = False,
    convert_checkpoint: bool = True,
    dtype: Optional[str] = None,
    checkpoint_dir: Path = Path("checkpoints"),
    model_name: Optional[str] = None,
) -> None:
    """Download weights or tokenizer data from the Hugging Face Hub.

    Arguments:
        repo_id: The repository ID in the format ``org/name`` or ``user/name``
            as shown in Hugging Face.
        access_token: Optional API token to access models with restrictions.
        tokenizer_only: Whether to download only the tokenizer files.
        convert_checkpoint: Whether to convert the checkpoint files to the
            LitGPT format after downloading.
        dtype: The data type to convert the checkpoint files to. If not
            specified, the weights will remain in the dtype they are downloaded in.
        checkpoint_dir: Where to save the downloaded files.
        model_name: The existing config name to use for this repo_id. This is
            useful to download alternative weights of existing architectures.

    Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file at
    https://github.com/Lightning-AI/litgpt/blob/main/LICENSE.
    """

    if repo_id is None:
        from .config import configs

        options = [
            f"{config['hf_config']['org']}/{config['hf_config']['name']}"
            for config in configs
        ]
        print("Please specify --repo_id <repo_id>. Available values:")
        print("\n".join(sorted(options, key=lambda x: x.lower())))
        return

    from huggingface_hub import snapshot_download

    # Locate files on the Huggingface Hub repository
    download_files = ["tokenizer*", "generation_config.json", "config.json"]
    from_safetensors = False
    if not tokenizer_only:
        # Get lists of *model* file names from the repo - either 'bin' or 'safetensor'
        bins, safetensors = find_weight_files(repo_id, access_token)
        if bins:
            # covers `.bin` files and `.bin.index.json`
            download_files.append("*.bin*")
        elif safetensors:
            if not _SAFETENSORS_AVAILABLE:
                raise ModuleNotFoundError(str(_SAFETENSORS_AVAILABLE))
            download_files.append("*.safetensors")
            from_safetensors = True
        else:
            raise ValueError(f"Couldn't find weight files for {repo_id}")

    import huggingface_hub._snapshot_download as download
    import huggingface_hub.constants as constants

    previous = constants.HF_HUB_ENABLE_HF_TRANSFER
    if _HF_TRANSFER_AVAILABLE and not previous:
        print("Setting HF_HUB_ENABLE_HF_TRANSFER=1")
        constants.HF_HUB_ENABLE_HF_TRANSFER = True
        download.HF_HUB_ENABLE_HF_TRANSFER = True

    directory = checkpoint_dir / repo_id  # Where the files will be downloaded
    # Download all specified files from the repository
    with gated_repo_catcher(repo_id, access_token):
        snapshot_download(
            repo_id,
            local_dir=directory,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=download_files,
            token=access_token,
        )

    constants.HF_HUB_ENABLE_HF_TRANSFER = previous
    download.HF_HUB_ENABLE_HF_TRANSFER = previous

    # convert safetensors to PyTorch binaries
    if from_safetensors:
        from safetensors import SafetensorError
        from safetensors.torch import load_file as safetensors_load

        print("Converting .safetensor files to PyTorch binaries (.bin)")
        for safetensor_path in directory.glob("*.safetensors"):
            bin_path = safetensor_path.with_suffix(".bin")
            try:
                result = safetensors_load(safetensor_path)
            except SafetensorError as e:
                raise RuntimeError(
                    f"{safetensor_path} is likely corrupted. Please try to re-download it."
                ) from e
            print(f"{safetensor_path} --> {bin_path}")
            torch.save(result, bin_path)
            os.remove(safetensor_path)

    if convert_checkpoint and not tokenizer_only:
        print("Converting checkpoint files to LitGPT format.")
        convert_hf_checkpoint(
            checkpoint_dir=directory, dtype=dtype, model_name=model_name
        )
