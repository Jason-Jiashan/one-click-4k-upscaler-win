# realesrgan-4k-batch-win

Windows batch 4K upscaling wrappers for Real-ESRGAN.

## What This Repo Does

This project provides simple Windows `.cmd` launchers for batch image upscaling to 4K using Real-ESRGAN.

Two modes are provided:
- `run_4k_anime_pytorch.cmd`: anime / illustration images
- `run_4k_photo_pytorch.cmd`: real photos

## Files

- `batch_realesrgan_exact_4k.py`: main batch upscaling script
- `run_4k_anime_pytorch.cmd`: launcher for anime images
- `run_4k_photo_pytorch.cmd`: launcher for photos

## Requirements

- Windows
- NVIDIA GPU recommended
- Python 3.11+
- PyTorch with CUDA
- Real-ESRGAN dependencies

## Usage

1. Put your images in the project folder.
2. Double-click the correct `.cmd` file.
3. Results will be written to:
   - `sr_3840_exact_anime_pytorch/`
   - `sr_3840_exact_photo_pytorch/`

## Notes

- Existing 4K outputs are skipped automatically.
- Very large images may fail on low VRAM GPUs.
- This repo does not include model weights, virtual environments, or sample images.

## Upstream

This project depends on [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).

## License

This repository contains wrapper scripts only.
Real-ESRGAN is a separate upstream project under BSD-3-Clause.

