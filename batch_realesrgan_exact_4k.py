from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

ROOT = Path(__file__).resolve().parent
REPO_DIR = ROOT / "Real-ESRGAN-official"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from realesrgan import RealESRGANer  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SKIP_NAMES = {"test_out.png"}
MODELS = {
    "anime": {
        "name": "RealESRGAN_x4plus_anime_6B",
        "url": (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/"
            "RealESRGAN_x4plus_anime_6B.pth"
        ),
        "num_block": 6,
    },
    "photo": {
        "name": "RealESRGAN_x4plus",
        "url": (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
            "RealESRGAN_x4plus.pth"
        ),
        "num_block": 23,
    },
}


@dataclass
class ItemResult:
    name: str
    original_width: int
    original_height: int
    output_width: int
    output_height: int
    original_bytes: int
    output_bytes: int
    passes: int
    output_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT)
    parser.add_argument("--mode", choices=sorted(MODELS), default="anime")
    parser.add_argument("--target-long", type=int, default=3840)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--temp-dir", type=str, default=None)
    parser.add_argument("--include", nargs="*", default=None)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def get_image_size(path: Path) -> tuple[int, int]:
    img = read_cv(path)
    return img.shape[1], img.shape[0]


def list_images(base: Path, includes: set[str] | None) -> list[Path]:
    items: list[Path] = []
    for path in sorted(base.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        if path.name in SKIP_NAMES:
            continue
        if includes is not None and path.name not in includes:
            continue
        items.append(path)
    return items


def any_existing_output(base: Path, stem: str, target_long: int) -> Path | None:
    pattern = f"sr_{target_long}_exact_*_pytorch/{stem}_SR{target_long}.png"
    for path in sorted(base.glob(pattern)):
        if path.is_file():
            return path
    return None


def build_upsampler(weights_dir: Path, mode: str) -> RealESRGANer:
    model_info = MODELS[mode]
    model_name = model_info["name"]
    model_url = model_info["url"]
    weights_dir.mkdir(parents=True, exist_ok=True)
    weight_path = weights_dir / f"{model_name}.pth"
    if not weight_path.exists():
        load_file_from_url(url=model_url, model_dir=str(weights_dir), progress=True, file_name=None)

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=int(model_info["num_block"]),
        num_grow_ch=32,
        scale=4,
    )
    use_cuda = torch.cuda.is_available()
    return RealESRGANer(
        scale=4,
        model_path=str(weight_path),
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=use_cuda,
        gpu_id=0 if use_cuda else None,
    )


def read_cv(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"Could not read image bytes: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def resize_exact(img, width: int, height: int):
    if img.shape[1] == width and img.shape[0] == height:
        return img
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)


def write_png(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError(f"Could not write image: {path}")
    encoded.tofile(str(path))


def process_one(
    upsampler: RealESRGANer,
    image_path: Path,
    target_long: int,
    out_dir: Path,
    temp_dir: Path,
    keep_temp: bool,
) -> ItemResult:
    orig_w, orig_h = get_image_size(image_path)
    orig_long = max(orig_w, orig_h)
    ratio = target_long / orig_long
    target_w = max(1, round(orig_w * ratio))
    target_h = max(1, round(orig_h * ratio))
    current_path = image_path
    per_temp = temp_dir / image_path.stem
    per_temp.mkdir(parents=True, exist_ok=True)
    passes = 0

    while True:
        cur_w, cur_h = get_image_size(current_path)
        cur_long = max(cur_w, cur_h)
        need = target_long / cur_long
        input_img = read_cv(current_path)
        if passes > 0 and need <= 1.5:
            final_img = resize_exact(input_img, target_w, target_h)
            break

        passes += 1
        if need <= 4:
            output, _ = upsampler.enhance(input_img, outscale=need)
            final_img = resize_exact(output, target_w, target_h)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break

        output, _ = upsampler.enhance(input_img, outscale=4)
        temp_path = per_temp / f"{image_path.stem}_pass{passes}.png"
        write_png(temp_path, output)
        current_path = temp_path
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = out_dir / f"{image_path.stem}_SR3840.png"
    write_png(out_path, final_img)

    if not keep_temp:
        shutil.rmtree(per_temp, ignore_errors=True)

    out_w, out_h = get_image_size(out_path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ItemResult(
        name=image_path.name,
        original_width=orig_w,
        original_height=orig_h,
        output_width=out_w,
        output_height=out_h,
        original_bytes=image_path.stat().st_size,
        output_bytes=out_path.stat().st_size,
        passes=passes,
        output_path=str(out_path),
    )


def main() -> int:
    args = parse_args()
    base = args.base.resolve()
    output_dir_name = args.output_dir or f"sr_{args.target_long}_exact_{args.mode}_pytorch"
    temp_dir_name = args.temp_dir or f"_tmp_sr_{args.target_long}_exact_{args.mode}_pytorch"
    out_dir = base / output_dir_name
    temp_dir = base / temp_dir_name
    includes = set(args.include) if args.include else None
    images = list_images(base, includes)
    if not images:
        print("No images found.")
        return 1

    pending_images: list[Path] = []
    skipped: list[str] = []
    if args.force:
        pending_images = images
    else:
        for image_path in images:
            out_path = out_dir / f"{image_path.stem}_SR3840.png"
            if out_path.exists():
                print(f"Skip existing: {image_path.name}")
                skipped.append(image_path.name)
                continue
            existing_any = any_existing_output(base, image_path.stem, args.target_long)
            if existing_any is not None:
                print(f"Skip existing in another output dir: {image_path.name} -> {existing_any}")
                skipped.append(image_path.name)
                continue
            pending_images.append(image_path)

    print(f"Base: {base}")
    print(f"Mode: {args.mode}")
    print(f"Output: {out_dir}")
    print(f"Temp: {temp_dir}")
    print(f"Images found: {len(images)}")
    print(f"Images pending: {len(pending_images)}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    if not pending_images:
        print("Nothing new to process.")
        return 0

    torch.set_grad_enabled(False)

    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    upsampler = build_upsampler(REPO_DIR / "weights", args.mode)

    results: list[ItemResult] = []

    for image_path in pending_images:

        print("=" * 80)
        print(f"Processing: {image_path.name}")
        try:
            result = process_one(
                upsampler=upsampler,
                image_path=image_path,
                target_long=args.target_long,
                out_dir=out_dir,
                temp_dir=temp_dir,
                keep_temp=args.keep_temp,
            )
        except Exception as exc:
            print(f"Failed: {image_path.name} -> {exc}")
            continue

        print(
            f"Done: {result.original_width}x{result.original_height} -> "
            f"{result.output_width}x{result.output_height} | "
            f"{result.output_bytes} bytes | passes={result.passes}"
        )
        results.append(result)

    report = {
        "target_long": args.target_long,
        "mode": args.mode,
        "model_name": MODELS[args.mode]["name"],
        "output_dir": str(out_dir),
        "temp_dir": str(temp_dir),
        "result_count": len(results),
        "pending_count": len(pending_images),
        "skipped": skipped,
        "results": [asdict(item) for item in results],
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=" * 80)
    print(f"Completed: {len(results)}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
