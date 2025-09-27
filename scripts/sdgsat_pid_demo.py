"""Demo pipeline for running PID on SDGSAT-1 multi-spectral/thermal imagery.

This script prepares SDGSAT-1 multi-spectral (MII) data together with the
thermal infrared sensor (TIS) observation and then reuses the pretrained PID
model to synthesise a physics-guided thermal image.  It is designed as a demo
for quickly validating PID on remote sensing imagery without the need of
retraining on the new dataset.

Typical usage (paths correspond to the metadata in the user request):

python scripts/sdgsat_pid_demo.py \
    --mii-paths \
        E:/xunlei/KX10_MII_20250922_E121.24_N31.50_202500099050_L4B/\
        KX10_MII_20250922_E121.24_N31.50_202500099050_L4B_A.tif \
        E:/xunlei/KX10_MII_20250922_E121.24_N31.50_202500099050_L4B/\
        KX10_MII_20250922_E121.24_N31.50_202500099050_L4B_B.tif \
    --tir-path \
        E:/download/KX10_TIS_20250922_E121.37_N31.90_202500099051_L4B/\
        KX10_TIS_20250922_E121.37_N31.90_202500099051_L4B.tiff \
    --config configs/latent-diffusion/config.yaml \
    --checkpoint path/to/pid.ckpt \
    --outdir outputs/sdgsat-demo

The script performs the following steps:
1. Load the visible multi-spectral imagery and calibrate the DNs to radiance.
2. Derive a pseudo-RGB composite so that the pretrained PID model can be
   executed without architectural changes.
3. Load and calibrate the thermal image, optionally converting band 2 to a
   brightness temperature estimate and resampling it to match the visible grid.
4. Run the PID sampler to predict an infrared image and store the result
   alongside intermediate products (pseudo-RGB input, brightness temperature
   map, etc.).

All reprojection/resampling is handled through rasterio, therefore you need to
install GDAL/rasterio bindings in the Python environment before running the
script.  See the README for more information.
"""

import argparse
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
from omegaconf import OmegaConf
from rasterio.enums import Resampling
import rasterio
from rasterio.warp import reproject

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


PLANCK_CONSTANT = 6.62607015e-34
LIGHT_SPEED = 2.99792458e8
BOLTZMANN_CONSTANT = 1.380649e-23


def _ensure_iterable(value: Iterable[float], target_length: int) -> np.ndarray:
    """Tile/calibrate the gain/bias arrays so that they match the band count."""

    array = np.asarray(list(value), dtype=np.float32)
    if array.size == target_length:
        return array
    if array.size == 1:
        return np.full(target_length, array.item(), dtype=np.float32)
    if target_length % array.size == 0:
        return np.resize(array, target_length).astype(np.float32)
    raise ValueError(
        f"Cannot broadcast {array.size} coefficients to {target_length} bands.")


def apply_radiometric_calibration(
    data: np.ndarray,
    gains: Sequence[float],
    biases: Sequence[float],
) -> np.ndarray:
    """Apply the DN to radiance calibration using band-wise gains and biases."""

    if data.ndim != 3:
        raise ValueError("Expected array shaped as (bands, height, width).")
    gains = _ensure_iterable(gains, data.shape[0])[:, None, None]
    biases = _ensure_iterable(biases, data.shape[0])[:, None, None]
    return data.astype(np.float32) * gains + biases


def load_multispectral_stack(
    paths: Sequence[Path],
    gains: Sequence[float],
    biases: Sequence[float],
) -> Tuple[np.ndarray, dict]:
    """Load and radiometrically calibrate the multi-spectral stack."""

    stacked = []
    reference_profile = None
    for path in paths:
        with rasterio.open(path) as src:
            data = src.read(out_dtype=np.float32)
            if reference_profile is None:
                reference_profile = src.profile
                calibrated = apply_radiometric_calibration(data, gains, biases)
            else:
                calibrated = apply_radiometric_calibration(data, gains, biases)
                if (
                    src.transform != reference_profile["transform"]
                    or src.width != reference_profile["width"]
                    or src.height != reference_profile["height"]
                    or src.crs != reference_profile["crs"]
                ):
                    dest = np.empty(
                        (calibrated.shape[0],
                         reference_profile["height"],
                         reference_profile["width"]),
                        dtype=np.float32,
                    )
                    reproject(
                        calibrated,
                        dest,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=reference_profile["transform"],
                        dst_crs=reference_profile["crs"],
                        resampling=Resampling.bilinear,
                    )
                    calibrated = dest
            stacked.append(calibrated)
    if not stacked:
        raise ValueError("No multi-spectral files found.")
    data = np.concatenate(stacked, axis=0)
    return data, reference_profile


def percentile_stretch(image: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    """Apply percentile stretching for visualisation."""

    low_val, high_val = np.percentile(image, [low, high])
    stretched = np.clip((image - low_val) / max(high_val - low_val, 1e-6), 0, 1)
    return stretched


def pseudo_rgb_from_multispectral(
    stack: np.ndarray,
    band_centres_nm: Sequence[float],
    rgb_targets_nm: Tuple[float, float, float] = (660.0, 560.0, 490.0),
) -> np.ndarray:
    """Create a pseudo RGB composite using the closest bands to the target wavelengths."""

    band_centres = np.asarray(band_centres_nm, dtype=np.float32)
    if stack.shape[0] != band_centres.size:
        raise ValueError(
            "Band centre metadata does not match multi-spectral stack dimensions." )
    rgb_channels = []
    for target in rgb_targets_nm:
        index = int(np.argmin(np.abs(band_centres - target)))
        rgb_channels.append(stack[index])
    rgb = np.stack(rgb_channels, axis=0)
    rgb = percentile_stretch(rgb)
    return rgb


def radiance_to_brightness_temperature(
    radiance: np.ndarray,
    wavelength_um: float,
) -> np.ndarray:
    """Convert spectral radiance to brightness temperature via Planck's law."""

    wavelength_m = wavelength_um * 1e-6
    radiance = np.maximum(radiance, 1e-9)
    c1 = 2 * PLANCK_CONSTANT * LIGHT_SPEED ** 2
    c2 = PLANCK_CONSTANT * LIGHT_SPEED / BOLTZMANN_CONSTANT
    exponent = c1 / (wavelength_m ** 5 * radiance) + 1.0
    return c2 / (wavelength_m * np.log(exponent))


def load_brightness_temperature(
    path: Path,
    gains: Sequence[float],
    biases: Sequence[float],
    band_centres_um: Sequence[float],
    reference_profile: dict,
    temperature_band_index: int = 1,
) -> np.ndarray:
    """Load and resample the thermal brightness temperature to the reference grid."""

    with rasterio.open(path) as src:
        thermal = apply_radiometric_calibration(src.read(out_dtype=np.float32), gains, biases)
        thermal_band = thermal[temperature_band_index]
        brightness = radiance_to_brightness_temperature(
            thermal_band, band_centres_um[temperature_band_index]
        )
        dest = np.empty(
            (reference_profile["height"], reference_profile["width"]),
            dtype=np.float32,
        )
        reproject(
            brightness,
            dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=reference_profile["transform"],
            dst_crs=reference_profile["crs"],
            resampling=Resampling.bilinear,
        )
        return dest


def make_batch_from_rgb(rgb: np.ndarray, device: torch.device) -> dict:
    """Build the conditional batch for PID from a 3xHxW pseudo RGB array."""

    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError("Expected a 3 x H x W pseudo RGB array.")
    image = np.clip(rgb, 0.0, 1.0)
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image.transpose(1, 2, 0))

    crop = min(image.size)
    left = (image.width - crop) // 2
    upper = (image.height - crop) // 2
    image = image.crop((left, upper, left + crop, upper + crop))
    image = image.resize((512, 512), resample=Image.BICUBIC)

    tensor = np.asarray(image).astype(np.float32) / 255.0
    tensor = tensor[None].transpose(0, 3, 1, 2)
    tensor = torch.from_numpy(tensor)

    batch = {"conditional": tensor.to(device=device)}
    batch["conditional"] = batch["conditional"] * 2.0 - 1.0
    return batch


def save_array_as_png(array: np.ndarray, output_path: Path, cmap: str = "gray") -> None:
    """Store a single channel array as an 8-bit PNG for quick inspection."""

    array = percentile_stretch(array)
    array = (array * 255).astype(np.uint8)
    image = Image.fromarray(array)
    if cmap == "inferno":
        try:
            import matplotlib.pyplot as plt

            plt.imsave(output_path, array, cmap="inferno")
            return
        except Exception:
            pass
    image.save(output_path)


def run_pid_inference(
    batch: dict,
    model_config: Path,
    checkpoint: Path,
    device: torch.device,
    steps: int,
    ddim_eta: float,
) -> torch.Tensor:
    """Load the PID model, run DDIM sampling and return the predicted tensor."""

    config = OmegaConf.load(model_config)
    model = instantiate_from_config(config.model)
    checkpoint_data = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data["state_dict"], strict=False)
    model = model.to(device)
    sampler = DDIMSampler(model)

    with torch.no_grad():
        with model.ema_scope():
            c = model.cond_stage_model.encode(batch["conditional"])
            if c.shape[1] == 4:
                shape = (c.shape[1],) + c.shape[2:]
            else:
                shape = (c.shape[1] + 1,) + c.shape[2:]
            samples_ddim, _ = sampler.sample(
                S=steps,
                conditioning=c,
                batch_size=c.shape[0],
                shape=shape,
                verbose=False,
                ddim_eta=ddim_eta,
            )
            x_samples = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PID remote sensing demo using SDGSAT-1 imagery."
    )
    parser.add_argument(
        "--mii-paths",
        nargs="+",
        required=True,
        help="List of SDGSAT-1 multi-spectral (MII) GeoTIFF files.",
    )
    parser.add_argument(
        "--tir-path",
        required=True,
        help="SDGSAT-1 thermal infrared (TIS) GeoTIFF file.",
    )
    parser.add_argument("--config", required=True, help="PID config file path.")
    parser.add_argument("--checkpoint", required=True, help="PID checkpoint file.")
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for the demo products.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of DDIM sampling steps (default: 200).",
    )
    parser.add_argument(
        "--ddim-eta",
        type=float,
        default=0.0,
        help="DDIM eta value (default: 0.0).",
    )
    parser.add_argument(
        "--mii-gains",
        type=float,
        nargs="*",
        default=[
            0.052084676,
            0.038928845,
            0.025864978,
            0.017501881,
            0.016499392,
            0.021554446,
            0.015360482,
        ],
        help="Radiance gains for the MII bands.",
    )
    parser.add_argument(
        "--mii-biases",
        type=float,
        nargs="*",
        default=[0.0],
        help="Radiance biases for the MII bands.",
    )
    parser.add_argument(
        "--mii-band-centres",
        type=float,
        nargs="*",
        default=[400, 440, 490, 560, 660, 785, 850],
        help="Center wavelength (nm) of each MII band.",
    )
    parser.add_argument(
        "--tir-gains",
        type=float,
        nargs="*",
        default=[0.003947, 0.003946, 0.005329],
        help="Radiance gains for the TIS bands.",
    )
    parser.add_argument(
        "--tir-biases",
        type=float,
        nargs="*",
        default=[0.167126, 0.124622, 0.22253],
        help="Radiance biases for the TIS bands.",
    )
    parser.add_argument(
        "--tir-band-centres",
        type=float,
        nargs="*",
        default=[9.35, 10.73, 11.72],
        help="Center wavelength (μm) of each TIS band.",
    )
    parser.add_argument(
        "--temperature-band-index",
        type=int,
        default=1,
        help="Index of the TIS band to convert into brightness temperature.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mii_paths = [Path(p) for p in args.mii_paths]
    multispectral, profile = load_multispectral_stack(
        mii_paths, args.mii_gains, args.mii_biases
    )
    pseudo_rgb = pseudo_rgb_from_multispectral(
        multispectral, args.mii_band_centres
    )
    Image.fromarray(
        (pseudo_rgb.transpose(1, 2, 0) * 255).astype(np.uint8)
    ).save(outdir / "pseudo_rgb_preview.png")

    brightness_temperature = load_brightness_temperature(
        Path(args.tir_path),
        args.tir_gains,
        args.tir_biases,
        args.tir_band_centres,
        profile,
        temperature_band_index=args.temperature_band_index,
    )
    save_array_as_png(
        brightness_temperature,
        outdir / "brightness_temperature.png",
        cmap="inferno",
    )
    np.save(outdir / "brightness_temperature.npy", brightness_temperature)

    batch = make_batch_from_rgb(pseudo_rgb, device=device)
    prediction = run_pid_inference(
        batch=batch,
        model_config=Path(args.config),
        checkpoint=Path(args.checkpoint),
        device=device,
        steps=args.steps,
        ddim_eta=args.ddim_eta,
    )
    predicted_image = (
        prediction.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
    ).astype(np.uint8)
    Image.fromarray(predicted_image).save(outdir / "pid_prediction.png")


if __name__ == "__main__":
    main()
