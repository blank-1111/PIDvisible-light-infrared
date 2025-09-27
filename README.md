# PID: Physics-Informed Diffusion Model for Infrared Image Generation

<img src="PID.png" alt="PID" style="zoom:50%;" />

## Update 

* 2025/05 The paper is accepted by Pattern Recognition: https://doi.org/10.1016/j.patcog.2025.111816
* We have released our code.

## Environment

It is recommended to install the environment with environment.yaml. 

```bash
conda env create --file=environment.yaml
```

## Datasets

Download **KAIST** dataset from https://github.com/SoonminHwang/rgbt-ped-detection

Download **FLIRv1** dataset from https://www.flir.com/oem/adas/adas-dataset-form/

Download **VEDAI** dataset from https://downloads.greyc.fr/vedai/

We adopt the official dataset split in our experiments.

## Checkpoint

VQGAN can be downloaded from https://ommer-lab.com/files/latent-diffusion/vq-f8.zip (Other GAN models can be downloaded from https://github.com/CompVis/latent-diffusion).

TeVNet and PID heckpoints can be found in [HuggingFace](https://huggingface.co/FerrisMao/PID).

## Evaluation

Use the shellscript to evaluate. `indir` is the input directory of visible RGB images, `outdir` is the output directory of translated infrared images, `config` is the chosen config in `configs/latent-diffusion/config.yaml`.  We prepare some RGB images in `dataset/KAIST` for quick evaluation.

```sh
bash run_test_kaist512_vqf8.sh
```

## SDGSAT-1 Remote Sensing Demo

We provide a standalone script that demonstrates how to apply PID to
SDGSAT-1 multi-spectral (MII) and thermal infrared (TIS) imagery.  The
script performs radiometric calibration, resamples the thermal band to match
the multi-spectral grid, and runs the pretrained PID checkpoint to predict a
thermal image using the pseudo-RGB composite as the condition.

1. Install `rasterio` (GDAL bindings are required).  If you are using the
   provided Conda environment you can simply run

   ```bash
   conda install -c conda-forge rasterio
   ```

2. Execute the demo script with the SDGSAT-1 GeoTIFF files and a pretrained
   PID checkpoint:

   ```bash
   python scripts/sdgsat_pid_demo.py \
       --mii-paths \
           E:/xunlei/KX10_MII_20250922_E121.24_N31.50_202500099050_L4B/KX10_MII_20250922_E121.24_N31.50_202500099050_L4B_A.tif \
           E:/xunlei/KX10_MII_20250922_E121.24_N31.50_202500099050_L4B/KX10_MII_20250922_E121.24_N31.50_202500099050_L4B_B.tif \
       --tir-path \
           E:/download/KX10_TIS_20250922_E121.37_N31.90_202500099051_L4B/KX10_TIS_20250922_E121.37_N31.90_202500099051_L4B.tiff \
       --config configs/latent-diffusion/config.yaml \
       --checkpoint /path/to/pid.ckpt \
       --outdir outputs/sdgsat-demo
   ```

   The script will create a pseudo-RGB preview, save the brightness
   temperature estimate derived from the TIS band, and export the PID
   prediction inside the output directory.  You can further customise the
   radiometric coefficients via command-line arguments if the metadata of the
   provided imagery differs from the default values.

## Train

### Dataset preparation

Prepare corresponding RGB and infrared images with same names in two directories.

### Stage 1: Train TeVNet

```bash
cd TeVNet
bash shell/train.sh
```

### Stage 2: Train PID

To accelerate training, we recommend using our pretrained model. 

```bash
bash shell/run_train_kaist512_vqf8.sh
```

## Acknowledgements

Our code is built upon [LDM](https://github.com/CompVis/latent-diffusion) and [HADAR](https://github.com/FanglinBao/HADAR). We thank the authors for their excellent work.

## Citation

If you find this work is helpful in your research, please consider citing our paper:

```
@article{mao2026pid,
  title={PID: physics-informed diffusion model for infrared image generation},
  author={Mao, Fangyuan and Mei, Jilin and Lu, Shun and Liu, Fuyang and Chen, Liang and Zhao, Fangzhou and Hu, Yu},
  journal={Pattern Recognition},
  volume={169},
  pages={111816},
  year={2026},
  publisher={Elsevier}
}
```

If you have any question, feel free to contact maofangyuan23s@ict.ac.cn.
