# SlimFormerPlusPlus

A [3D Slicer](https://www.slicer.org) extension that segments 15 abdominal
organs in CT images with SlimFormer++.

**For research use only. This is not a medical device. Do not use the results
for diagnosis or treatment.**

<!-- TODO: screenshot from a real Slicer run (Screenshots/Screenshot01.png) -->

## What it does

- **Input:** a CT volume of the abdomen, loaded in 3D Slicer from DICOM, NIfTI,
  NRRD or any other supported format.
- **Output:** a segmentation with up to 15 structures: spleen, right kidney,
  left kidney, gallbladder, esophagus, liver, stomach, aorta, inferior vena
  cava, pancreas, right adrenal gland, left adrenal gland, duodenum, bladder,
  prostate/uterus.
- Runs on an NVIDIA GPU or on the CPU, on Windows, macOS and Linux. Nothing
  has to be compiled.

## Installation

Requires 3D Slicer 5.10 or later.

1. Open the Extensions Manager, search for **SlimFormerPlusPlus**, install it and
   restart Slicer. The PyTorch and NNUNet extensions are installed with it.
2. The first time you click **Segment**, the module asks to install PyTorch
   and nnU-Net into Slicer's Python environment (needs internet; this can take
   several minutes) and then downloads the network weights (about 22 MB).

<!-- TODO: remove this section after the extension is in the Extensions Index -->
Until the extension is listed in the Extensions Manager:

1. Install the **PyTorch** and **NNUNet** extensions from the Extensions Manager
   and restart Slicer.
2. Download this repository (Code → Download ZIP) and unpack it.
3. In Slicer, open Edit → Application Settings → Modules, add the unpacked
   `SlimFormerPlusPlus` folder (the one that contains `SlimFormerPlusPlus.py`) to
   *Additional module paths*, and restart Slicer.

## Tutorial

1. Open the **Sample Data** module and load **CTA abdomen (Panoramix)**, or load
   your own abdominal CT.
2. Open the **SlimFormer++** module (category *Segmentation*).
3. Select the CT as *Input CT volume*. Leave *Output segmentation* at
   *Create new segmentation*.
4. Click **Segment**. Progress is shown in the progress bar and the *Log*
   section.
5. The organs appear in the slice views and, with *Show in 3D* enabled, as
   surfaces in the 3D view. Use the **Segment Editor** to correct them or the
   **Segment Statistics** module to measure volumes.

### How long does it take?

The image is resampled to 0.68 × 0.68 × 2.0 mm and processed in overlapping
64 × 192 × 160 voxel patches inside the bounding box of the body (*Crop to
body*), so the time grows with the scanned body volume. Measured per scan as one
complete run (Python start, model loading, inference, writing the result), without
test-time augmentation:

| Hardware | Scans | Crop to body (default) | Without cropping |
| --- | --- | --- | --- |
| NVIDIA GPU (data-centre GPU), PyTorch 2.6 | 100 AMOS2022 validation CTs | median 43 s (max 72 s) | median 62 s (max 108 s) |
| NVIDIA GPU (data-centre GPU), PyTorch 2.6 | 100 AbdomenCT-1K CTs | median 35 s (max 163 s) | median 46 s (max 192 s) |
| CPU, 8 threads (ARM server, PyTorch 2.2) | 5 AMOS2022 validation CTs | mean 13 min (11–18 min) | mean 24 min (14–38 min) |

On the GPU most of the time is spent resampling on the CPU, so very large scans
(about 1000 thin slices) take longest. Consumer GPUs are slower than the GPU above
but still take only minutes. On macOS, 3D Slicer currently runs through Rosetta 2
and uses the CPU, so expect tens of minutes per scan. Peak memory (RAM) was 5–8 GB
for the AMOS2022 scans and up to 13 GB for the largest AbdomenCT-1K scans; on a GPU
up to about 10 GB of GPU memory is used. 16 GB of RAM or more is recommended.

## Settings (Advanced section)

- **Device:** automatic (GPU if PyTorch finds a CUDA GPU), GPU or CPU.
- **Crop to body** (on by default): segment only the bounding box of the
  patient's body (voxels above −500 HU, largest connected region, plus a margin
  of about 1 cm), not the air and table around it. Everything outside the box is
  background. Turn it off for images that are not in Hounsfield units or when
  structures outside the body outline are expected.
- **Test-time augmentation:** averages the prediction over mirrored copies of
  the image. About 8 times slower; small accuracy gain.
- **CPU threads:** number of threads on the CPU (automatic by default).
- **Install model from file:** install the network weights from a downloaded
  model `.zip` file, for computers without internet access.
- **Keep temporary files:** keep the exchanged files for troubleshooting.

## Model

- **Network:** SlimFormer++ with window self-attention in encoder 1 (the
  Mamba-free implementation in the
  [SlimFormer++ repository](https://github.com/WHHHHHY822/SlimFormerPlusPlus)),
  6.0 M parameters. It needs only PyTorch, no custom CUDA kernels.
- **Training:** nnU-Net v2 `3d_fullres` on AMOS2022 CT (200 training scans),
  fine-tuned for 100 epochs from the released SlimFormer++ AMOS2022 CT weights.
- **Validation:** the 100 official AMOS2022 validation CT scans, nnU-Net's
  validation with mirroring, Dice in %:

| Structure | This model | SlimFormer++ (paper model) |
| --- | --- | --- |
| Spleen | 97.04 | 97.16 |
| Right kidney | 96.62 | 96.81 |
| Left kidney | 96.43 | 96.56 |
| Gallbladder | 84.72 | 84.81 |
| Esophagus | 87.68 | 88.22 |
| Liver | 97.40 | 97.48 |
| Stomach | 92.41 | 93.04 |
| Aorta | 95.84 | 95.99 |
| Inferior vena cava | 92.01 | 92.46 |
| Pancreas | 88.74 | 88.99 |
| Right adrenal gland | 81.42 | 81.84 |
| Left adrenal gland | 82.25 | 82.44 |
| Duodenum | 84.75 | 84.98 |
| Bladder | 90.89 | 91.19 |
| Prostate/uterus | 83.84 | 84.03 |
| **Mean** | **90.14** | **90.40** |

  The paper model uses a selective-scan (Mamba) mixer in encoder 1, which
  needs compiled CUDA extensions and does not run on Windows, macOS or CPUs;
  this extension uses the Mamba-free implementation instead.
- **Pre- and post-processing:** nnU-Net v2. Before inference the image axes are
  permuted/flipped (never interpolated) to the orientation of the training
  data, and the result is mapped back to the original voxel grid. The
  conversion of the network output back to the original grid gives the same
  labels as nnU-Net but needs less memory.
- **Limitations:** trained only on AMOS2022 abdominal CT; not
  validated on other body regions, contrast phases or scanners outside that
  data, on MRI, or on pathology-heavy cases. Always review the result.

## License

The Slicer module and its helper code are under the
[Apache License 2.0](LICENSE.txt). The network definition
(`SlimFormerPlusPlus/SlimFormerPlusPlusLib/model.py`) and the network weights are under the
[SlimFormer++ Noncommercial Research License 1.0](LICENSE-MODEL.txt):
noncommercial research use only. See [NOTICE.md](NOTICE.md) for third-party code.

## Citation

If you use this extension, please cite:

- SlimFormer++: <https://github.com/WHHHHHY822/SlimFormerPlusPlus>
- nnU-Net: Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net:
  a self-configuring method for deep learning-based biomedical image
  segmentation. *Nature Methods* 18, 203–211 (2021).
- AMOS: Ji Y, et al. AMOS: A large-scale abdominal multi-organ benchmark for
  versatile medical image segmentation. *NeurIPS 2022 Datasets and Benchmarks*.
