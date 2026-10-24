# SlimFormerPlusPlus

A [3D Slicer](https://www.slicer.org) extension that segments 15 abdominal
organs in CT with SlimFormer++. It runs on an ordinary CPU, with no GPU
needed (an NVIDIA GPU is used if present), on Windows, macOS and Linux.

**For research use only. Not a medical device; do not use the results for
diagnosis or treatment.**

![SlimFormer++ segmentation of the Panoramix sample CT in 3D Slicer](Screenshots/Screenshot01.png)

## Fast and accurate on a CPU

| CPU only | Time per scan | Mean Dice |
| --- | --- | --- |
| **SlimFormer++ Fast** | **~30 s** | **85.2 %** |
| TotalSegmentator `--fast` | ~45 s | 80.7 % |
| **SlimFormer++ Standard** | **~3 min** | **89.1 %** |
| TotalSegmentator | ~5.5 min | 84.5 % |

Laptop with an Apple M5 CPU (10 threads), both tools in 3D Slicer's Python.

## Installation

Requires 3D Slicer 5.10 or later.

1. Download this repository (green **Code** button → **Download ZIP**) and
   unzip it to a folder you will keep; Slicer loads the module from there.
2. In Slicer, open **View → Extensions Manager** and install the **PyTorch**
   and **NNUNet** extensions.
3. Open **Edit → Application Settings → Modules** and add the unzipped
   `SlimFormerPlusPlus` folder (the one that contains `SlimFormerPlusPlus.py`)
   to *Additional module paths*: click **Add**, or drag the folder onto the
   list. Click **OK**.
4. Restart Slicer.

The first **Segment** asks to install PyTorch and nnU-Net into Slicer; click
**OK**. This happens once. If Slicer then asks to restart, restart it and click
**Segment** again. The network weights (about 22 MB per mode) are downloaded
on first use.

<!-- TODO: after the extension is in the Extensions Index, replace steps 1-4 with
"In the Extensions Manager, install SlimFormerPlusPlus and restart Slicer." -->

## Usage

The extension has one module, **SlimFormer++** (category *Segmentation*): it
segments 15 abdominal organs in a CT volume into a segmentation.

1. Load an abdominal CT, or the sample **CTA abdomen (Panoramix)** from the
   Sample Data module.
2. Open the **SlimFormer++** module, select the CT and a mode: **Standard**
   (most accurate) or **Fast** (quickest).
3. Click **Segment**. Correct the result in the Segment Editor if needed.

- No internet? Download the model `.zip` from
  [Releases](https://github.com/WHHHHHY822/SlicerSlimFormerPlusPlus/releases) and
  use *Advanced → Install from file*.
- Images not in Hounsfield units: turn off *Advanced → Crop to body*.

## Model

- SlimFormer++ with window self-attention in encoder 1 (the Mamba-free
  implementation in the
  [SlimFormer++ repository](https://github.com/WHHHHHY822/SlimFormerPlusPlus)),
  6.0 M parameters, plain PyTorch.
- Trained with nnU-Net v2 on 200 AMOS2022 CT scans, on 1.5 mm (Standard) or
  3 mm (Fast) voxels.
- Mean Dice on all 100 AMOS2022 validation CT scans (15 structures): 89.3 %
  Standard, 84.4 % Fast.

## License

The Slicer module is under the [Apache License 2.0](LICENSE.txt). The network
definition (`SlimFormerPlusPlus/SlimFormerPlusPlusLib/model.py`) and the weights
are under the [SlimFormer++ Noncommercial Research License 1.0](LICENSE-MODEL.txt)
(noncommercial research use only). Third-party code: [NOTICE.md](NOTICE.md).

## How to cite

If you use this extension, please cite:

- SlimFormer++: <https://github.com/WHHHHHY822/SlimFormerPlusPlus>
- nnU-Net: Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net:
  a self-configuring method for deep learning-based biomedical image
  segmentation. *Nature Methods* 18, 203–211 (2021).
- AMOS: Ji Y, et al. AMOS: A large-scale abdominal multi-organ benchmark for
  versatile medical image segmentation. *NeurIPS 2022 Datasets and Benchmarks*.
