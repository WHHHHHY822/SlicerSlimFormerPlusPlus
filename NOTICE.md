# Licenses and third-party notices

This repository contains material under two licenses.

| Material | License |
| --- | --- |
| Everything not listed below (Slicer module, installer, pre/post-processing glue, tests) | Apache License 2.0, [LICENSE.txt](LICENSE.txt) |
| `SlimFormerPlusPlus/SlimFormerPlusPlusLib/model.py` (SlimFormer++ network definition) | SlimFormer++ Noncommercial Research License 1.0, [LICENSE-MODEL.txt](LICENSE-MODEL.txt) |
| Network weights downloaded by the module | SlimFormer++ Noncommercial Research License 1.0, [LICENSE-MODEL.txt](LICENSE-MODEL.txt) |

The network and its weights may be used for noncommercial research only.
The module is not a medical device and must not be used for clinical diagnosis
or treatment.

`model.py` is a modified copy of `nnUNetTrainer_SlimFormerPlusPlus_MambaFree.py`
from [SlimFormerPlusPlus](https://github.com/WHHHHHY822/SlimFormerPlusPlus):
the nnU-Net trainer was removed, the imports of MONAI and MedNeXt blocks now
point to `blocks.py` and the network class is named `SlimFormerPlusPlusAttention`.

`SlimFormerPlusPlus/SlimFormerPlusPlusLib/blocks.py` contains code copied or adapted from:

- [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt), revision
  `0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba`, Copyright German Cancer Research
  Center (DKFZ), Apache License 2.0 (`MedNeXtBlock`, `MedNeXtDownBlock`,
  `MedNeXtUpBlock`; unused code paths removed).
- [MONAI](https://github.com/Project-MONAI/MONAI) 1.4.0, Copyright MONAI
  Consortium, Apache License 2.0 (`UnetrBasicBlock` with a residual block,
  `UnetOutBlock`; re-implemented with the same structure).

At run time the module uses, without redistributing them,
[nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) (Apache License 2.0),
[PyTorch](https://github.com/pytorch/pytorch) (BSD-style) and
[einops](https://github.com/arogozhnikov/einops) (MIT), installed through the
3D Slicer PyTorch and NNUNet extensions.

The network weights were trained on [AMOS2022](https://zenodo.org/records/7262581)
(Ji et al., NeurIPS 2022 Datasets and Benchmarks).
