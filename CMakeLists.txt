cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlimFormerPlusPlus)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/WHHHHHY822/SlicerSlimFormerPlusPlus")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "WHHHHHY822")
set(EXTENSION_DESCRIPTION "Automatic segmentation of 15 abdominal organs in CT with SlimFormer++ (Mamba-free implementation, trained on AMOS2022). The network weights are for noncommercial research use only. Not for clinical use.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/WHHHHHY822/SlicerSlimFormerPlusPlus/main/SlimFormerPlusPlus.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/WHHHHHY822/SlicerSlimFormerPlusPlus/main/Screenshots/Screenshot01.png")
set(EXTENSION_DEPENDS "PyTorch NNUNet")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SlimFormerPlusPlus)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
