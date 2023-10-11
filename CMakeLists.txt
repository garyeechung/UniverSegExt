cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(UniverSegExt)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/UniverSegExt")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Gary Chung (Vanderbilt University)")
set(EXTENSION_DESCRIPTION "Extension based on Butoi et al.  2023")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/UniverSegExt.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/UniverSegExt/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(UniverSegEditor)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
