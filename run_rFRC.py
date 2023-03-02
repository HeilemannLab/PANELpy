"""
Apply rFRC to matching image pairs in two directories. Images within directories must have matching names.
"""

# TODO save global info as file
# TODO wiki px size in nm
# TODO load images from two directories
# TODO work with png images
# TODO work with 16 bit images (background value?, does rfrc map look reasonable)
# TODO outputs, PANEL map with confocal img?
# TODO speed up calculation by parallelization


import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PANEL import PANEL
from fSNR.fSNRmap import fSNRmap
from Utils.shifted_jet import sjet_colorbar
from PIL import Image
import os

# define path to directory A
DIR_A = r"D:\Chemie_phd\_data\STED_denoising\rFRC_testdata\high_small2_png_8bit"
# define path to directory B
DIR_B = r"D:\Chemie_phd\_data\STED_denoising\rFRC_testdata\highpair_small2_png_8bit"
# image format, only files with that type are red
IMG_FORMAT = "png"
# define save directory, folder is automatically created
SAVE_DIR = r"D:\Chemie_phd\_data\STED_denoising\rFRC_testdata\test2"
# background value in rFRC
BACKGROUND = 15
# px size in nm
PIXEL_SIZE = 15
# px to skip
SKIP = 1


def load_imgs(dir_path, format="tif"):
    """Read all images from a directory as np.arrays."""
    imgs = []
    file_names = []
    for file in os.listdir(dir_path):
        if file.endswith(format):
            imgs.append(np.array(Image.open(dir_path + "\\" + file)))
            file_names.append(file)
    return imgs, file_names


def make_stacks(imgs_A, imgs_B):
    """Create stacks of matching image pairs."""
    imgs_stacks = []
    for img_A, img_B in zip(imgs_A, imgs_B):
        img_stack = np.stack((img_A, img_B), axis=0)
        imgs_stacks.append(img_stack)
    return imgs_stacks


def run_rFRC(stack, file_name):
    """Apply rFRC to a stack and save results."""
    [rFRC_map, PANEL_map, global_vals] = PANEL(stack, pixelsize=PIXEL_SIZE, skip=SKIP, backgroundIntensity=BACKGROUND, EnableOtsu=False)
    # save values as txt
    header = "min resolution / nm\tmax resolution / nm\tmean resolution / nm\trFRC value\n"
    with open(SAVE_DIR + "\\" + os.path.splitext(file_name)[0] + ".txt", "w") as f:
        f.write(header)
        row = "\t".join(str(i) for i in global_vals)
        f.write(row)
    # save rFRC map
    tifffile.imsave(SAVE_DIR + "\\" + os.path.splitext(file_name)[0] + "_rFRC_map.tif", rFRC_map)
    # save PANEL map
    tifffile.imsave(SAVE_DIR + "\\" + os.path.splitext(file_name)[0] + "_PANEL.tif", PANEL_map)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    imgs_A, file_names_A = load_imgs(DIR_A, IMG_FORMAT)
    imgs_B, file_names_B = load_imgs(DIR_B, IMG_FORMAT)
    imgs_stacks = make_stacks(imgs_A, imgs_B)
    for img_stack, file_name in zip(imgs_stacks, file_names_A):
        run_rFRC(img_stack, file_name)


def main_single():
    file = r'D:\Chemie_phd\_data\STED_denoising\rFRC_testdata\HDSMLM_20nmpixel_background_15_small.tif'
    stack = tifffile.imread(file)
    [map_bg_15, PANELs_bg_15, absolute_value_bg_15] = PANEL(stack, pixelsize=20, skip=1, backgroundIntensity=15, EnableOtsu=False)
    tifffile.imsave(r'D:\Chemie_phd\_data\STED_denoising\rFRC_testdata\map_bg_15.tif', map_bg_15)


if __name__ == "__main__":
    main()
    # main_single()
