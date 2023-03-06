# -*- coding: utf-8 -*-
import numpy as np
import math
import concurrent.futures
from fSNR.FRCAnalysis import FRCAnalysis
from Utils.AMF import AMF


def calc_frc(items):
    '''
    Calculate resolution per window.
    '''
    xstart, ystart, blockSize, skip, pixelsize, correctDrift, image1_pad, image2_pad, BI = items
    image1_ROI = np.zeros((blockSize, blockSize), dtype='float32')
    image1_ROI[:, :] = image1_pad[xstart:xstart + blockSize, ystart:ystart + blockSize]
    image2_ROI = np.zeros((blockSize, blockSize), dtype='float32')
    image2_ROI[:, :] = image2_pad[xstart:xstart + blockSize, ystart:ystart + blockSize]
    suming = 0
    flage = 0
    for xsum in range(skip):
        for ysum in range(skip):
            xstart_center = np.array((blockSize / 2 - 1), dtype='int')
            image1_ROI_center = np.zeros((1, 1), dtype='float32')
            image1_ROI_center[:, :] = image1_ROI[xstart_center + xsum:xstart_center + xsum + 1,
                                      xstart_center + ysum:xstart_center + ysum + 1]
            image2_ROI_center = np.zeros((1, 1), dtype='float32')
            image2_ROI_center[:, :] = image2_ROI[xstart_center + xsum:xstart_center + xsum + 1,
                                      xstart_center + ysum:xstart_center + ysum + 1]
            value1 = image1_ROI_center[:, :]
            value2 = image2_ROI_center[:, :]
            suming = suming + value1 + value2
            flage = flage + 1
    suming = suming / flage;
    if ((suming) > (BI)):
        image1_ROI = image1_ROI / np.max(image1_ROI)
        image2_ROI = image2_ROI / np.max(image2_ROI)
        __, largeAngles, threeSigma, fiveSigma, twoSigma, __, res = FRCAnalysis(image1_ROI, image2_ROI, pixelsize,
                                                                                correctDrift)
        [fixed_resolution, threesigma_resolution, fivesigma_resolution, twosigma_resolution] = res[:, 0]
        if not math.isnan(threesigma_resolution) and not math.isinf(threesigma_resolution) and not (
                threesigma_resolution < 0):
            resolution = threesigma_resolution * (3 / 3) * (3 / 3)
        elif not math.isnan(fivesigma_resolution) and not math.isinf(fivesigma_resolution) and not (
                fivesigma_resolution < 0):
            resolution = fivesigma_resolution * (3 / 5) * (3 / 5)
        elif not math.isnan(twosigma_resolution) and not math.isinf(twosigma_resolution) and not (
                twosigma_resolution < 0):
            resolution = twosigma_resolution * (3 / 2) * (3 / 2)
        # elif not math.isnan(fixed_resolution) and not math.isinf(fixed_resolution)and not(fixed_resolution < 0):
        #     resolution = fixed_resolution
        if resolution == None:
            resolution = 0
        resolutions = [resolution]
        xy_coordinates = [(xstart, ystart)]
        for i in range(xstart, xstart + skip):
            for j in range(ystart, ystart + skip):
                resolutions.append(resolution)
                xy_coordinates.append((i,j))
        print("inside", resolutions, xy_coordinates)
        return [resolutions, xy_coordinates]


def fSNRmap(stack, pixelsize = 30.25, backgroundIntensity = 5, skip = 1, blockSize = 64, \
            correctDrift = False, amedianfilter = True):
    '''
    Caculate the rFRCmap for quantitatively mapping the local image quality

    Parameters
    ----------
    stack : input two images to be evaluated, ndarray, shape (2, M, N).
    pixelsize : pixel size in nanometer {default: 30.25}
    backgroundIntensity :  background intensity (0~255 range, 8bit) {default: 5}
    skip : skip size to accelerate the rFRC calculation {default: 1}
    blockSize : rFRC block size {default: 64}
    driftCorrection : if do drift correction {default: false}
    amedianfilter : whether do adaptive filter after rFRC mapping {default: true}

    Returns
    -------
    rFRC_map 

    '''
    image1 = stack[0]
    image2 = stack[1]
    if (blockSize % 2 != 0):
        blockSize = blockSize + 1
    tem = (np.array(blockSize / 2)).astype('int')
    skip = skip if skip < tem else tem
    
    [w, h] = image1.shape 

    padw = w + blockSize
    padh = h + blockSize
    
    image1_pad = createPaddedImage(image1, padw, padh)
    image2_pad = createPaddedImage(image2, padw, padh)
    
    BI = (np.array(backgroundIntensity*2/255)).astype('float32')

    xstarts = list(range(0, w, skip))
    ystarts = list(range(0, h, skip))
    items = [(xstart, ystart, blockSize, skip, pixelsize, correctDrift, image1_pad, image2_pad, BI)
             for xstart in xstarts for ystart in ystarts]

    # collect resolution per window and window xy coordinates
    windows_resolution, windows_xy = [], []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for item, rfrc_calculation in zip(items, executor.map(calc_frc, items)):
            print("outside", rfrc_calculation)
            resolutions, xy_coordinates = rfrc_calculation
            windows_resolution.extend(resolutions)
            windows_xy.extend(xy_coordinates)

    # # fill with skip values
    # windows_resolution_filled = windows_resolution
    # windows_xy_filled = windows_xy
    # if skip > 0:
    #     for resolution, windows_xy in zip(windows_resolution, windows_xy):
    #         for i in range(windows_xy[0]+skip):
    #             for j in range(windows_xy[1]+skip):
    #                 windows_resolution_filled.append(resolution)
    #                 windows_xy_filled.append((i, j))

    # convert rfrc window resolutions to a 2D numpy array
    # rFRC_map = np.zeros((w, h), dtype='float32')
    # rFRC_map[tuple(zip(*windows_xy_filled))] = windows_resolution_filled
    rFRC_map = np.reshape(windows_resolution, (w, h))
    # rFRC_map[np.isnan(rFRC_map)] = 0.
    rFRC_map[rFRC_map == None] = 0.

    if amedianfilter:
        rFRC_map = AMF(rFRC_map)
    return rFRC_map


def createPaddedImage(image1, paddedWidth, paddedHeight):
    imageWidth = image1.shape[0]
    imageHeight = image1.shape[0]
    extraWidth = paddedWidth - imageWidth
    extraHeight = paddedHeight - imageHeight
    if image1.dtype == 'uint8':
        image1 = image1.astype('float32') / 255
    image_tem = np.array(image1)
    extracol = np.array((extraWidth/2), dtype = 'int')
    extrarow = np.array((extraHeight/2), dtype = 'int')
    paddedArray = np.pad(image_tem, (extracol, extrarow), 'symmetric')
    return paddedArray
