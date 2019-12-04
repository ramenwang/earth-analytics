#########################################
# Handcrafted tools for image processing
# Auther: Qing Wang
# Date:   12/03/2019
#########################################

import rasterio as rio
import os
import math
from rasterio.warp import reproject, Resampling, calculate_default_transform
import matplotlib.pyplot as plt
import earthpy.plot as ep


def project_stack(img_stk, out_file, out_crs):

    """
    A function to reproject image stack

    :param img_stk (object): rasterio object created from rasterio.open()
    :param out_file (str): path of the ouput - reprojected image stack
    :param out_crs (str): code for target spatial reference (e.g. EPSG:32611),
                         one can check 'https://spatialreference.org/ref/epsg/'
    :return 0 (int): success
    """

    # define the parameters for reprojection
    transform, width, height = calculate_default_transform(img_stk.crs, \
        out_crs, img_stk.width, img_stk.height, *img_stk.bounds)

    # update meta data
    out_meta = img_stk.meta.copy()
    out_meta.update({
            'crs':out_crs,
            'transform':transform,
            'width':width,
            'height':height
    })

    # reprojection loops over image stacks, and write into path
    with rio.open(out_file, 'w', **out_meta) as out_img:
        for i in range(1, img_stk.count+1):
            reproject(
                source=rio.band(img_stk, i),
                destination=rio.band(out_img, i),
                src_transform = img_stk.transform,
                src_crs=img_stk.crs,
                dst_transform=transform,
                dst_crs=out_crs,
                resampling=Resampling.cubic
            )

    return 0


def maximum_intersect_transform(img1, img2, img_res):

    """
    A function to find the maximum available boundary on the intersect area between
    two images based on a given image spatial resolution.

    :param img1, img2 (object): rasterio object created from rasterio.open(),
                                which are the inputs forming the intersect image
    :param img_res (float): target image spatial resolution (square pixel size)
    :return (3 objects): a transform Affine object used in rasterio,
                         image (matrix) width, image (matrix) height
    """

    bounds1 = img1.bounds
    bounds2 = img2.bounds

    xmin = round(max(bounds1[0], bounds2[0]))
    ymin = round(max(bounds1[1], bounds2[1]))
    xmax = round(min(bounds1[2], bounds2[2]))
    ymax = round(min(bounds1[3], bounds2[3]))

    width = math.floor((xmax-xmin) / img_res)
    height = math.floor((ymax-ymin) / img_res)
    print("width is: ", (xmax-xmin))
    east = xmin + width*img_res
    north = ymin + height*img_res

    # create shapefile
    return rio.transform.from_bounds(west=xmin, south=ymin, east=east,\
                                     north=north, width=width, height=height), \
                                     width, height


def resample_pixel_size(img_stk, out_file, target_transform, target_width, target_height):

    """
    A function to resample the pixel size of an image stack based on target affine
    transformation object. The method is bi-cubic resampling.

    :param img_stk (object): rasterio object created from rasterio.open()
    :param out_file (str): path of the ouput - resampled image stack
    :param target_transform (object): affine transformation object (rasterio)
    :param target_width: target image width (matrix column number)
    :param target_height: traget image height (matrix row number)
    :return 0 (int): seccuss
    """
    
    out_meta = img_stk.meta.copy()
    out_meta.update({
        'transform':target_transform,
        'width':target_width,
        'height':target_height
    })

    with rio.open(out_file, 'w', **out_meta) as out_img:
        for i in range(1, img_stk.count+1):
            reproject(
                source=rio.band(img_stk, i),
                destination=rio.band(out_img, i),
                src_transform = img_stk.transform,
                src_crs=img_stk.crs,
                dst_transform=target_transform,
                dst_crs=img_stk.crs,
                resampling=Resampling.cubic
            )

    return 0
