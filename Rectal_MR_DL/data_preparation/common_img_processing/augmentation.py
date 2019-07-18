# ------------------------------------------------------------------------------
# imports
import SimpleITK as sitk
import numpy as np
import random
from itertools import count, takewhile

import matplotlib.pyplot as plt

# medinfer inner imports
from .resampling import resample_sitkImage_by_reference as augmentation_resample_ref
from .resampling import resample_sitkImageLabel_byTransform as augmentation_transform
from .resampling import resample_sitkImage_by_spacing as augmentation_resample_spacing
from .cropping import crop_to_max_bounding_box as augmentation_crop_to_max_bounding_box
from .cropping import crop_mask_region as augmentation_crop_mask_region
from .cropping import crop_full_mask_region_with_largest_mask_area_slice
from .labels import create_sliced_label_center_aligned_no_slice_axis_spare

# ----------------------------------------------------------------------------------
# get logger
from utils.logger import get_logger

logger = get_logger()

# ------------------------------------------------------------------------------------------------
#
def get_sitkimage_physicalcenter(sitkImage):
    if sitkImage is None: return None
    dim = sitkImage.GetDimension()
    size = np.array(sitkImage.GetSize()).reshape(dim, 1)
    spacing = np.array(sitkImage.GetSpacing()).reshape(dim, 1)
    origin = np.array(sitkImage.GetOrigin()).reshape(dim, 1)
    direction = np.array(sitkImage.GetDirection()).reshape(dim, dim)
    shift = np.dot(direction, spacing * size * 0.5)
    center = origin + shift
    return tuple(center[:, 0])


# ------------------------------------------------------------------------------------------------
#
def sitk_scale(vol, label, scale=1.0, filled_vol_value='min', **kwargs):
    if vol is None: return None
    dim = vol.GetDimension()
    scale_dim = tuple([scale] * dim)
    transform = sitk.ScaleTransform(dim)
    transform.SetScale(scale_dim)
    transform.SetCenter(get_sitkimage_physicalcenter(vol))
    additional_label_list = kwargs.get('additional_label_list')
    if additional_label_list is None:
        return augmentation_transform(vol, label, transform, filled_vol_value)
    else:
        return augmentation_transform(vol, label, transform, filled_vol_value,
                                      additional_label_list=additional_label_list)


# ------------------------------------------------------------------------------------------------
#
def sitk_rotate(vol, label, axis=(0, 0, 1), degree=0.0, filled_vol_value='min', **kwargs):
    radian = np.pi * degree / 180
    rotate_transform = sitk.VersorTransform(axis, radian)
    rotate_transform.SetCenter(get_sitkimage_physicalcenter(vol))
    additional_label_list = kwargs.get('additional_label_list')
    if additional_label_list is None:
        return augmentation_transform(vol, label, rotate_transform, filled_vol_value)
    else:
        return augmentation_transform(vol, label, rotate_transform, filled_vol_value,
                                      additional_label_list=additional_label_list)


# ------------------------------------------------------------------------------------------------
#
def sitk_translation(vol, label, offset=(0.0, 0.0, 0.0), filled_vol_value='min', **kwargs):
    if vol is None: return None
    dim = vol.GetDimension()
    if len(offset) != dim: return None
    transform = sitk.TranslationTransform(dim)
    transform.SetOffset(offset)
    additional_label_list = kwargs.get('additional_label_list')
    if additional_label_list is None:
        return augmentation_transform(vol, label, transform, filled_vol_value)
    else:
        return augmentation_transform(vol, label, transform, filled_vol_value,
                                      additional_label_list=additional_label_list)


# ------------------------------------------------------------------------------------------------
#
def sitk_slicethickness(vol, label, slicethickness, **kwargs):
    if vol is None: return None
    dim = vol.GetDimension()
    old_spacing = vol.GetSpacing()
    new_spacing = (old_spacing[0], old_spacing[1], slicethickness)
    new_vol = augmentation_resample_spacing(vol, new_spacing)
    new_label = augmentation_resample_spacing(label, new_spacing)
    additional_label_list = kwargs.get('additional_label_list')
    if additional_label_list is None:
        return new_vol, new_label
    else:
        new_additional_label_list = []
        for additional_label in additional_label_list:
            new_additional_label_list.append(augmentation_resample_spacing(additional_label, new_spacing))
        return new_vol, new_label, new_additional_label_list

# ------------------------------------------------------------------------------------------------
# total vols = scales * (sum(rotations_per_axis) + 1)
def sitk_augment_by_rotate_roi_3d(sitk_image_list, sitk_label,
                                  slice_axis=(0, 0, 1), nb_rotations=20):
    """
    :param sitk_image_list:
    :param sitk_label:
    :param slice_axis:
    :param nb_rotations:
    :return:
    """
    # check inputs - class
    if sitk_image_list.__class__ != list:
        logger.error('%s is not list.', sitk_image_list.__class__)
        return []
    checks = [sitk_image.__class__ == sitk.Image for sitk_image in sitk_image_list]
    if False in checks:
        logger.error('inside sitk_image_list are not all the sitk_image type.')
        return []
    if sitk_label.__class__ != sitk.Image:
        logger.error('sitk_label is not the sitk_label type.')
        return []

    # check input - dim
    dim = sitk_label.GetDimension()
    if dim != 3:
        logger.info('not volume image.')
        return []
    checks = [dim == sitk_image.GetDimension() for sitk_image in sitk_image_list]
    if False in checks:
        logger.error('sitk_image dimension error')
        return []
    if dim != len(slice_axis):
        logger.error('slice_axis dimension error')
        return []

    # check input - slice_axis
    if slice_axis not in [(0, 0, 1), (0, 1, 0), (1, 0, 0)]:
        logger.error('slice_axis coding error')
        return []

    # crop the label
    sitk_label = crop_full_mask_region_with_largest_mask_area_slice(sitk_label, sitk_label)
    np_imgssitk_label = sitk.GetArrayFromImage(sitk_label)

    # to remove the effect of crop mask
    np_spacing = np.array(sitk_label.GetSpacing())
    np_size = np.array(sitk_label.GetSize())
    np_range = np_size * np_spacing
    np_slice_axis = np.array(slice_axis)
    np_range[np_slice_axis == 1] = 0  # no need for spare range of slice axis
    np_range_mm_max = np.max(np_range)

    # generate translation
    # sts: sliced, translated, scaled
    sitk_label_sts = []

    np_nb_slices = np_size[np_slice_axis == 1]
    trange_mm = np_range_mm_max * 0.5

    sitk_label_sts.append(create_sliced_label_center_aligned_no_slice_axis_spare(sitk_label,
                                                                                 nb_slices=tuple(np_nb_slices.tolist()),
                                                                                 translation_mm=(0, 0, 0),
                                                                                 spare_boundary_mm=0.02 * trange_mm))
    # to generate sitk_label with 2 times of bounding box size
    np_nb_slices_d = np_nb_slices * 1
    sitk_label_sliced = create_sliced_label_center_aligned_no_slice_axis_spare(sitk_label,
                                                                               nb_slices=tuple(np_nb_slices_d.tolist()),
                                                                               translation_mm=(0.0, 0.0, 0.0),
                                                                               spare_boundary_mm=2 * trange_mm)
    sitk_images_bb, sitk_labels_bb = augmentation_crop_to_max_bounding_box(sitk_image_list,
                                                                           [sitk_label_sliced],
                                                                           spare_boundary_mm=0.0,
                                                                           crop_axis=(1, 1, 1))
    # to generate rotate and scale sitk_image
    sitk_images_rois = []
    sitk_labels_rois = []
    sitk_label_bb = sitk_labels_bb[0]

    # to generate rotated and scaled rois and labels
    # the default none rotated is added into the first
    degrees = [0.0]
    for n in range(0, nb_rotations):
        degree = 360 / (nb_rotations + 1) * (n + 1)
        degrees.append(degree)

    for sitk_label_sts_i in sitk_label_sts:
        for degree in degrees:
            # print('specific degree:')
            # print(degree)
            sitk_image_rois = []
            sitk_label_rois = []
            for sitk_image_bb in sitk_images_bb:
                direction0 = sitk_image_bb.GetDirection()
                slice_axis0 = (direction0[2], direction0[5], direction0[8])
                vol0, label0 = sitk_rotate(sitk_image_bb, sitk_label_bb, axis=slice_axis0, degree=degree)
                sitk_label_roi_vol = augmentation_resample_ref(sitk_label_bb, sitk_label_sts_i)
                vol1 = augmentation_crop_mask_region(vol0, sitk_label_roi_vol)
                label1 = augmentation_crop_mask_region(label0, sitk_label_roi_vol)
                sitk_image_rois.append(vol1)
                sitk_label_rois.append(label1)
            sitk_images_rois.append(sitk_image_rois)
            sitk_labels_rois.append(sitk_label_rois)

    return sitk_images_rois, sitk_labels_rois
