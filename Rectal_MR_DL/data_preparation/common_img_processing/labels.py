# ------------------------------------------------------------------------------
# imports
import SimpleITK as sitk
import numpy as np

from skimage.morphology import watershed
from skimage.segmentation import random_walker
from scipy import cluster

from .cropping import crop_mask_region as labels_crop_mask_region


# ------------------------------------------------------------------------------------------------
#
def get_label_physicalcenter(sitkLabel):
    if sitkLabel is None: return None
    sitkLabel = labels_crop_mask_region(sitkLabel, sitkLabel)
    dim = sitkLabel.GetDimension()
    size = np.array(sitkLabel.GetSize()).reshape(dim, 1)
    spacing = np.array(sitkLabel.GetSpacing()).reshape(dim, 1)
    origin = np.array(sitkLabel.GetOrigin()).reshape(dim, 1)
    direction = np.array(sitkLabel.GetDirection()).reshape(dim, dim)
    shift = np.dot(direction, spacing * size * 0.5)
    center = origin + shift
    return tuple(center[:, 0])


# ------------------------------------------------------------------------------
# merger all foreground labels naming from 1 - n into 1
def merge_foreground_labels(sitkLabel):
    """
    :param sitkLabel:
    :return:
    """
    if sitkLabel is None: return None
    if sitkLabel.GetPixelID() != sitk.sitkUInt8: sitkLabel = sitk.Cast(sitkLabel, sitk.sitkUInt8)

    img_array = sitk.GetArrayFromImage(sitkLabel)
    new_img_array = np.zeros(img_array.shape, dtype=np.uint8)
    new_img_array[img_array > 0] = 1
    origin = sitkLabel.GetOrigin()
    spacing = sitkLabel.GetSpacing()
    direction = sitkLabel.GetDirection()
    new_sitkMask = sitk.GetImageFromArray(new_img_array)
    new_sitkMask.SetOrigin(origin)
    new_sitkMask.SetSpacing(spacing)
    new_sitkMask.SetDirection(direction)

    return new_sitkMask


# ------------------------------------------------------------------------------
# get selected labels
def get_selected_labels(sitkLabel, selected_label_list, mode='retrieve'):
    """
    :param sitkLabels:
    :param selected_labels_list:
    :param mode: 'merge', 'retrieve'
    :return:
    """
    if sitkLabel is None: return None
    if len(selected_label_list) < 1: return sitkLabel
    if sitkLabel.GetPixelID() != sitk.sitkUInt8: sitkLabel = sitk.Cast(sitkLabel, sitk.sitkUInt8)

    img_array = sitk.GetArrayFromImage(sitkLabel)
    new_img_array = np.zeros(img_array.shape, dtype=np.uint8)
    for i in range(len(selected_label_list)):
        if mode == 'retrieve':
            new_img_array[img_array == selected_label_list[i]] = selected_label_list[i]
        elif mode == 'merge':
            new_img_array[img_array == selected_label_list[i]] = 1
        else:
            new_img_array = img_array
    origin = sitkLabel.GetOrigin()
    spacing = sitkLabel.GetSpacing()
    direction = sitkLabel.GetDirection()
    new_sitkMask = sitk.GetImageFromArray(new_img_array)
    new_sitkMask.SetOrigin(origin)
    new_sitkMask.SetSpacing(spacing)
    new_sitkMask.SetDirection(direction)

    return new_sitkMask


# ------------------------------------------------------------------------------
# get largest connected label
def get_largest_connected_labels(sitkLabel, **kwargs):
    """
    :param sitkLabel:
    :param kwargs:
    :return:
    """
    threshold = kwargs.get('threshold')
    dilate_radius = kwargs.get('dilate_radius')

    if sitkLabel is None: return None
    # if sitkLabel.GetPixelID() != sitk.sitkUInt8: sitkLabel = sitk.Cast(sitkLabel, sitk.sitkUInt8)
    if threshold is not None:
        if len(threshold) == 2:
            sitkLabel = sitk.BinaryThreshold(sitkLabel,
                                             lowerThreshold=threshold[0],
                                             upperThreshold=threshold[1])
    if dilate_radius is not None: sitkLabel = sitk.BinaryDilate(sitkLabel,
                                                                dilate_radius)

    sitkLabel = sitk.ConnectedComponent(sitkLabel)
    sitkLabel = sitk.RelabelComponent(sitkLabel)

    return get_selected_labels(sitkLabel, selected_label_list=[1], mode='retrieve')


# ------------------------------------------------------------------------------
# get foreground labels list
def get_foreground_labels_list(sitkLabel, **kwargs):
    """
    :param sitkLabel:
    :param kwargs:
    :return:
    """
    threshold = kwargs.get('threshold')
    dilate_radius = kwargs.get('dilate_radius')
    reset_label_id = kwargs.get('reset_label_id')

    if sitkLabel is None: return None
    if sitkLabel.GetPixelID() != sitk.sitkUInt8: sitkLabel = sitk.Cast(sitkLabel, sitk.sitkUInt8)
    if threshold is not None:
        if len(threshold) == 2:
            sitkLabel = sitk.BinaryThreshold(sitkLabel,
                                             lowerThreshold=threshold[0], upperThreshold=threshold[1])
    if dilate_radius is not None: sitkLabel = sitk.BinaryDilate(sitkLabel, dilate_radius)
    if reset_label_id is None: reset_label_id = False

    sitk_labels_list = []

    origin = sitkLabel.GetOrigin()
    spacing = sitkLabel.GetSpacing()
    direction = sitkLabel.GetDirection()
    img_array = sitk.GetArrayFromImage(sitkLabel)
    labels_id_list = list(np.unique(img_array.flatten()))
    if 0 in labels_id_list: labels_id_list.remove(0)

    for label_id in labels_id_list:
        new_img_array = np.zeros(img_array.shape, dtype=np.uint8)
        if reset_label_id is False:
            new_img_array[img_array == label_id] = label_id
        else:
            new_img_array[img_array == label_id] = 1
        new_sitkLabel = sitk.GetImageFromArray(new_img_array)
        new_sitkLabel.SetOrigin(origin)
        new_sitkLabel.SetSpacing(spacing)
        new_sitkLabel.SetDirection(direction)
        sitk_labels_list.append(new_sitkLabel)

    return sitk_labels_list


# ------------------------------------------------------------------------------
# get foreground labels list
def get_foreground_labels_list_for_bjzl(sitkLabel, **kwargs):
    """
    :param sitkLabel:
    :param kwargs:
    :return:
    """
    reset_label_id = kwargs.get('reset_label_id')

    if sitkLabel is None: return None
    if sitkLabel.GetPixelID() != sitk.sitkUInt8: sitkLabel = sitk.Cast(sitkLabel, sitk.sitkUInt8)

    if reset_label_id is None: reset_label_id = False

    sitk_labels_list = []

    origin = sitkLabel.GetOrigin()
    spacing = sitkLabel.GetSpacing()
    direction = sitkLabel.GetDirection()
    img_array = sitk.GetArrayFromImage(sitkLabel)
    labels_id_list = list(np.unique(img_array.flatten()))
    if 0 in labels_id_list: labels_id_list.remove(0)

    for label_id in labels_id_list:
        new_img_array = np.zeros(img_array.shape, dtype=np.uint8)
        if reset_label_id is False:
            new_img_array[img_array == label_id] = label_id
        else:
            new_img_array[img_array == label_id] = 1
        new_sitkLabel = sitk.GetImageFromArray(new_img_array)
        new_sitkLabel.SetOrigin(origin)
        new_sitkLabel.SetSpacing(spacing)
        new_sitkLabel.SetDirection(direction)
        sitk_labels_list.append(new_sitkLabel)

    return sitk_labels_list


# ------------------------------------------------------------------------------
# get foreground labels id list
def get_foreground_labels_id_list(sitkLabel):
    """
    :param sitkLabel:
    :return:
    """
    if sitkLabel is None: return None
    if sitkLabel.GetPixelID() != sitk.sitkUInt8: sitkLabel = sitk.Cast(sitkLabel, sitk.sitkUInt8)

    img_array = sitk.GetArrayFromImage(sitkLabel)
    labels_id_list = list(np.unique(img_array.flatten()))
    if 0 in labels_id_list: labels_id_list.remove(0)

    return labels_id_list


# ------------------------------------------------------------------------------
# renaming the foreground labels label
def renaming_foreground_labels(sitkLabel, **kwargs):
    """
    :param sitkLabel:
    :param kwargs:
    :return:
    """
    foreground_label_id_list = kwargs.get('foreground_label_id_list')

    if sitkLabel is None: return None
    if sitkLabel.GetPixelID() != sitk.sitkUInt8: sitkLabel = sitk.Cast(sitkLabel, sitk.sitkUInt8)

    img_array = sitk.GetArrayFromImage(sitkLabel)
    new_image_array = np.zeros(img_array.shape, dtype=np.uint8)
    if foreground_label_id_list is not None:
        for foreground_label_id in foreground_label_id_list:
            new_image_array[img_array == foreground_label_id] = foreground_label_id
    else:
        new_image_array = img_array

    new_sitkLabel = sitk.GetImageFromArray(new_image_array)
    new_sitkLabel.SetOrigin(sitkLabel.GetOrigin())
    new_sitkLabel.SetSpacing(sitkLabel.GetSpacing())
    new_sitkLabel.SetDirection(sitkLabel.GetDirection())
    new_sitkLabel = sitk.RelabelComponent(new_sitkLabel)

    return new_sitkLabel


# -------------------------------------------------------------------------------
#
def create_cubic_label_center_aligned(sitkLabel, spare_boundary_mm=10.0):
    """
    :param size:
    :param range:
    :return:
    """
    if sitkLabel.__class__ != sitk.Image: return None
    dim = sitkLabel.GetDimension()

    np_center = np.array(get_label_physicalcenter(sitkLabel))
    np_spacing = np.array(sitkLabel.GetSpacing())
    np_size = np.array(sitkLabel.GetSize())
    np_range = np_size * np_spacing + np.array([spare_boundary_mm] * dim)
    np_range = np.array([max(np_range)] * dim)
    np_size = np_range / np_spacing
    size = tuple(np_size.astype(np.int).tolist())
    size = size[::-1]
    np_image_array = np.ones(size, np.uint8)
    new_sitkLabel = sitk.GetImageFromArray(np_image_array)
    new_sitkLabel.SetDirection(sitkLabel.GetDirection())
    new_sitkLabel.SetSpacing(sitkLabel.GetSpacing())

    # calculate and set origin
    np_matrix_direction = np.array(sitkLabel.GetDirection()).reshape(dim, dim)
    np_calibrated_shift = np.dot(np_matrix_direction, np_spacing * np_size / 2)
    np_origin = np_center - np_calibrated_shift
    origin = tuple(np_origin.tolist())
    new_sitkLabel.SetOrigin(origin)

    return new_sitkLabel


# -------------------------------------------------------------------------------
#
def create_sliced_label_center_aligned(sitkLabel, nb_slices=(0, 0, 3),
                                       translation_mm=(0.0, 1.0, 5.0), spare_boundary_mm=0.0):
    """
    :param sitkLabel:
    :param nb_slices: 0 means full, >0 means number of slices
    :param translation_mm:
    :param spare_boundary_mm: only applied for nb_slice = 0 for nb_slice in nb_slices
    :return:
    """
    if sitkLabel.__class__ != sitk.Image: return None
    dim = sitkLabel.GetDimension()
    if dim != len(nb_slices): return None
    if dim != len(translation_mm): return None

    np_center = np.array(get_label_physicalcenter(sitkLabel))
    np_spacing = np.array(sitkLabel.GetSpacing())
    np_size = np.array(sitkLabel.GetSize())
    np_size_sliced = np.array(nb_slices)
    idx = np_size != np_size_sliced + np_size

    np_range = np_size * np_spacing + np.array([spare_boundary_mm] * dim)
    np_range[idx] = np_size_sliced[idx] * np_spacing[idx]
    np_range = np.array([max(np_range)] * dim)
    np_range[idx] = np_size_sliced[idx] * np_spacing[idx]
    np_size = np_range / np_spacing
    np_size[idx] = np_size_sliced[idx]
    size = tuple(np_size.astype(np.int).tolist())
    size = size[::-1]
    # print(size)

    np_image_array = np.ones(size, np.uint8)
    new_sitkLabel = sitk.GetImageFromArray(np_image_array)
    new_sitkLabel.SetDirection(sitkLabel.GetDirection())
    new_sitkLabel.SetSpacing(sitkLabel.GetSpacing())

    # calculate and set origin
    np_matrix_direction = np.array(sitkLabel.GetDirection()).reshape(dim, dim)
    np_calibrated_translation = np.dot(np_matrix_direction, np_spacing * np_size / 2)
    np_origin_translation = np.dot(np_matrix_direction, translation_mm)
    np_origin = np_center + np_origin_translation - np_calibrated_translation
    origin = tuple(np_origin.tolist())
    new_sitkLabel.SetOrigin(origin)

    return new_sitkLabel


# -------------------------------------------------------------------------------
#
def create_sliced_label_center_aligned_no_slice_axis_spare(sitkLabel, nb_slices=(0, 0, 3),
                                                           translation_mm=(0.0, 1.0, 5.0), spare_boundary_mm=0.0,
                                                           slice_axis=(0, 0, 1)):
    """
    :param sitkLabel:
    :param nb_slices: 0 means full, >0 means number of slices
    :param translation_mm:
    :param spare_boundary_mm: only applied for nb_slice = 0 for nb_slice in nb_slices
    :return:
    """
    if sitkLabel.__class__ != sitk.Image: return None
    dim = sitkLabel.GetDimension()
    if dim != len(translation_mm): return None

    np_center = np.array(get_label_physicalcenter(sitkLabel))
    np_spacing = np.array(sitkLabel.GetSpacing())
    np_size = np.array(sitkLabel.GetSize())

    np_slice_axis = np.array(slice_axis)
    np_range_spare = np.array([spare_boundary_mm] * dim)
    np_range_spare[np_slice_axis == 1] = 0  # no need for spare range of slice axis
    np_range = np_size * np_spacing + np_range_spare
    np_range_slice_axis = np_range[np_slice_axis == 1]

    np_range_non_slice_axis = np_range
    np_range_non_slice_axis[np_slice_axis == 1] = 0  # no need for spare range of slice axis

    np_range = np.array([max(np_range_non_slice_axis)] * dim)
    np_range[np_slice_axis == 1] = np_range_slice_axis
    np_size = np_range / np_spacing
    size = tuple(np_size.astype(np.int).tolist())
    size = size[::-1]
    # print(size)

    np_image_array = np.ones(size, np.uint8)
    new_sitkLabel = sitk.GetImageFromArray(np_image_array)
    new_sitkLabel.SetDirection(sitkLabel.GetDirection())
    new_sitkLabel.SetSpacing(sitkLabel.GetSpacing())

    # calculate and set origin
    np_matrix_direction = np.array(sitkLabel.GetDirection()).reshape(dim, dim)
    np_calibrated_translation = np.dot(np_matrix_direction, np_spacing * np_size / 2)
    np_origin_translation = np.dot(np_matrix_direction, translation_mm)
    np_origin = np_center + np_origin_translation - np_calibrated_translation
    origin = tuple(np_origin.tolist())
    new_sitkLabel.SetOrigin(origin)

    return new_sitkLabel


# ------------------------------------------------------------------------------
# renaming the foreground labels label
def measuring_foreground_labels(sitkLabel, **kwargs):
    """
    :param sitkLabel:
    :param kwargs:
    :return:
    """
    foreground_label_id_list = kwargs.get('foreground_label_id_list')

    results = {}
    new_sitkLabel = renaming_foreground_labels(sitkLabel, foreground_label_id_list=foreground_label_id_list)
    if new_sitkLabel is None: return results

    dim = sitkLabel.GetDimension()
    origin = sitkLabel.GetOrigin()
    spacing = sitkLabel.GetSpacing()
    direction = sitkLabel.GetDirection()

    np_origin = np.array(origin, np.float)
    np_spacing = np.array(spacing, np.float)
    np_direction = np.array(direction, np.float).reshape(dim, dim)

    lsf = sitk.LabelStatisticsImageFilter()
    lsf.Execute(new_sitkLabel, new_sitkLabel)
    labels = lsf.GetLabels()
    for label in labels:
        boundingbox_idx = lsf.GetBoundingBox(label)
        np_boundingbox_idx = np.array(boundingbox_idx, dtype=np.float)
        np_boundingbox_idx = np_boundingbox_idx.reshape(dim, 2)
        np_boundingbox_center_idx = np_boundingbox_idx[:, 0] + 0.5 * (
                    np_boundingbox_idx[:, 1] - np_boundingbox_idx[:, 0])
        np_boundingbox_center_pos = np.dot(np_direction, np_spacing * np_boundingbox_center_idx) + np_origin

        results[label] = []
        results[label].append({'Label ID:': label})
        results[label].append({'Bounding Box Index:': lsf.GetBoundingBox(label)})
        results[label].append({'Bounding Box Center Position (LPS):': tuple(np_boundingbox_center_pos[:].tolist())})

    return results


# ------------------------------------------------------------------------------
# renaming the foreground labels label
def refine_segment_from_2d_seeds(sitkImage, sitkLabelSeeds, **kwargs):
    """
    :param sitkImage:
    :param sitkLabelSeeds:
    :param kwargs:
    :return:
    """
    # foreground and background
    nb_cluster = 2

    # recounting the connected labels
    sitkLabel = sitk.ConnectedComponent(sitkLabelSeeds)
    sitkLabel = sitk.RelabelComponent(sitkLabel)
    sitkLabel = sitk.Cast(sitkLabel, sitk.sitkUInt8)

    # read the label info
    dim = sitkLabel.GetDimension()
    spacing = sitkLabel.GetSpacing()
    size = sitkLabel.GetSize()
    direction = sitkLabel.GetDirection()
    np_spacing = np.array(spacing, np.float)
    np_size = np.array(size, np.float)

    # get the bounding box
    lsf = sitk.LabelStatisticsImageFilter()
    lsf.Execute(sitkLabel, sitkLabel)
    labels = list(lsf.GetLabels())
    if 0 in labels: labels.remove(0)

    # working on foreground labels
    for label in labels:
        # read the roi specified by label
        boundingbox = lsf.GetBoundingBox(label)
        np_boundingbox = np.array(boundingbox, dtype=np.float).reshape(dim, 2)
        np_boundingbox_size = np_boundingbox[:, 1] - np_boundingbox[:, 0] + 1.0
        np_boundingbox_center_idx = np_boundingbox[:, 0] + np_boundingbox_size / 2
        np_boundingbox_vol_size = np.repeat(np.max(np_boundingbox_size * np_spacing), dim, axis=0) / np_spacing
        np_boundingbox_vol_size = np_boundingbox_vol_size.clip(np.zeros(dim, dtype=np.float), np_size)
        np_boundingbox_origin_idx = np_boundingbox_center_idx - np_boundingbox_vol_size / 2
        np_boundingbox_origin_idx = np_boundingbox_origin_idx.clip(np.zeros(dim, dtype=np.float), np_size)
        boundingbox_idx = np_boundingbox_origin_idx.astype(np.uint).tolist()
        boundingbox_size = np_boundingbox_vol_size.astype(np.uint).tolist()
        sitkImageROI = sitk.RegionOfInterest(sitkImage, boundingbox_size, boundingbox_idx)

        # classify the roi
        # scipy / scikit-image processing of images
        np_roi_array = sitk.GetArrayFromImage(sitkImageROI)
        np_roi_array_shape = np_roi_array.shape
        np_roi_array_idx = np_roi_array != 0
        np_roi_array_1d = np_roi_array[np_roi_array_idx].astype(np.float)
        # to classify using kmean
        (np_centroid, np_cluster_labels) = cluster.vq.kmeans2(np_roi_array_1d, k=nb_cluster)
        for i_c in range(nb_cluster, 0, -1):
            np_cluster_labels[np_cluster_labels == i_c - 1] = i_c
        np_cluster_labels_array = np_roi_array
        np_cluster_labels_array[np_roi_array_idx] = np_cluster_labels
        np_cluster_labels_array = np_cluster_labels_array.reshape(np_roi_array_shape)
        # to processing labels using scikit-image
        np_roi_labels = watershed(np_roi_array, np_cluster_labels_array)
        # to convert the uint8 type for paste
        np_roi_labels = np_roi_labels.astype(np.uint8)

        # read back to sitkLabel
        sitkLabel_ROI = sitk.GetImageFromArray(np_roi_labels)
        sitkLabel_ROI.SetDirection(direction)
        sitkLabel_ROI.SetSpacing(spacing)

        sitkLabel = sitk.Paste(sitkLabel, sitkLabel_ROI, sitkLabel_ROI.GetSize(),
                               tuple(np.zeros(dim, dtype=np.uint8).tolist()), boundingbox_idx)
    return sitkLabel


# ------------------------------------------------------------------------------
#
def randomwalker_segment_from_2d_seeds(sitkImage, sitkLabelSeeds, **kwargs):
    """
    :param sitkImage:
    :param sitkLabelSeeds:
    :param kwargs:
    :return:
    """
    nb_cluster = 3
    # recounting the connected labels
    sitkLabel = sitk.ConnectedComponent(sitkLabelSeeds)
    sitkLabel = sitk.RelabelComponent(sitkLabel)
    sitkLabel = sitk.Cast(sitkLabel, sitk.sitkUInt8)

    # read the label info
    dim = sitkLabel.GetDimension()
    spacing = sitkLabel.GetSpacing()
    size = sitkLabel.GetSize()
    direction = sitkLabel.GetDirection()
    np_spacing = np.array(spacing, np.float)
    np_size = np.array(size, np.float)

    # get the bounding box
    lsf = sitk.LabelStatisticsImageFilter()
    lsf.Execute(sitkImage, sitkLabel)
    labels = list(lsf.GetLabels())
    if 0 in labels: labels.remove(0)

    # working on foreground labels
    for label in labels:
        # read the roi specified by label
        boundingbox = lsf.GetBoundingBox(label)
        np_boundingbox = np.array(boundingbox, dtype=np.float).reshape(dim, 2)
        np_boundingbox_origin_idx = np_boundingbox[:, 0]
        np_boundingbox_size = np_boundingbox[:, 1] - np_boundingbox[:, 0] + 1.0
        boundingbox_idx = np_boundingbox_origin_idx.astype(np.uint).tolist()
        boundingbox_size = np_boundingbox_size.astype(np.uint).tolist()
        sitkImageROI = sitk.RegionOfInterest(sitkImage, boundingbox_size, boundingbox_idx)
        sitkLabelROI = sitk.RegionOfInterest(sitkLabel, boundingbox_size, boundingbox_idx)
        sitkImageROI = sitk.Mask(sitkImageROI, sitkLabelROI)

        # classify the roi
        # scipy / scikit-image processing of images
        np_roi_array = sitk.GetArrayFromImage(sitkImageROI)
        np_roi_array_idx = np_roi_array != 0
        np_roi_array_1d = np_roi_array[np_roi_array_idx].astype(np.float)
        # to classify using kmean
        (np_centroid, np_cluster_labels) = cluster.vq.kmeans2(np_roi_array_1d, k=nb_cluster)

        # VOI
        np_boundingbox_center_idx = np_boundingbox[:, 0] + np_boundingbox_size / 2
        np_boundingbox_vol_size = np.repeat(np.max(np_boundingbox_size * np_spacing), dim, axis=0) / np_spacing
        np_boundingbox_vol_size = np_boundingbox_vol_size.clip(np.zeros(dim, dtype=np.float), np_size)
        np_boundingbox_origin_idx = np_boundingbox_center_idx - np_boundingbox_vol_size / 2
        np_boundingbox_origin_idx = np_boundingbox_origin_idx.clip(np.zeros(dim, dtype=np.float), np_size)
        boundingbox_idx = np_boundingbox_origin_idx.astype(np.uint).tolist()
        boundingbox_size = np_boundingbox_vol_size.astype(np.uint).tolist()
        sitkImageROI = sitk.RegionOfInterest(sitkImage, boundingbox_size, boundingbox_idx)

        np_roi_array = sitk.GetArrayFromImage(sitkImageROI)
        np_roi_array_shape = np_roi_array.shape

        np_cluster_labels_new = np.zeros_like(np_roi_array, dtype=np.uint8)
        np_cluster_labels_new[np_roi_array > np_centroid[np_centroid.argsort()[-1]]] = 2
        np_cluster_labels_new[np_roi_array <= np_centroid[np_centroid.argsort()[0]]] = 1

        # to processing labels using scikit-image
        # np_roi_labels = watershed(np_roi_array, np_cluster_labels_array)
        np_roi_labels = random_walker(np_roi_array, np_cluster_labels_new)
        np_roi_labels[np_roi_labels == 1] = 0
        # to convert the uint8 type for paste
        np_roi_labels = np_roi_labels.astype(np.uint8)

        # read back to sitkLabel
        sitkLabel_ROI = sitk.GetImageFromArray(np_roi_labels)
        sitkLabel_ROI.SetDirection(direction)
        sitkLabel_ROI.SetSpacing(spacing)
        sitkLabelROI = sitk.ConnectedComponent(sitkLabelROI)
        # sitkLabel_ROI = get_largest_connected_labels(sitkLabelROI)

        sitkLabel = sitk.Paste(sitkLabel, sitkLabel_ROI, sitkLabel_ROI.GetSize(),
                               tuple(np.zeros(dim, dtype=np.uint8).tolist()), boundingbox_idx)
    return sitkLabel