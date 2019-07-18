# ------------------------------------------------------------------------------
# imports
import SimpleITK as sitk
import numpy as np
from scipy import ndimage

from .resampling import resample_sitkImage_by_reference as crop_resample_reference


# ------------------------------------------------------------------------------
#
def local_maxium_filter(sitkImage, sitkShape=(1, 1, 1)):
    """
    #:param sitkImage:   3D SimpleITK image
    #:param sitkShape:   shape definition in mm (x, y, z)
    #:return:            3D SimpleITK image with the same size as input
    """
    if sitkImage is None: return sitkImage

    origin = sitkImage.GetOrigin()
    spacing = sitkImage.GetSpacing()

    _l = len(sitkShape)
    shape_size = (int(sitkShape[_l - i - 1] / spacing[_l - i - 1]) for i in range(_l))
    img_array = sitk.GetArrayFromImage(sitkImage)
    img_array_max_filtered = ndimage.maximum_filter(img_array, size=shape_size, mode='reflect')

    sitkImageMaxFiltered = sitk.GetImageFromArray(img_array_max_filtered)
    sitkImageMaxFiltered.SetSpacing(spacing)
    sitkImageMaxFiltered.SetOrigin(origin)

    return sitkImageMaxFiltered


# ------------------------------------------------------------------------------
# find region corner index
def find_corner_index(line, threshold):
    _l = len(line)
    down = 0
    up = _l - 1
    for i in range(_l):
        if line[i] > threshold:
            down = i
            break
    for i in range(_l - 1, -1, -1):
        if line[i] > threshold:
            up = i
            break
    return down, up


# ------------------------------------------------------------------------------
# crop sitkImage to the boundary of referenced mask
# example:
#   crop_mask_region(sitkImage, sitkMask)
#   crop_mask_region(sitkImage, sitkMask, spare_boundary_mm=(6,6,6))
def crop_mask_region(sitkImage, sitkMask, **kwargs):
    """
    :param sitkImage:
    :param sitkMask:
    :param kwargs: spare_boundary_mm = (1,1,1)
    :return:
    """
    sizeMask = sitkMask.GetSize()
    sizeImage = sitkImage.GetSize()
    if sizeMask != sizeImage: return None

    np_imgs = sitk.GetArrayFromImage(sitkMask)

    np_imgs_xy = np.sum(np_imgs, axis=0)
    x_line = np.sum(np_imgs_xy, axis=0)
    y_line = np.sum(np_imgs_xy, axis=1)

    np_imgs_yz = np.sum(np_imgs, axis=2)
    z_line = np.sum(np_imgs_yz, axis=1)

    # 2D image parameters
    slices, height, width = np_imgs.shape

    x_1, x_2 = find_corner_index(x_line, 0.9)
    y_1, y_2 = find_corner_index(y_line, 0.9)
    z_1, z_2 = find_corner_index(z_line, 0.9)

    # print('(x_1, y_1, z_1):')
    # print((x_1, y_1, z_1))
    # print('(x_2, y_2, z_2):')
    # print((x_2, y_2, z_2))

    # spare_boundary_mm = (6, 6, 6)
    new_spare_boundary_mm = None
    spare_boundary_mm = kwargs.get('spare_boundary_mm')
    if spare_boundary_mm.__class__ == tuple and len(spare_boundary_mm) == 3: \
            new_spare_boundary_mm = spare_boundary_mm
    elif spare_boundary_mm.__class__ == int or spare_boundary_mm.__class__ == float: \
            new_spare_boundary_mm = (spare_boundary_mm, spare_boundary_mm, spare_boundary_mm)
    else:
        new_spare_boundary_mm = (0.0, 0.0, 0.0)
    if new_spare_boundary_mm is not None:
        spacing = sitkImage.GetSpacing()
        # x_1 = max(0, x_1-1 - int(new_spare_boundary_mm[0]/spacing[0]))
        # x_2 = min(width, x_2 +1+ int(new_spare_boundary_mm[0]/spacing[0]))
        # y_1 = max(0, y_1-1 - int(new_spare_boundary_mm[1]/spacing[1]))
        # y_2 = min(height, y_2+1 + int(new_spare_boundary_mm[1]/spacing[1]))
        # z_1 = max(0, z_1-1 - int(new_spare_boundary_mm[2]/spacing[2]))
        # z_2 = min(slices, z_2+1 + int(new_spare_boundary_mm[2]/spacing[2]))

        x_1 = max(0, x_1 - int(new_spare_boundary_mm[0] / spacing[0]))
        x_2 = min(width, x_2 + int(new_spare_boundary_mm[0] / spacing[0]))
        y_1 = max(0, y_1 - int(new_spare_boundary_mm[1] / spacing[1]))
        y_2 = min(height, y_2 + int(new_spare_boundary_mm[1] / spacing[1]))
        z_1 = max(0, z_1 - int(new_spare_boundary_mm[2] / spacing[2]))
        z_2 = min(slices, z_2 + int(new_spare_boundary_mm[2] / spacing[2]))

    lowerBoundary = (x_1, y_1, z_1)
    upperBoundary = (width - 1 - x_2, height - 1 - y_2, slices - 1 - z_2)

    # print('np_imgs.shape:')
    # print(np_imgs.shape)
    # print('lowerBoundary:')
    # print(lowerBoundary)
    # print('upperBoundary:')
    # print(upperBoundary)

    return sitk.Crop(sitkImage, lowerBoundary, upperBoundary)


# ------------------------------------------------------------------------------
# crop sitkImage to the boundary of referenced mask with a single slice occupied the largest mask area
# example:
#   crop_mask_region(sitkImage, sitkMask)
#   crop_mask_region(sitkImage, sitkMask, spare_boundary_mm=(6,6,6))
def crop_mask_region_with_largest_mask_area_slice(sitkImage, sitkMask, **kwargs):
    """
    :param sitkImage:
    :param sitkMask:
    :param kwargs: spare_boundary_mm = (1,1,1)
    :return:
    """
    sizeMask = sitkMask.GetSize()
    sizeImage = sitkImage.GetSize()
    if sizeMask != sizeImage: return None

    np_imgs = sitk.GetArrayFromImage(sitkMask)

    np_imgs_xy = np.sum(np_imgs, axis=0)
    x_line = np.sum(np_imgs_xy, axis=0)
    y_line = np.sum(np_imgs_xy, axis=1)

    np_imgs_yz = np.sum(np_imgs, axis=2)
    z_line = np.sum(np_imgs_yz, axis=1)

    # 2D image parameters
    slices, height, width = np_imgs.shape

    x_1, x_2 = find_corner_index(x_line, 0.9)
    y_1, y_2 = find_corner_index(y_line, 0.9)
    z_1, z_2 = find_corner_index(z_line, 0.9)
    non_zero_slices = z_2 - z_1 + 1

    # get the slice index with the largest area of mask
    bb = z_line.tolist()
    target_slice_index = bb.index(max(bb))
    z_1 = target_slice_index
    z_2 = target_slice_index

    # print('(x_1, y_1, z_1):')
    # print((x_1, y_1, z_1))
    # print('(x_2, y_2, z_2):')
    # print((x_2, y_2, z_2))

    # spare_boundary_mm = (6, 6, 6)
    new_spare_boundary_mm = None
    spare_boundary_mm = kwargs.get('spare_boundary_mm')
    if spare_boundary_mm.__class__ == tuple and len(spare_boundary_mm) == 3: \
            new_spare_boundary_mm = spare_boundary_mm
    elif spare_boundary_mm.__class__ == int or spare_boundary_mm.__class__ == float: \
            new_spare_boundary_mm = (spare_boundary_mm, spare_boundary_mm, spare_boundary_mm)
    else:
        new_spare_boundary_mm = (0.0, 0.0, 0.0)
    if new_spare_boundary_mm is not None:
        spacing = sitkImage.GetSpacing()
        # x_1 = max(0, x_1-1 - int(new_spare_boundary_mm[0]/spacing[0]))
        # x_2 = min(width, x_2 +1+ int(new_spare_boundary_mm[0]/spacing[0]))
        # y_1 = max(0, y_1-1 - int(new_spare_boundary_mm[1]/spacing[1]))
        # y_2 = min(height, y_2+1 + int(new_spare_boundary_mm[1]/spacing[1]))
        # z_1 = max(0, z_1-1 - int(new_spare_boundary_mm[2]/spacing[2]))
        # z_2 = min(slices, z_2+1 + int(new_spare_boundary_mm[2]/spacing[2]))

        x_1 = max(0, x_1 - int(new_spare_boundary_mm[0] / spacing[0]))
        x_2 = min(width, x_2 + int(new_spare_boundary_mm[0] / spacing[0]))
        y_1 = max(0, y_1 - int(new_spare_boundary_mm[1] / spacing[1]))
        y_2 = min(height, y_2 + int(new_spare_boundary_mm[1] / spacing[1]))

    lowerBoundary = (x_1, y_1, z_1)
    upperBoundary = (width - 1 - x_2, height - 1 - y_2, slices - 1 - z_2)

    # print('np_imgs.shape:')
    # print(np_imgs.shape)
    # print('lowerBoundary:')
    # print(lowerBoundary)
    # print('upperBoundary:')
    # print(upperBoundary)

    return sitk.Crop(sitkImage, lowerBoundary, upperBoundary), non_zero_slices


# ------------------------------------------------------------------------------
# crop sitkImage to the boundary of referenced mask
# example:
#   crop_mask_region(sitkImage, sitkMask)
#   crop_mask_region(sitkImage, sitkMask, spare_boundary_mm=(6,6,6))
def crop_mask_region_and_generate_bbox(sitkImage, sitkMask, **kwargs):
    """
    :param sitkImage:
    :param sitkMask:
    :param kwargs: spare_boundary_mm = (1,1,1)
    :return:
    """
    sizeMask = sitkMask.GetSize()
    sizeImage = sitkImage.GetSize()
    if sizeMask != sizeImage: return None

    np_imgs = sitk.GetArrayFromImage(sitkMask)

    np_imgs_xy = np.sum(np_imgs, axis=0)
    x_line = np.sum(np_imgs_xy, axis=0)
    y_line = np.sum(np_imgs_xy, axis=1)

    np_imgs_yz = np.sum(np_imgs, axis=2)
    z_line = np.sum(np_imgs_yz, axis=1)

    # 2D image parameters
    slices, height, width = np_imgs.shape

    x_1, x_2 = find_corner_index(x_line, 0.9)
    y_1, y_2 = find_corner_index(y_line, 0.9)
    z_1, z_2 = find_corner_index(z_line, 0.9)

    # spare_boundary_mm = (6, 6, 6)
    new_spare_boundary_mm = None
    # spare_boundary_mm = kwargs.get('spare_boundary_mm')
    # if spare_boundary_mm.__class__ == tuple and len(spare_boundary_mm) == 3: \
    #         new_spare_boundary_mm = spare_boundary_mm
    # elif spare_boundary_mm.__class__ == int or spare_boundary_mm.__class__ == float: \
    #         new_spare_boundary_mm = (spare_boundary_mm, spare_boundary_mm, spare_boundary_mm)
    # else: new_spare_boundary_mm = (0.0, 0.0, 0.0)
    # if new_spare_boundary_mm is not None:
    #     spacing = sitkImage.GetSpacing()
    #     x_1 = max(0, x_1-1 - int(new_spare_boundary_mm[0]/spacing[0]))
    #     x_2 = min(width, x_2+1 + int(new_spare_boundary_mm[0]/spacing[0]))
    #     y_1 = max(0, y_1-1 - int(new_spare_boundary_mm[1]/spacing[1]))
    #     y_2 = min(height, y_2+1 + int(new_spare_boundary_mm[1]/spacing[1]))
    #     z_1 = max(0, z_1-1 - int(new_spare_boundary_mm[2]/spacing[2]))
    #     z_2 = min(slices, z_2 +1+ int(new_spare_boundary_mm[2]/spacing[2]))

    lowerBoundary = (x_1, y_1, z_1)
    upperBoundary = (width - 1 - x_2, height - 1 - y_2, slices - 1 - z_2)

    cropped_images = sitk.Crop(sitkImage, lowerBoundary, upperBoundary)

    np_size = np.array(sitkImage.GetSize())
    size = tuple(np_size.astype(np.int).tolist())
    size = size[::-1]

    np_image_array = np.zeros(size, np.uint8)
    np_image_array[z_1:z_2 + 1, y_1:y_2 + 1, x_1:x_2 + 1] = 1
    new_sitkLabel = sitk.GetImageFromArray(np_image_array)
    new_sitkLabel.SetDirection(sitkImage.GetDirection())
    new_sitkLabel.SetSpacing(sitkImage.GetSpacing())

    # calculate and set origin
    new_sitkLabel.SetOrigin(sitkImage.GetOrigin())

    return cropped_images, new_sitkLabel


# ------------------------------------------------------------------------------
# crop sitkImage to the boundary of referenced mask with a single slice occupied the largest mask area
# example:
#   crop_mask_region(sitkImage, sitkMask)
#   crop_mask_region(sitkImage, sitkMask, spare_boundary_mm=(6,6,6))
def crop_full_mask_region_with_largest_mask_area_slice(sitkImage, sitkMask, **kwargs):
    """
    :param sitkImage:
    :param sitkMask:
    :param kwargs: spare_boundary_mm = (1,1,1)
    :return:
    """
    sizeMask = sitkMask.GetSize()
    sizeImage = sitkImage.GetSize()
    if sizeMask != sizeImage: return None

    np_imgs = sitk.GetArrayFromImage(sitkMask)

    np_imgs_xy = np.sum(np_imgs, axis=0)
    x_line = np.sum(np_imgs_xy, axis=0)
    y_line = np.sum(np_imgs_xy, axis=1)

    np_imgs_yz = np.sum(np_imgs, axis=2)
    z_line = np.sum(np_imgs_yz, axis=1)

    # 2D image parameters
    slices, height, width = np_imgs.shape

    x_1, x_2 = find_corner_index(x_line, 0.9)
    y_1, y_2 = find_corner_index(y_line, 0.9)
    z_1, z_2 = find_corner_index(z_line, 0.9)

    # print('(x_1, y_1, z_1):')
    # print((x_1, y_1, z_1))
    # print('(x_2, y_2, z_2):')
    # print((x_2, y_2, z_2))

    x_1 = max(0, x_1)
    x_2 = min(width, x_2)
    y_1 = max(0, y_1)
    y_2 = min(height, y_2)
    z_1 = max(0, z_1)
    z_2 = min(slices, z_2)

    lowerBoundary = (x_1, y_1, z_1)
    upperBoundary = (width - 1 - x_2, height - 1 - y_2, slices - 1 - z_2)

    # print('np_imgs.shape:')
    # print(np_imgs.shape)
    # print('lowerBoundary:')
    # print(lowerBoundary)
    # print('upperBoundary:')
    # print(upperBoundary)

    return sitk.Crop(sitkImage, lowerBoundary, upperBoundary)


# --------------------------------------------------------------------------------
# computing max bounding box and crop all to the max bounding box
def crop_to_max_bounding_box(sitkImageList, sitkLabelList, spare_boundary_mm, crop_axis=(0, 0, 1)):
    """
    # the first sitkLabel in sitkLabelList is used as reference
    :param sitkImageList:
    :param sitkLabelList:
    :param spare_boundary_mm:
    :return:
    """
    # check whether all have the same physical settings, and return None if not
    ref_label = None
    for i, sitkLabel in enumerate(sitkLabelList):
        if sitkLabel is None: return None
        if ref_label is None: ref_label = sitkLabel
        check_label = [ref_label.GetOrigin() == sitkLabel.GetOrigin(),
                       ref_label.GetSpacing() == sitkLabel.GetSpacing(),
                       ref_label.GetSize() == sitkLabel.GetSize(),
                       ref_label.GetDirection == sitkLabel.GetDirection()]
        if False in check_label: sitkLabelList[i] = crop_resample_reference(ref_label, sitkLabel)
    for i, sitkImage in enumerate(sitkImageList):
        if sitkImage is None: return None
        check_image = [ref_label.GetOrigin() == sitkImage.GetOrigin(),
                       ref_label.GetSpacing() == sitkImage.GetSpacing(),
                       ref_label.GetSize() == sitkImage.GetSize(),
                       ref_label.GetDirection == sitkImage.GetDirection()]
        if False in check_image: sitkImageList[i] = crop_resample_reference(ref_label, sitkImage)

    # get the physical settings for following usage
    dim = ref_label.GetDimension()
    origin = ref_label.GetOrigin()
    spacing = ref_label.GetSpacing()
    direction = ref_label.GetDirection()
    size = ref_label.GetSize()

    # setting the spare_boundary_mm
    if spare_boundary_mm.__class__ == tuple:
        tmp_spare_boundary_mm = spare_boundary_mm
    elif spare_boundary_mm.__class__ in [int, float]:
        tmp_spare_boundary_mm = [spare_boundary_mm] * dim
    else:
        tmp_spare_boundary_mm = (0.0, 0.0, 0.0)
    np_spare_boundary_mm = np.array(tmp_spare_boundary_mm, dtype=np.float)

    # check the crop_axis
    if len(crop_axis) != dim: crop_axis = tuple([1] * dim)

    # assign the matrix
    np_image_array = sitk.GetArrayFromImage(ref_label)
    for sitkLabel in sitkLabelList:
        tmp_np_image_array = sitk.GetArrayFromImage(sitkLabel)
        np_image_array[tmp_np_image_array != 0] = 1
    new_ref_label = sitk.Cast(sitk.GetImageFromArray(np_image_array), sitk.sitkUInt8)
    new_ref_label.SetOrigin(origin)
    new_ref_label.SetSpacing(spacing)
    new_ref_label.SetDirection(direction)

    # get the maxium boundary
    lsf = sitk.LabelStatisticsImageFilter()
    lsf.Execute(new_ref_label, new_ref_label)
    np_bounding_box = np.array(lsf.GetBoundingBox(1), dtype=np.float).reshape((dim, 2))
    np_spacing = np.array(spacing, dtype=np.float)
    np_sparing_size = np.divide(np_spare_boundary_mm, np_spacing).astype(np.float)
    np_bounding_box[:, 0] -= np_sparing_size
    np_bounding_box[:, 1] += np_sparing_size
    np_zeros = np.zeros(dim, dtype=np.float)
    np_ones = np.ones(dim, dtype=np.float)
    np_size = np.array(size, dtype=np.float)
    np_low_bounding_idx = np_bounding_box[:, 0].clip(np_zeros, np_size)
    np_upper_bounding_idx = np_bounding_box[:, 1].clip(np_zeros, np_size)
    np_bounding_box_size = (np_upper_bounding_idx - np_low_bounding_idx + (1, 1, 1)).clip(np_ones, np_size)

    for i, axis in enumerate(crop_axis):
        if axis == 0:
            np_low_bounding_idx[i] = 0
            np_bounding_box_size[i] = np_size[i]

    low_bounding_idx = tuple(np_low_bounding_idx.astype(np.uint).tolist())
    bounding_box_size = tuple(np_bounding_box_size.astype(np.uint).tolist())

    # retrieve ROI
    new_sitkImageList = []
    for sitkImage in sitkImageList:
        new_sitkImageList.append(sitk.RegionOfInterest(sitkImage, bounding_box_size, low_bounding_idx))
    new_sitkLabelList = []
    for sitkLabel in sitkLabelList:
        new_sitkLabelList.append(sitk.RegionOfInterest(sitkLabel, bounding_box_size, low_bounding_idx))

    return new_sitkImageList, new_sitkLabelList
