# ------------------------------------------------------------------------------
# imports
import SimpleITK as sitk
import numpy as np

# ------------------------------------------------------------------------------
# resampling the sitkImages with center aligned
def resample_sitkImage_centerAligned(sitkImage, newSize, newSpacing, vol_default_value='min'):
    """
    :param sitkImage:
    :param newSize:
    :param newSpacing:
    :param vol_default_value:
    :return:
    """
    if sitkImage is None: return None
    dim = sitkImage.GetDimension()
    if len(newSize) != dim: return None
    if len(newSpacing) != dim: return None

    vol_value = 0.0
    if vol_default_value == 'min':
        vol_value = float(np.ndarray.min(sitk.GetArrayFromImage(sitkImage)))
    elif vol_default_value == 'zero':
        vol_value = 0.0
    elif str(vol_default_value).isnumeric():
        vol_value = float(vol_default_value)

    np_old_size = np.array(sitkImage.GetSize(), dtype=np.int).reshape(dim, 1)
    np_old_spacing = np.array(sitkImage.GetSpacing(), dtype=np.float).reshape(dim, 1)
    np_old_origin = np.array(sitkImage.GetOrigin(), dtype=np.float).reshape(dim, 1)

    np_matrix_direction = np.array(sitkImage.GetDirection()).reshape(dim, dim)

    np_calibrated_shift = np.dot(np_matrix_direction, np_old_spacing * np_old_size / 2)
    np_center = np_old_origin + np_calibrated_shift

    np_new_size = np.array(newSize, dtype=np.int).reshape(dim, 1)
    np_new_spacing = np.array(newSpacing, dtype=np.float).reshape(dim, 1)
    np_calibrated_shift = np.dot(np_matrix_direction, np_new_spacing * np_new_size / 2)
    np_new_origin = np_center - np_calibrated_shift

    newOrigin = tuple(np_new_origin[:, 0].tolist())

    transform = sitk.Transform()
    centerAlignedResampledsitkImage = sitk.Resample(sitkImage, newSize, transform,
                                                    sitk.sitkNearestNeighbor, newOrigin, newSpacing,
                                                    sitkImage.GetDirection(), vol_value, sitkImage.GetPixelID())

    return centerAlignedResampledsitkImage


# ------------------------------------------------------------------------------
# resampling the sitkImages with origin aligned
def resample_sitkImage_originAligned(sitkImage, newSize, newSpacing, vol_default_value='min'):
    """
    :param sitkImage:
    :param newSize:
    :param newSpacing:
    :return:
    """
    if sitkImage == None: return None
    if newSize is None: return None
    if newSpacing is None: return None
    dim = sitkImage.GetDimension()
    if len(newSize) != dim: return None
    if len(newSpacing) != dim: return None

    vol_value = 0.0
    if vol_default_value == 'min':
        vol_value = float(np.ndarray.min(sitk.GetArrayFromImage(sitkImage)))
    elif vol_default_value == 'zero':
        vol_value = 0.0
    elif str(vol_default_value).isnumeric():
        vol_value = float(vol_default_value)

    # resample sitkImaging into new specs
    transform = sitk.Transform()
    return sitk.Resample(sitkImage, newSize, transform, sitk.sitkNearestNeighbor, sitkImage.GetOrigin(),
                         newSpacing, sitkImage.GetDirection(), vol_value, sitkImage.GetPixelID())


# ------------------------------------------------------------------------------
# resampling the sitkImages with new spacing
def resample_sitkImage_by_spacing(sitkImage, newSpacing, vol_default_value='min'):
    """
    :param sitkImage:
    :param newSpacing:
    :return:
    """
    if sitkImage == None: return None
    if newSpacing is None: return None
    dim = sitkImage.GetDimension()
    if len(newSpacing) != dim: return None

    # determine the default value
    vol_value = 0.0
    if vol_default_value == 'min':
        vol_value = float(np.ndarray.min(sitk.GetArrayFromImage(sitkImage)))
    elif vol_default_value == 'zero':
        vol_value = 0.0
    elif str(vol_default_value).isnumeric():
        vol_value = float(vol_default_value)

    # calculate new size
    np_oldSize = np.array(sitkImage.GetSize())
    np_oldSpacing = np.array(sitkImage.GetSpacing())
    np_newSpacing = np.array(newSpacing)
    np_newSize = np.divide(np.multiply(np_oldSize, np_oldSpacing), np_newSpacing)
    newSize = tuple(np_newSize.astype(np.uint).tolist())

    # resample sitkImage into new specs
    transform = sitk.Transform()
    return sitk.Resample(sitkImage, newSize, transform, sitk.sitkNearestNeighbor, sitkImage.GetOrigin(),
                         newSpacing, sitkImage.GetDirection(), vol_value, sitkImage.GetPixelID())


# ------------------------------------------------------------------------------
# resampling the sitkImages with new spacing
def resample_sitkImage_by_size(sitkImage, newSize, vol_default_value='min'):
    """
    :param sitkImage:
    :param newSize:
    :return:
    """
    if sitkImage == None: return None
    if newSize is None: return None
    dim = sitkImage.GetDimension()
    if len(newSize) != dim: return None

    # determine the default value
    vol_value = 0.0
    if vol_default_value == 'min':
        vol_value = float(np.ndarray.min(sitk.GetArrayFromImage(sitkImage)))
    elif vol_default_value == 'zero':
        vol_value = 0.0
    elif str(vol_default_value).isnumeric():
        vol_value = float(vol_default_value)

    # calculate new size
    np_oldSize = np.array(sitkImage.GetSize())
    np_oldSpacing = np.array(sitkImage.GetSpacing())
    np_newSize = np.array(newSize)
    np_newSpacing = np.divide(np.multiply(np_oldSize, np_oldSpacing), np_newSize)
    newSpacing = tuple(np_newSpacing.astype(np.float).tolist())

    # resample sitkImage into new specs
    transform = sitk.Transform()
    return sitk.Resample(sitkImage, newSize, transform, sitk.sitkNearestNeighbor, sitkImage.GetOrigin(),
                         newSpacing, sitkImage.GetDirection(), vol_value, sitkImage.GetPixelID())


# ------------------------------------------------------------------------------
# resampling the sitkImage by extension to keep the same scale
def resample_sitkImage_by_extension(sitkImage, newSize, vol_default_value='min'):
    """
    :param sitkImage:
    :param newSize:
    :return:
    """
    if sitkImage == None: return None
    if newSize is None: return None
    dim = sitkImage.GetDimension()
    if len(newSize) != dim: return None
    if dim < 2: return None

    # determine the default value
    vol_value = 0.0
    if vol_default_value == 'min':
        vol_value = float(np.ndarray.min(sitk.GetArrayFromImage(sitkImage)))
    elif vol_default_value == 'zero':
        vol_value = 0.0
    elif str(vol_default_value).isnumeric():
        vol_value = float(vol_default_value)

    # calculate new size
    np_oldSize = np.array(sitkImage.GetSize())
    np_newSize = np.array(newSize)
    check_dim = [np_oldSize[each_dim] < np_newSize[each_dim] for each_dim in range(dim)]

    np_intermediate_size = []

    for i, each_check in enumerate(check_dim):
        if each_check == True:
            np_intermediate_size.append(np_newSize[i])
        else:
            np_intermediate_size.append(np_oldSize[i])

    start_dim = [int((np_intermediate_size[each_dim] - np_oldSize[each_dim]) / 2) for each_dim in range(dim)]
    start_dim = start_dim[::-1]
    np_oldSize = np_oldSize[::-1]

    new_im = np.zeros(np_intermediate_size[::-1], np.float)
    old_im = sitk.GetArrayFromImage(sitkImage)
    if dim == 2:
        new_im[start_dim[0]: start_dim[0] + np_oldSize[0], start_dim[1]: start_dim[1] + np_oldSize[1]] = old_im

    if dim == 3:
        new_im[start_dim[0]: start_dim[0] + np_oldSize[0], start_dim[1]: start_dim[1] + np_oldSize[1], \
        start_dim[2]: start_dim[2] + np_oldSize[2]] = old_im

    sitkImage1 = sitk.GetImageFromArray(new_im)
    sitkImage1.SetOrigin(sitkImage.GetOrigin())
    sitkImage1.SetSpacing(sitkImage.GetSpacing())
    sitkImage1.SetDirection(sitkImage.GetDirection())

    np_oldSize1 = np.array(sitkImage1.GetSize())
    check_dim1 = [np_oldSize1[each_dim] > np_newSize[each_dim] for each_dim in range(dim)]
    if True in check_dim1:
        # calculate new size
        np_oldSpacing1 = np.array(sitkImage1.GetSpacing())
        np_newSize = np.array(newSize)
        np_newSpacing = np.divide(np.multiply(np_oldSize1, np_oldSpacing1), np_newSize)
        newSpacing = tuple(np_newSpacing.astype(np.float).tolist())

        # resample sitkImage into new specs
        # sitk.sitkNearestNeighbor
        transform = sitk.Transform()
        return sitk.Resample(sitkImage1, newSize, transform, sitk.sitkLinear, sitkImage.GetOrigin(),
                             newSpacing, sitkImage.GetDirection(), vol_value, sitkImage.GetPixelID())
    else:
        return sitkImage1


# -------------------------------------------------------------------------------
# resample sitkImage to referenced sitkImage
def resample_sitkImage_by_reference(reference, image):
    """
    :param reference:
    :param image:
    :return:
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetReferenceImage(reference)
    return resampler.Execute(image)


# ------------------------------------------------------------------------------
# the result boundary images size may not equal depending on the grid and overlapping setting
#
def split_sitkImage_into_BlocksList(sitkImage, blockGridSize=(1, 1, 1), blockOverlappingSize=(0, 0, 0)):
    """
    :param sitkImage:
    :param blockGridSize:
    :param blockOverlappingSize:
    :return:
    """
    # check the inputs
    if sitkImage is None: return []
    if blockGridSize == (1, 1, 1): return [sitkImage]
    dim = sitkImage.GetDimension()
    if len(blockGridSize) != dim: return []
    if len(blockOverlappingSize) != dim: return []
    # clip the input -> convert to numpy array
    np_sitkImageSize = np.array(sitkImage.GetSize(), dtype=np.int)
    np_blockGridSize = np.array(blockGridSize, dtype=np.int).clip(1, np_sitkImageSize)
    np_blockOverlappingSize = np.array(blockOverlappingSize, dtype=np.int)
    np_blockImageSize = (np_sitkImageSize / np_blockGridSize).astype(np.int)
    np_blockOverlappingSize = np_blockOverlappingSize.clip(0, np_blockImageSize)
    np_blockImageSize_overlapped = np_blockImageSize + np_blockOverlappingSize

    # forming a np_xyz to go through [x, y, z] with sequence of x->y->z - 'F' order
    np_xyz = np.zeros(np_blockGridSize, dtype=np.int)
    np_xyz_flatten = np_xyz.flatten()
    for i in range(np_xyz_flatten.shape[0]):
        np_xyz_flatten[i] = i
    np_xyz = np_xyz_flatten.reshape(np_blockGridSize, order='F')

    # split in the order of np_xyz
    sitkImageBlocksList = []
    for i in range(np_xyz_flatten.shape[0]):
        np_xyz_idx = np.argwhere(np_xyz == i)[0]
        np_block_idx = (np_xyz_idx * np_blockImageSize).astype(np.int)
        np_tmp = np.stack([np_blockImageSize_overlapped, np_sitkImageSize - np_block_idx], axis=1)
        np_final_blockImageSize = np.min(np_tmp, axis=1)

        final_blockImageSize = tuple(np_final_blockImageSize.tolist())
        final_block_idx = tuple(np_block_idx.tolist())
        sitkImageBlocksList.append(sitk.RegionOfInterest(sitkImage, size=final_blockImageSize,
                                                         index=final_block_idx))

    return sitkImageBlocksList


# ------------------------------------------------------------------------------
# the result images size is the equal as defined in blockImageSize
#
def resample_sitkImage_into_BlockList(sitkImage, blockImageSize=(32, 32, 32), blockOverlappingSize=(0, 0, 0)):
    """
    :param sitkImage:
    :param blockImageSize:
    :param blockOverlappingSize:
    :return:
    """
    # check the input
    if sitkImage is None: return []
    if blockImageSize == sitkImage.GetSize(): return [sitkImage], (1, 1, 1)
    dim = sitkImage.GetDimension()
    if len(blockImageSize) != dim: return []
    if len(blockOverlappingSize) != dim: return []

    # clip the input and convert to numpy array
    np_sitkImageSize = np.array(sitkImage.GetSize(), dtype=np.int)
    np_blockImageSize = np.array(blockImageSize, dtype=np.int)
    np_blockOverlappingSize = np.array(blockOverlappingSize, dtype=np.int)

    np_blockImageSize = np_blockImageSize.clip(1, np_sitkImageSize)
    np_blockOverlappingSize = np_blockOverlappingSize.clip(0, np_blockImageSize)

    np_blockGridSize, np_blockGrid_rems = np.divmod(np_sitkImageSize, np_blockImageSize - np_blockOverlappingSize)
    np_blockGridSize[np_blockGrid_rems != 0] = np_blockGridSize[np_blockGrid_rems != 0] + 1
    np_sitkImageSize_new = np_blockGridSize * (np_blockImageSize - np_blockOverlappingSize) + np_blockOverlappingSize

    sitkImage_new = resample_sitkImage_centerAligned(sitkImage, tuple(np_sitkImageSize_new.tolist()), \
                                                     sitkImage.GetSpacing())

    # forming a np_xyz to go through [x, y, z] with sequence of x->y->z - 'F' order
    np_xyz = np.zeros(np_blockGridSize, dtype=np.int)
    np_xyz_flatten = np_xyz.flatten()
    for i in range(np_xyz_flatten.shape[0]):
        np_xyz_flatten[i] = i
    np_xyz = np_xyz_flatten.reshape(np_blockGridSize, order='F')

    # split in the order of np_xyz
    sitkImageBlocksList = []
    for i in range(np_xyz_flatten.shape[0]):
        np_xyz_idx = np.argwhere(np_xyz == i)[0]
        np_block_idx = (np_xyz_idx * (np_blockImageSize - np_blockOverlappingSize)).astype(np.int)
        np_tmp = np.stack([np_blockImageSize, np_sitkImageSize_new - np_block_idx], axis=1)
        np_final_blockImageSize = np.min(np_tmp, axis=1)

        final_blockImageSize = tuple(np_final_blockImageSize.tolist())
        final_block_idx = tuple(np_block_idx.tolist())
        sitkImageBlocksList.append(sitk.RegionOfInterest(sitkImage_new, size=final_blockImageSize, \
                                                         index=final_block_idx))

    return sitkImageBlocksList, tuple(np_blockGridSize.tolist())


# ---------------------------------------------------------------------------------
# composing the grid lists of blocks into a composed vol
# the spacing and size is used from the sitkImageGridList[0]
def composing_sitkImageBlocksList_into_SingleVol(sitkImageBlocksList, blockGridSize=(1, 1, 1),
                                                 blockOverlappingSize=(0, 0, 0)):
    """
    :param sitkImageBlocksList:
    :param blockGridSize:
    :param blockOverlappingSize:
    :return:
    """
    # check and clip the inputs -> convert to numpy array
    nb_blocksList = len(sitkImageBlocksList)
    dims = []
    for i in range(nb_blocksList):
        if sitkImageBlocksList[i] is None: return None
        dims.append(sitkImageBlocksList[i].GetDimension())
    if dims != [dims[0]] * nb_blocksList: return None

    np_blockImageSize = np.array(sitkImageBlocksList[0].GetSize(), dtype=np.int)
    np_blockGridSize = np.array(blockGridSize, dtype=np.int).clip(1, )
    np_blockOverlappingSize = np.array(blockOverlappingSize, dtype=np.int)
    np_blockOverlappingSize = np_blockOverlappingSize.clip(0, (np_blockImageSize / 2).astype(np.int))
    if nb_blocksList != np.prod(np_blockGridSize, axis=0): return None

    # forming a np_xyz to go through [x, y, z] with sequence of x->y->z -> 'F' order
    np_xyz = np.zeros(np_blockGridSize, dtype=np.int)
    np_xyz_flatten = np_xyz.flatten()
    for i in range(np_xyz_flatten.shape[0]):
        np_xyz_flatten[i] = i
    np_xyz = np_xyz_flatten.reshape(np_blockGridSize, order='F')

    # composing in the order of np_xyz
    sitkImageComposeList = []
    for i in range(nb_blocksList):
        i_np_blockImageSize = np.array(sitkImageBlocksList[i].GetSize(), dtype=np.int)
        np_lower_blockOverlappingPixel = (np_blockOverlappingSize / 2).astype(np.int)
        np_upper_blockOverlappingPixel = np_blockOverlappingSize - np_lower_blockOverlappingPixel
        idx_xyz = np.argwhere(np_xyz == i)[0]
        np_lower_blockOverlappingPixel[idx_xyz == 0] = 0
        np_upper_blockOverlappingPixel[idx_xyz == np_blockGridSize - 1] = 0
        np_roiSize = i_np_blockImageSize - np_lower_blockOverlappingPixel - np_upper_blockOverlappingPixel

        sitkImageROI = sitk.RegionOfInterest(sitkImageBlocksList[i], \
                                             size=tuple(np_roiSize.tolist()), \
                                             index=tuple(np_lower_blockOverlappingPixel.tolist()))
        sitkImageComposeList.append(sitkImageROI)

    # reset composed image with settings of 1st image in the list
    composed_sitkImage = sitk.Tile(sitkImageComposeList, blockGridSize)
    composed_sitkImage.SetDirection(sitkImageComposeList[0].GetDirection())
    composed_sitkImage.SetSpacing(sitkImageComposeList[0].GetSpacing())
    composed_sitkImage.SetOrigin(sitkImageComposeList[0].GetOrigin())
    return composed_sitkImage


# -----------------------------------------------------------------------------------------------
# resampling the vol and its label by the transform
def resample_sitkImageLabel_byTransform(image, label, transform, image_default_value, **kwargs):
    """
    :param image:
    :param label:
    :param transform:
    :param image_default_value:
    :return:
    """
    image_value = 0.0
    if image_default_value == 'min':
        image_value = float(np.ndarray.min(sitk.GetArrayFromImage(image)))
    elif image_default_value == 'zero':
        image_value = 0.0
    elif str(image_default_value).isnumeric():
        image_value = float(image_default_value)

    if image is None or label is None: return None
    if transform is None: return None

    """""
    plt.figure()
    plt.subplot(2, 2, 1)
    np_image = sitk.GetArrayFromImage(image)
    plt.imshow(np_image[0, ::])
    plt.subplot(2, 2, 2)
    np_label = sitk.GetArrayFromImage(label)
    plt.imshow(np_label[0, ::])
    spacing0 = image.GetSpacing()
    origin0 = image.GetOrigin()
    direction0 = image.GetDirection()
    # spacing1 = [spacing0[0], spacing0[1], 100*spacing0[2]]
    spacing1 = spacing0
    image = sitk.GetImageFromArray(np_image)
    image.SetSpacing(spacing1)
    image.SetOrigin(origin0)
    image.SetDirection(direction0)
    label = sitk.GetImageFromArray(np_label)
    label.SetSpacing(spacing1)
    label.SetOrigin(origin0)
    label.SetDirection(direction0)
    """""

    ref_image = image
    new_image = sitk.Resample(image, ref_image, transform, sitk.sitkNearestNeighbor, image_value,
                              image.GetPixelIDValue())
    new_label = sitk.Resample(label, ref_image, transform, sitk.sitkNearestNeighbor, 0.0, label.GetPixelIDValue())

    """""
    plt.subplot(2, 2, 3)
    np_new_image = sitk.GetArrayFromImage(new_image)
    plt.imshow(np_new_image[0, ::])
    plt.subplot(2, 2, 4)
    np_new_label = sitk.GetArrayFromImage(new_label)
    plt.imshow(np_new_label[0, ::])
    plt.show()
    new_image = sitk.GetImageFromArray(np_new_image)
    new_image.SetSpacing(spacing0)
    new_image.SetOrigin(origin0)
    new_image.SetDirection(direction0)
    new_label = sitk.GetImageFromArray(np_new_label)
    new_label.SetSpacing(spacing0)
    new_label.SetOrigin(origin0)
    new_label.SetDirection(direction0)
    """""

    additional_label_list = kwargs.get('additional_label_list')
    if additional_label_list is None: return new_image, new_label

    new_additional_label_list = []
    for additional_label in additional_label_list:
        new_additional_label_list.append(sitk.Resample(additional_label, ref_image, transform, \
                                                       sitk.sitkNearestNeighbor, 0.0, label.GetPixelIDValue()))
    return new_image, new_label, new_additional_label_list