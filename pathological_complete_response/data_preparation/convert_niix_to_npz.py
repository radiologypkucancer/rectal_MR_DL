# ----------------------------------------------------------------------------------
# imports
import numpy as np
import SimpleITK as sitk
import json, os
from tqdm import *
from glob import glob
from enum import Enum

from data_preparation.common_img_processing.resampling import resample_sitkImage_by_extension

# ----------------------------------------------------------------------------------
# get logger
from utils.logger import get_logger

logger = get_logger()
# ----------------------------------------------------------------------------------
#
class DataType(Enum):
    TRAINING = 0
    VALIDATION = 1

#-------------------------------------------------------------------------------------
#
class AnnotationDict2Nets:
    # initialization
    def __init__(self):
        #
        # {class_id: description :
        #            one_hot :
        # }
        #
        self.__dict_nets_data = {}
        self.__num_classes = 0
    #-------------------------------------------------------------------------------------
    #
    def init_annotation_by_dict(self, anno_dict):
        """
        :param anno_dict: {class_id(int); class_description(str)}
        :return:
        """
        # check the input
        if anno_dict.__class__  is not dict: return False
        self.__dict_nets_data = {}
        self.__num_classes = max(anno_dict.keys()) + 1
        for key in anno_dict.keys():
            if self._check_annotation_pairs(key, anno_dict.get(key)):
                key_dict = {}
                key_dict['description'] = anno_dict.get(key)
                key_dict['one_hot'] = self._convert_int_to_onehot(key, output_datatype='float')
                self.__dict_nets_data[key] = key_dict
        return True
    #-------------------------------------------------------------------------------------
    #
    def init_annotation_by_list(self, class_ids, class_descriptions):
        """
        :param class_ids:
        :param class_descriptions:
        :return:
        """
        # check inputs
        if class_ids.__class__ is not list or class_descriptions is not dict: return False
        if len(class_ids) != len(class_descriptions): return False
        self.__dict_nets_data = {}
        self.__num_classes = max(class_ids) + 1
        for class_id, class_description in zip(class_ids, class_descriptions):
            if not self._check_annotation_pairs(class_id, class_description): continue
            key_dict = {}
            key_dict['description'] = class_description
            key_dict['one_hot'] = self._convert_int_to_onehot(class_id, output_datatype='float')
            self.__dict_nets_data[class_id] = class_description
        return True
    #-------------------------------------------------------------------------------------
    #
    def get_anno_dict(self):
        """
        :return:
        """
        return self.__dict_nets_data
    # -------------------------------------------------------------------------------------
    #
    def get_num_classes(self):
        """
        :return:
        """
        return self.__num_classes
    #-------------------------------------------------------------------------------------
    #
    def get_class_description(self, class_id):
        """
        :param class_id:
        :return: str
        """
        key_dict = self.__dict_nets_data.get(class_id)
        if key_dict.__class__ is not dict: return None
        return key_dict.get('description')
    #-------------------------------------------------------------------------------------
    #
    def get_class_code(self, class_id):
        """
        :param class_id:
        :return: 1d numpy array
        """
        key_dict = self.__dict_nets_data.get(class_id)
        if key_dict.__class__ is not dict: return None
        return key_dict.get('one_hot')
    #-------------------------------------------------------------------------------------
    #
    def get_class_description_encode(self, class_id):
        """
        :param class_id:
        :return: dict
        """
        key_dict = self.__dict_nets_data.get(class_id)
        if key_dict.__class__ is not dict: return None
        return key_dict.get('description'), key_dict.get('one_hot')
    #-------------------------------------------------------------------------------------
    #
    def get_class_id(self, **kwargs):
        """
        :param kwargs:
        :return:
        """
        # check input
        one_hot = kwargs.get('class_code')
        description = kwargs.get('class_description')
        if one_hot is not None: assert isinstance(one_hot, np.ndarray)
        if description is not None: assert isinstance(description, str)
        class_id = 0
        for key in self.__dict_nets_data.keys():
            one_hot_np = self.__dict_nets_data[key].get('one_hot')
            description_str = self.__dict_nets_data[key].get('description')
            if one_hot is not None and self._is_same_one_hot_code(one_hot, one_hot_np):
                class_id = key
            elif description is not None and description == description_str: class_id = key
        return class_id
    #-------------------------------------------------------------------------------------
    #
    def _convert_int_to_onehot(self, class_id, output_datatype='float'):
        """
        Converts an input integer into an output
        1-D array of one-hot vector,
        """
        assert isinstance(class_id, int)

        result = np.zeros(shape=(self.__num_classes))
        result[class_id] = 1.0
        if output_datatype == 'float': result = result.astype(float)
        elif output_datatype == 'int': result = result.astype(int)
        return result
    #-------------------------------------------------------------------------------------
    #
    def _convert_vector_to_onehot(self, vector, output_datatype='float'):
        """
        Converts an input 1-D vector of integers into an output
        2-D array of one-hot vectors, where an i'th input value
        of j will set a '1' in the i'th row, j'th column of the
        output array.
        Example:
            v = np.array((1, 0, 4))
            one_hot_v = convertToOneHot(v)
            print one_hot_v

            [[0 1 0 0 0]
             [1 0 0 0 0]
             [0 0 0 0 1]]
        """
        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0

        result = np.zeros(shape=(len(vector), self.__num_classes))
        result[np.arange(len(vector)), vector] = 1.0
        if output_datatype == 'float': result = result.astype(float)
        elif output_datatype == 'int': result = result.astype(int)
        return result
    #-------------------------------------------------------------------------------------
    #
    def _check_annotation_pairs(self, class_id, class_description):
        """
        :param class_id:
        :param class_description:
        :return:
        """
        return class_id.__class__ is int and class_description.__class__ is str
    #-------------------------------------------------------------------------------------
    #
    def _is_same_one_hot_code(self, one_hot_1, one_hot_2):
        """
        :param one_hot_1:
        :param one_hot_2:
        :return:
        """
        assert isinstance(one_hot_1, np.ndarray)
        assert isinstance(one_hot_2, np.ndarray)

        ok = one_hot_1 == one_hot_2
        return ok.all()

# ------------------------------------------------------------------------------
# normalize Rectal MR from [cmin cmax] to [-1.0 1.0]
def normalize_Rectal_MR_to_unit(sitkImage):
    sitkImage = sitk.Cast(sitkImage, sitk.sitkFloat64)
    img_array = sitk.GetArrayFromImage(sitkImage)
    cmin = np.min(img_array)
    cmax = np.max(img_array)
    cmiddle1 = (cmax + cmin) / 2
    cmiddle2 = (cmax - cmin) / 2
    img_array = (img_array - cmiddle1) / (cmiddle2 + 0.0001)
    img_array[img_array < -1.0] = -1.0
    img_array[img_array > 1.0] = 1.0
    normalized_sitkImage = sitk.GetImageFromArray(img_array)
    normalized_sitkImage.SetOrigin(sitkImage.GetOrigin())
    normalized_sitkImage.SetSpacing(sitkImage.GetSpacing())
    normalized_sitkImage.SetDirection(sitkImage.GetDirection())
    return normalized_sitkImage

# -------------------------------------------------------------------------------
#
def convert_niix2nets(sitkImageList):
    """
    :param sitkImageList:
    :param sitkLabelList:
    :return: np_nets: [sample / nb_blocks, channels, z, y, x] for channel_axis = 1
                      [sample / nb_blocks, z, y, x, channels] for channel_axis = -1
    """

    # -----------------------------------------------------------------------------
    # sitkImageList
    # -----------------------------------------------------------------------------
    if len(sitkImageList) < 1: return False

    for sitkImage in sitkImageList:
        if sitkImage is None: return False

    # convert all sitkImageList into float32 type
    new_sitkImageList = []
    for sitkImage in sitkImageList: new_sitkImageList.append(sitk.Cast(sitkImage, sitk.sitkFloat32))
    sitkImageList = new_sitkImageList

    np_images_array_list = []
    for sitkImage_block in sitkImageList:
        if sitkImage_block.GetNumberOfComponentsPerPixel() == 1:
            # np_images_array = np.array([sitk.GetArrayFromImage(sitkImage_block)])
            np_images_array = np.moveaxis(sitk.GetArrayFromImage(sitkImage_block), -1, 0)
        else:
            np_images_array = np.moveaxis(sitk.GetArrayFromImage(sitkImage_block), -1, 0)
        np_images_array_list.append(np_images_array)
    np_images_nets = np.array(np_images_array_list)

    # channel_axis: channel_last mode
    # default channel_axis = 1
    channel_axis = -1
    if channel_axis == -1: np_images_nets = np.moveaxis(np_images_nets, 1, -1)

    return np_images_nets

# ----------------------------------------------------------------------------------
#
def convert_niix_into_nets(sitkImageList, niix_size):
    # --------------------------------------------------------------------------
    # processing of niix - resampling / normalization
    # --------------------------------------------------------------------------
    for i_sitkImage, each_sitkImage in enumerate(sitkImageList):
        if i_sitkImage == 0:
            tmp_checks = [each_sitkImage.GetSize()]
            if tmp_checks != [niix_size]:
                sitkImageList[i_sitkImage] = resample_sitkImage_by_extension(each_sitkImage, niix_size)
        else:
            tmp_checks = [each_sitkImage.GetSize()]
            if tmp_checks != [niix_size]:
                sitkImageList[i_sitkImage] = resample_sitkImage_by_extension(each_sitkImage, niix_size)
                img_array = sitk.GetArrayFromImage(sitkImageList[i_sitkImage])
                sitkImageList[i_sitkImage] = sitk.GetImageFromArray(img_array)
            sitkImageList[i_sitkImage].SetOrigin(sitkImageList[0].GetOrigin())
            sitkImageList[i_sitkImage].SetSpacing(sitkImageList[0].GetSpacing())
            sitkImageList[i_sitkImage].SetDirection(sitkImageList[0].GetDirection())

        sitkImageList[i_sitkImage] = normalize_Rectal_MR_to_unit(sitkImageList[i_sitkImage])

    nets_data = convert_niix2nets(sitkImageList)

    return  nets_data

# ----------------------------------------------------------------------------
#
def convert_classdescription_into_nets(class_description, classx_nets):
    """
    :param class_description:
    :param classx_nets:
    :return: (samples, one-hot-code)
    """
    class_id = classx_nets.get_class_id(class_description=class_description)
    class_np_code = classx_nets.get_class_code(class_id=class_id)
    # add sample axis into class_np_code
    class_np = np.array([class_np_code])
    return class_np

# ----------------------------------------------------------------------------------
#
def get_np_data_prefix(data_type=DataType.TRAINING):
    """
    :return:
    """
    if data_type == DataType.TRAINING:
        images_np_prefix = 'training_images_'
        targets_np_prefix = 'training_targets_'
    elif data_type == DataType.VALIDATION:
        images_np_prefix = 'validation_images_'
        targets_np_prefix = 'validation_targets_'
    return images_np_prefix, targets_np_prefix

# ----------------------------------------------------------------------------------
#
def get_np_data_as_groupids(model_root, data_type=DataType.TRAINING):
    """
    :param model_root:
    :param data_type:
    :return:
    """
    groupids = []
    # read info from stored files
    images_prefix, targets_prefix = get_np_data_prefix(data_type=data_type)
    images_npzs = glob(os.path.join(model_root, images_prefix + '*.npz'))
    targets_npzs = glob(os.path.join(model_root, targets_prefix + '*.npz'))
    if len(images_npzs) != len(targets_npzs): return groupids
    if len(images_npzs) < 1 or len(targets_npzs) < 1: return groupids

    # get the group ids
    images_groupids = []
    targets_groupids = []
    for images_npz in images_npzs: images_groupids.append(
        int(os.path.basename(images_npz[:-4]).replace(images_prefix, '')))
    for targets_npz in targets_npzs: targets_groupids.append(
        int(os.path.basename(targets_npz[:-4]).replace(targets_prefix, '')))
    if images_groupids != targets_groupids: return images_groupids

    return images_groupids

# ----------------------------------------------------------------------------------
#
def get_np_data_filename( i_subgroup, data_type=DataType.TRAINING):
    """
    :param i_subgroup:
    :return: images_00000, targets_00000
    """
    images_prefix, targets_prefix = get_np_data_prefix(data_type=data_type)
    images_np_name = images_prefix + '{:05d}.npz'.format(i_subgroup)
    targets_np_name = targets_prefix + '{:05d}.npz'.format(i_subgroup)
    return images_np_name, targets_np_name

# ----------------------------------------------------------------------------------
#
def save_np_data(model_root, np_images, np_targets, data_type=DataType.TRAINING):
    """
    :param np_images:
    :param group_name:
    :param files_list_dict:
    :return:
    """
    # groups starting from 0 then appending to previous groupids
    # counting groups
    group_ids = get_np_data_as_groupids(model_root, data_type=data_type)
    if group_ids == []: group_ids = [0]
    group_ids = sorted(group_ids)
    group_ids_expand = group_ids + [group_ids[-1] + 1]
    group_ids_left = [x for x in group_ids_expand if x not in group_ids]
    images_np, targets_np = get_np_data_filename(group_ids_left[0], data_type=data_type)
    images_np_file = os.path.join(model_root, images_np)
    targets_np_file = os.path.join(model_root, targets_np)
    np.savez_compressed(images_np_file, np_images)
    np.savez_compressed(targets_np_file, np_targets)
    return

# # --------------------------------------------------------------------------------------------
#
def add_np_data(grouped_data_sets, classx_dict, niix_size, model_root, data_type=DataType.TRAINING):
    """
    :param grouped_data_sets:
    :param classx_dict:
    :param niix_size:
    :param model_root:
    :param data_type:
    :return:
    """
    # check grouped_data_sets
    if grouped_data_sets.__class__ is not dict: logger.info('invalid data config file.')

    # settings
    classx_nets = AnnotationDict2Nets()
    # classx_dict is configured in classx_net
    classx_nets.init_annotation_by_dict(classx_dict)

    group_names = grouped_data_sets.keys()
    for group_name in group_names:
        filesListDict = grouped_data_sets.get(group_name)
        if filesListDict.__class__ is not dict: continue
        logger.info('converting %s into nets data from niix input...', group_name)
        grouped_np_images = None
        grouped_np_classes = None
        for sample in tqdm(filesListDict.keys()):
            each_trainingSets = filesListDict.get(sample)

            # list images_data_niix in each dataset
            images_data_niix = each_trainingSets.get('images_data_niix')
            if images_data_niix is None: continue
            sitkImageList = []
            for each_niix_file in images_data_niix:
                sitkImageList.append(sitk.ReadImage(each_niix_file))
            # converts to nets data into np format as (samples, channels, slices, height, width)
            eachset_nets_data = convert_niix_into_nets(sitkImageList, niix_size)

            # list labels_data_niix in each dataset
            class_description = each_trainingSets.get('class_description')
            class_nets_data = convert_classdescription_into_nets(class_description=class_description, classx_nets=classx_nets)

            # concatenate the np_nets as samples
            if grouped_np_images is None:
                grouped_np_images = [eachset_nets_data]
            # sample axis is default to 0
            else:
                grouped_np_images = np.concatenate([grouped_np_images, [eachset_nets_data]], axis=0)

            if grouped_np_classes is None:
                grouped_np_classes = class_nets_data
            # sample axis is default to 0
            else:
                grouped_np_classes = np.concatenate([grouped_np_classes, class_nets_data], axis=0)

        save_np_data(model_root, grouped_np_images, grouped_np_classes, data_type=data_type)

# # --------------------------------------------------------------------------------------------
#
def load_np_array_from_npz(npz_file):
    npz = np.load(npz_file)
    np_arrs = []
    for np_file in npz.keys(): np_arrs.append(npz[np_file])
    return np_arrs

# # --------------------------------------------------------------------------------------------
#
def load_np_datagroups(model_root, groupids, data_type=DataType.VALIDATION):
    """
    :param groupids:
    :param data_type:
    :return:
    """
    np_images_list = []
    np_clinical_information_list = []
    np_targets_list = []
    for groupid in groupids:
        images_npz_filename, targets_npz_filename = get_np_data_filename(groupid, data_type=data_type)
        images_npz_file = os.path.join(model_root, images_npz_filename)
        targets_npz_file = os.path.join(model_root, targets_npz_filename)
        if not os.path.exists(images_npz_file) or not os.path.exists(targets_npz_file): continue
        for np_images, np_targets in zip(load_np_array_from_npz(images_npz_file),
                                         load_np_array_from_npz(targets_npz_file)):
            np_images_list.append(np_images.astype(np.float32))
            np_targets_list.append(np_targets.astype(np.float32))
    if np_images_list == []  or np_targets_list == []: return None, None
    np_images = np.concatenate(np_images_list, axis=0)
    np_targets = np.concatenate(np_targets_list, axis=0)
    return np_images, np_targets
# # --------------------------------------------------------------------------------------------
#
def load_training_validation_npz_data(model_root, nb_combined_groups=10, data_type=DataType.TRAINING):
    """
    :param model_root:
    :param data_type:
    :return:
    """
    validation_data_group_ids = get_np_data_as_groupids(data_type=DataType.VALIDATION)
    validation_vols, validation_targets = load_np_datagroups(validation_data_group_ids, data_type=DataType.VALIDATION)

    training_data_group_ids = get_np_data_as_groupids(model_root=model_root, data_type=data_type)
    if training_data_group_ids == []:
        logger.warning('there is no available training nets data.')
        return False

    # regroup the data
    training_data_combined_groupids_list = []
    combined_groupids = []
    for i, group_id in enumerate(training_data_group_ids):
        if i % nb_combined_groups == 0:
            combined_groupids = [group_id]
        else:
            combined_groupids.append(group_id)
        if i % nb_combined_groups == nb_combined_groups - 1 or i == len(training_data_group_ids) - 1:
            training_data_combined_groupids_list.append(combined_groupids)

    for combined_groupids in training_data_combined_groupids_list:
        train_vols, train_targets = load_np_datagroups(combined_groupids,data_type=DataType.TRAINING)


