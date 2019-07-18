# ----------------------------------------------------------------------------------
# imports
from networks.DCPCNN import DCPCNN
from data_preparation.convert_niix_to_npz import AnnotationDict2Nets, convert_niix_into_nets

import os, json
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np

# ----------------------------------------------------------------------------------
# get logger
from utils.logger import get_logger

logger = get_logger()

#-----------------------------------------------------------------------------------
# GPU setting
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#-----------------------------------------------------------------------------------
#
def get_classx_id(np_code, classx_nets):
    np_code[np_code == np_code.max()] = 1.0
    np_code[np_code < np_code.max()] = 0.0
    return classx_nets.get_class_id(class_code=np_code)

def get_classx_description(class_id, classx_nets):
    return classx_nets.get_class_description(class_id=class_id)

# inner fucntion to handle the prediction of sitkImageList
def handle_niix_predict(sitkImageList, niix_size, classx_nets, classify_net, **kwargs):
    """
    :param sitkImageList:
    :return:
    """
    np_niix_size = np.array(niix_size)
    np_niix_size = np_niix_size[::-1]
    niix_size = np_niix_size.tolist()
    np_images_list = convert_niix_into_nets(sitkImageList, niix_size)
    input_netsdata_np = np.expand_dims(np_images_list, axis=0)
    input_netsdata = [input_netsdata_np[:, 0],input_netsdata_np[:, 1],
                      input_netsdata_np[:, 2],input_netsdata_np[:, 3],
                      input_netsdata_np[:, 4],input_netsdata_np[:, 5],
                      input_netsdata_np[:, 6],input_netsdata_np[:, 7]
                      ]
    np_classify_nets_data = classify_net.predict(input_netsdata=input_netsdata)
    results = {}
    results['one_hot'] = np_classify_nets_data[0].tolist()
    one_hot = np_classify_nets_data[0].tolist()
    class_id = get_classx_id(np_classify_nets_data[0], classx_nets)
    results['id'] = class_id
    results['description'] = []
    results['description'].append(get_classx_description(class_id, classx_nets))
    results['description'].append(one_hot)
    return results

#-----------------------------------------------------------------------------------
#
if __name__ == '__main__':
    # model root
    model_root = 'D:\\data\\ailab\\Rectal_MR\\models\\BJCH_rectal_MR_full_lesion_modle1_T_degrade_g3_extension'

    # model settings
    niix_size = (16, 128, 128)
    nb_classx = 2
    conv_depth = 5
    filters = 16
    ClassifyNet = DCPCNN(input_img_size= niix_size, nb_classx=nb_classx, conv_depth=conv_depth, filters=filters)

    # settings
    classx_nets = AnnotationDict2Nets()
    # classx_dict is configured in classx_net
    classx_dict = {
        "0": "ab_yes_downstage",
        "1": "no_downstage"
    }
    # convert key from str into int
    new_classx_dict = {}
    for key in classx_dict.keys():
        new_classx_dict[int(key)] = classx_dict.get(key)
    classx_nets.init_annotation_by_dict(new_classx_dict)

    # initialize the model
    ClassifyNet.initialize_model()

    # load weights
    weights = os.path.join(model_root, 'weights.hdf5')
    ClassifyNet.load_weights(weights)

    # load test data
    predict_config_root = "D:\\data\\sunyingshi\\cross_grouped_data_for_3kinds_label_Kapp_Dapp_full_lesion_0410\\model_configs\\model8_Tdegrade_abyes_no_with_blog1000_data"
    lesion_classification_predict_configfile = os.path.join(predict_config_root,
        'cross_group3_predict_config.json')

    with open(lesion_classification_predict_configfile, 'r', encoding='utf-8') as f_json:
        predict_config = json.load(f_json)

    # prediction
    data_images_nb = 8

    predict_groups = predict_config.get('predict')
    if predict_groups.__class__ is not dict: logger.warning('invalid predict config.')

    for predict_group_name in tqdm(predict_groups.keys()):
        predict_group = predict_groups.get(predict_group_name)
        for sample_name in tqdm(predict_group.keys()):
            sample = predict_group.get(sample_name)
            niix_images_files = sample.get('images_data_niix')
            sitkImages = []
            for images in niix_images_files:
                sitkImages.append(sitk.ReadImage(images))
            if len(sitkImages) != data_images_nb:
                logger.warning('number of input images is not consistent with modeling for %s of %s',
                               sample_name, predict_group_name)
                continue
            classify_result = handle_niix_predict(sitkImages,  niix_size, classx_nets, ClassifyNet)
            predict_config['predict'][predict_group_name][sample_name]['class_description'] = \
                classify_result.get('description')
    predict_results_dict = predict_config
    lesion_classification_predict_results_file = os.path.join(model_root, \
                                                              'cross_group3_predict_config.json')
    with open(lesion_classification_predict_results_file, 'w', encoding='utf-8') as f_json:
        json.dump(predict_results_dict, f_json, indent=4)






