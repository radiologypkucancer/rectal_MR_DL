# ----------------------------------------------------------------------------------
# imports
from networks.DCPCNN import DCPCNN
from data_preparation.convert_niix_to_npz import DataType, get_np_data_as_groupids, load_np_datagroups

import os

# ----------------------------------------------------------------------------------
# get logger
from utils.logger import get_logger

logger = get_logger()

#-----------------------------------------------------------------------------------
# GPU setting
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#-----------------------------------------------------------------------------------
#
if __name__ == '__main__':
    # model root
    model_root = 'D:\\data\\ailab\\Rectal_MR\\models\\BJCH_rectal_MR_model_c_test'
    model_name = 'BJCH_rectal_MR_model_c_T_downstage_prediction'

    # model settings
    niix_size = (16, 128, 128)
    nb_classx = 2
    conv_depth = 5
    filters = 16
    ClassifyNet = DCPCNN(input_img_size= niix_size, nb_classx=nb_classx, conv_depth=conv_depth, filters=filters)

    # initialize the model
    ClassifyNet.initialize_model()
    model_png_file = os.path.join(model_root, model_name + '.png')
    if not os.path.exists(model_png_file): ClassifyNet.save_net_summary_to_file(model_png_file)

    # load training and validation npz data
    validation_data_group_ids = get_np_data_as_groupids(model_root=model_root, data_type=DataType.VALIDATION)
    if validation_data_group_ids == []:
        logger.warning('there is no available validation nets data.')
    validation_vols, validation_targets = load_np_datagroups(model_root, validation_data_group_ids, data_type=DataType.VALIDATION)
    training_data_group_ids = get_np_data_as_groupids(model_root=model_root, data_type=DataType.TRAINING)
    if training_data_group_ids == []: logger.warning('there is no available training nets data.')
    # regroup the data
    training_data_combined_groupids_list = []
    combined_groupids = []
    nb_combined_groups = 1
    for i, group_id in enumerate(training_data_group_ids):
        if i % nb_combined_groups == 0:
            combined_groupids = [group_id]
        else:
            combined_groupids.append(group_id)
        if i % nb_combined_groups == nb_combined_groups - 1 or i == len(training_data_group_ids) - 1:
            training_data_combined_groupids_list.append(combined_groupids)

    # traing the model using regrouped data
    # load weights and then training the model with iterations
    weights = os.path.join(model_root, 'weights.hdf5')
    for combined_groupids in training_data_combined_groupids_list:
        train_vols, train_targets = load_np_datagroups(model_root, combined_groupids, data_type=DataType.TRAINING)
        if train_vols is None or train_targets is None: continue
        ClassifyNet.train(input_netsdata=[train_vols[:, 0],train_vols[:, 1], \
                                          train_vols[:, 2],train_vols[:, 3], \
                                          train_vols[:, 4],train_vols[:, 5], \
                                          train_vols[:, 6],train_vols[:, 7]],
                                    classx_netsdata=train_targets,
                                    batch_size=25,
                                    epochs=2,
                                    validation_input_netsdata=[validation_vols[:, 0],validation_vols[:, 1], \
                                                               validation_vols[:, 2],validation_vols[:, 3], \
                                                               validation_vols[:, 4],validation_vols[:, 5], \
                                                               validation_vols[:, 6],validation_vols[:, 7]],
                                    validation_classx_netsdata=validation_targets)
        ClassifyNet.save_weights(weights)

