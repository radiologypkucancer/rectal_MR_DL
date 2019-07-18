# ----------------------------------------------------------------------------------
# imports
import json, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# ----------------------------------------------------------------------------------
# get logger
from utils.logger import get_logger

logger = get_logger()

# ----------------------------------------------------------------------------------
def evalue_model(predict_file_list):
    with open(predict_file_list[0], 'r', encoding='utf-8') as f_json:
        predict_results_config = json.load(f_json)
    predict_results_temp = predict_results_config['predict']
    predict_results_list = [predict_results_temp.get(my_key) for my_key in predict_results_temp.keys()]
    predict_results = predict_results_list[0]
    lesion_type_number = 0
    lesion_type = []
    for key in predict_results.keys():
        lesion_gt_description = predict_results.get(key)['reference_class_description']
        if lesion_gt_description not in lesion_type:
            lesion_type_number += 1
            lesion_type.append(lesion_gt_description)

    if len(lesion_type) < 1:
        logger.warning('Reference class number is less than 1.')
    elif len(lesion_type) is not lesion_type_number:
        logger.warning('Reference class number is inconsistent with lesion type number.')

    lesion_type_number = len(lesion_type)
    confusion_matrix = [[0 for i in range(lesion_type_number)] for i in range(lesion_type_number)]
    total_lesion_num = 0
    for predict_file in predict_file_list:
        with open(predict_file, 'r', encoding='utf-8') as f_json:
            predict_results_config = json.load(f_json)
        predict_results_temp = predict_results_config['predict']
        predict_results_list = [predict_results_temp.get(my_key) for my_key in predict_results_temp.keys()]
        predict_results = predict_results_list[0]
        for key in predict_results.keys():
            total_lesion_num += 1
            lesion_gt_description = predict_results.get(key)['reference_class_description']
            lesion_predict_description = predict_results.get(key)['class_description'][0]
            gt_index = lesion_type.index(lesion_gt_description)
            predict_index = lesion_type.index(lesion_predict_description)
            confusion_matrix[gt_index][predict_index] += 1

    my_sum = sum(map(sum,confusion_matrix))
    if my_sum != total_lesion_num:
        logger.warning('confusion matrix is wrong cause matrix sum is inconsistent with total lesion num.')

    evaluation_results = {}
    evaluation_results['evaluation'] = {}
    total_right_lesion_num = 0
    for gt_lesion in lesion_type:
        gt_index = lesion_type.index(gt_lesion)
        evaluation_results['evaluation'][gt_lesion] = {}
        evaluation_results['evaluation'][gt_lesion][gt_lesion + ' num'] = sum(confusion_matrix[gt_index])
        evaluation_results['evaluation'][gt_lesion][gt_lesion + ' TP rate'] = \
            round(confusion_matrix[gt_index][gt_index] / sum(confusion_matrix[gt_index]), 3)
        total_right_lesion_num += confusion_matrix[gt_index][gt_index]
        for predict_lesion in lesion_type:
            predict_index = lesion_type.index(predict_lesion)
            evaluation_results['evaluation'][gt_lesion][predict_lesion] = confusion_matrix[gt_index][predict_index]
    evaluation_results['evaluation']['Total test num'] = my_sum
    evaluation_results['evaluation']['Total right prediction num'] = total_right_lesion_num
    evaluation_results['evaluation']['Total TP rate'] = round(total_right_lesion_num / my_sum, 3)
    return evaluation_results

def get_wrong_patient_ids(predict_file_list):
    with open(predict_file_list[0], 'r', encoding='utf-8') as f_json:
        predict_results_config = json.load(f_json)
    predict_results_temp = predict_results_config['predict']
    predict_results_list = [predict_results_temp.get(my_key) for my_key in predict_results_temp.keys()]
    predict_results = predict_results_list[0]
    lesion_type_number = 0
    lesion_type = []
    for key in predict_results.keys():
        lesion_gt_description = predict_results.get(key)['reference_class_description']
        if lesion_gt_description not in lesion_type:
            lesion_type_number += 1
            lesion_type.append(lesion_gt_description)

    if len(lesion_type) < 1:
        logger.warning('Reference class number is less than 1.')
    elif len(lesion_type) is not lesion_type_number:
        logger.warning('Reference class number is inconsistent with lesion type number.')

    lesion_type_number = len(lesion_type)
    confusion_matrix = [[0 for i in range(lesion_type_number)] for i in range(lesion_type_number)]
    confusion_matrix_id = [[[] for i in range(lesion_type_number)] for i in range(lesion_type_number)]
    total_lesion_num = 0
    for predict_file in predict_file_list:
        with open(predict_file, 'r', encoding='utf-8') as f_json:
            predict_results_config = json.load(f_json)
        predict_results_temp = predict_results_config['predict']
        predict_results_list = [predict_results_temp.get(my_key) for my_key in predict_results_temp.keys()]
        predict_results = predict_results_list[0]
        for key in predict_results.keys():
            total_lesion_num += 1
            lesion_gt_description = predict_results.get(key)['reference_class_description']
            lesion_predict_description = predict_results.get(key)['class_description'][0]
            lesion_predict_ims = predict_results.get(key)['images_data_niix'][0]
            lesion_predict_vec = lesion_predict_ims.split(os.path.sep)
            lesion_predict_id = lesion_predict_vec[-4] + '_' + lesion_predict_vec[-3]
            gt_index = lesion_type.index(lesion_gt_description)
            predict_index = lesion_type.index(lesion_predict_description)
            confusion_matrix[gt_index][predict_index] += 1
            confusion_matrix_id[gt_index][predict_index].append(lesion_predict_id)

    my_sum = sum(map(sum,confusion_matrix))
    if my_sum != total_lesion_num:
        logger.warning('confusion matrix is wrong cause matrix sum is inconsistent with total lesion num.')

    evaluation_results = {}
    evaluation_results['evaluation'] = {}
    evaluation_results['evaluation']['Total test number'] = my_sum
    for gt_lesion in lesion_type:
        gt_index = lesion_type.index(gt_lesion)
        evaluation_results['evaluation'][gt_lesion] = {}
        evaluation_results['evaluation'][gt_lesion][gt_lesion + ' number'] = sum(confusion_matrix[gt_index])
        evaluation_results['evaluation'][gt_lesion][gt_lesion + ' TP rate'] = \
            round(confusion_matrix[gt_index][gt_index] / sum(confusion_matrix[gt_index]),3)
        for predict_lesion in lesion_type:
            predict_index = lesion_type.index(predict_lesion)
            evaluation_results['evaluation'][gt_lesion][predict_lesion] = {}
            evaluation_results['evaluation'][gt_lesion][predict_lesion]['number'] = confusion_matrix[gt_index][predict_index]
            evaluation_results['evaluation'][gt_lesion][predict_lesion]['patient ids'] = confusion_matrix_id[gt_index][predict_index]
    return evaluation_results

def convert_int_to_onehot(class_id,num_classes, output_datatype='float'):
    assert isinstance(class_id, int)
    result = np.zeros(shape=(num_classes))
    result[class_id] = 1.0
    if output_datatype == 'float': result = result.astype(float)
    elif output_datatype == 'int': result = result.astype(int)
    return result

def draw_ROC(predict_file_lists):
    lesion_num = 0
    n_classes = 2
    classx_dict= {
        "ab_yes_downstage": "0",
        "no_downstage": "1"
    }

    for predict_file in predict_file_lists[0]:
        with open(predict_file, 'r', encoding='utf-8') as f_json:
            predict_results_config = json.load(f_json)
        predict_results_temp = predict_results_config['predict']
        predict_results_list = [predict_results_temp.get(my_key) for my_key in predict_results_temp.keys()]
        predict_results = predict_results_list[0]
        for key in predict_results.keys():
            lesion_num += 1

    y_test = np.zeros(shape=[lesion_num, n_classes])
    y_score = np.zeros(shape=[lesion_num, n_classes])

    for predict_file_list in predict_file_lists:
        lesion_num_ind = -1
        for predict_file in predict_file_list:
            with open(predict_file, 'r', encoding='utf-8') as f_json:
                predict_results_config = json.load(f_json)
            predict_results_temp = predict_results_config['predict']
            predict_results_list = [predict_results_temp.get(my_key) for my_key in predict_results_temp.keys()]
            predict_results = predict_results_list[0]
            for key in predict_results.keys():
                lesion_num_ind += 1
                lesion_gt_description = predict_results.get(key)['reference_class_description']
                label_onehot = convert_int_to_onehot(class_id = int(classx_dict[lesion_gt_description]) , num_classes = n_classes, output_datatype='int')
                y_test[lesion_num_ind, :] = label_onehot
                predict_onehot = predict_results.get(key)['class_description'][1]
                y_score[lesion_num_ind, :] = predict_onehot

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()
    colors = cycle(['aqua', 'darkorange'])
    lw = 1.5
    classx = ["ab_yes_downstage","no_downstage"]

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of  {0} (area = {1:0.2f})'
                       ''.format(classx[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

#-----------------------------------------------------------------------------------
#
if __name__ == '__main__':
    # evaluate a single model
    root1 = "D:\\data\\ailab\\Rectal_MR\\models\\BJCH_rectal_MR_full_lesion_modle1_T_degrade_g3_extension"
    file_1 = os.path.join(root1, 'cross_group3_predict_config.json')
    #
    evaluation_results1 = evalue_model([ file_1])
    evaluation_results2 = get_wrong_patient_ids([ file_1])
    draw_ROC([[ file_1]])
    #
    evaluation_results_file1 = os.path.join(root1, 'lession_classfication_predict_evaluate_results_1.json')
    evaluation_results_file2 = os.path.join(root1, 'lession_classfication_predict_evaluate_results_2.json')
    #
    with open(evaluation_results_file1, 'w', encoding='utf-8') as f_json:
        json.dump(evaluation_results1, f_json, indent=4)

    with open(evaluation_results_file2, 'w', encoding='utf-8') as f_json:
        json.dump(evaluation_results2, f_json, indent=4)
