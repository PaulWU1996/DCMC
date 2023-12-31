import numpy as np

def gt_preprocess(config, gt_obj):
    """
        Preprocess the data (especially for ground truth data)
    """

    # the data will be processed in the range of 0-1
    processed_data = np.zeros([config['settings']['duration'], config['settings']['target_num_max'], 2]) # shape [duration, target_num_max, 2]
    for i in range(config['settings']['duration']):
        gt_step = gt_obj[str(i)]
        for j in range(config['settings']['target_num_max']):
            if j < len(gt_step):
                processed_data[i][j] = [gt_step[j].x/config['settings']['scenario_x'], gt_step[j].y/config['settings']['scenario_y']]
            else:
                # should add custom exception here but no need at this moment
                processed_data[i][j] = [-1, -1]

    # the data will be expanded to match the source_num
    newarray = np.zeros([config['settings']['source_num'], config['settings']['duration'], config['settings']['target_num_max'], 2]) # shape [source_num, duration, target_num_max, 2]    
    for i in range(config['settings']['source_num']):
        newarray[i] = processed_data
    processed_data = newarray.transpose(1, 0, 2, 3) # shape [duration, source_num, target_num_max, 2]
    return processed_data

def measurement_preprocess(config, measurement_obj, flag='origin'):
    """
        Preprocess the data (especially for measurement data)
        flag: origin or feat (feat will be processed in the range of 0-1)
    """
    processed_data = np.zeros([config['settings']['duration'], config['settings']['source_num'], config['settings']['target_num_max'], 2]) # shape [duration, source_num, target_num_max, 2]
    for i in range(config['settings']['duration']):
        measurement_step = measurement_obj[str(i)]
        for j in range(config['settings']['source_num']):
            for k in range(config['settings']['target_num_max']):
                if k < len(measurement_step[j]):
                    if flag == 'origin':
                        processed_data[i][j][k] = [measurement_step[j][k].x, measurement_step[j][k].y]
                    elif flag == 'feat':
                        processed_data[i][j][k] = [measurement_step[j][k].x/config['settings']['scenario_x'], measurement_step[j][k].y/config['settings']['scenario_y']]
                else:
                    # should add custom exception here but no need at this moment
                    processed_data[i][j][k] = [-1, -1]
    return processed_data