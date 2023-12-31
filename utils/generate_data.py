# from .utils.utlis import *
from utils.utils import *
import os
import pickle
import numpy as np

class Generator():
    def __init__(self, config):
        #TODO: optimize the code with the
        self.config = config

        self.scenario = State(config['settings']['scenario_x'], config['settings']['scenario_x'])
        self.target_num_max = config['settings']['target_num_max']#config.target_num
        self.duration = config['settings']['duration']#config.duration
        self.source_num = config['settings']['source_num']#config.source_num
        self.noise_std = config['settings']['measurement_noise']#config.noise_std
        self.box_size = config['settings']['box_size']#config.box_size
        self.data_dir = config['filesystem']['root']+config['filesystem']['data_dir']
        self.feat_dir = config['filesystem']['root']+config['filesystem']['feat_dir']

    def _generate_gt_step(self, scenario:State, target_num:int): #TODO: optimize the code with self
        """
            Generate ground truth (x, y) in a 2D map for each step
        """
        gt = []
        for i in range(target_num):
            x = np.random.randint(0, scenario.x) 
            y = np.random.randint(0, scenario.y)
            gt.append(State(x,y))
        return gt

    def _generate_measurement_step(self, gts:list, target_num:int, source_num:int, noise_std:int):
        """
            Generate measurement (x, y) in a 2D map for each souruce in one step

            Return:
                measurement: a list of list of State over the sources as [[source1],[source2],...,[sourceN]] where source1 = [measure1, measure2, ..., measureN]
        """
        measurement = []
        for n in range(source_num):
            measurement_per_source = []
            for i in range(target_num):
                gt = gts[i]
                x = int(gt.x + noise_std * np.random.randn())
                y = int(gt.y + noise_std * np.random.randn())
                while x < 0 or x > self.scenario.x or y < 0 or y > self.scenario.y:
                    x = int(gt.x + noise_std * np.random.randn())
                    y = int(gt.y + noise_std * np.random.randn())
                measurement_per_source.append(State(x,y,self.box_size,self.box_size))
            measurement.append(measurement_per_source)
        return measurement

    def _save_pickle(self, filepath:str, data:list, dtype:str='GT'):
        """
            Save the data as pickle file
        """
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)


    def generate_data(self):
        """
            Generate ground truth (x, y) and measurement for each source node in a 2D map for the whole duration
            Data structure:
                timestep:
                gt: a list of list of State over the duration as [[step1],[step2],...,[stepN]] where step1 = [gt1, gt2, ..., gtN]
                measurement: a list of list of State over the duration as [[step1],[step2],...,[stepN]] where step1 = [measure1, measure2, ..., measureN]
                {'timestep:int': {'GT': gt, 'Measurement': measurement}}
        """
        #TODO: optimize the code with self
        ngts = dict()
        nmeasurements = dict()
        for i in range(self.duration):
            gt = self._generate_gt_step(self.scenario, self.target_num_max)
            ngts['{}'.format(i)] = gt
            measurement = self._generate_measurement_step(gt, self.target_num_max, self.source_num, self.noise_std)
            nmeasurements['{}'.format(i)] = measurement
            self._convert_measurement_to_vector(i, measurement, self.target_num_max, self.source_num)

        #TODO: save the data as pickle file / edit the file path and the obj file's name
        # filepath = root + data_dir + "gt.obj"
        # gt_path = os.path.join(self.config['filesystem']['root']+self.config['filesystem']['data_dir'], "gt.obj")
        self._save_pickle(ngts, dtype='GT')
        self._save_pickle(nmeasurements, dtype='Measurement')

    def _save_pickle(self, data:list, dtype:str='GT'):
        filename = dtype + '.pkl'
        filepath = os.path.join(self.data_dir, filename)
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        except:
            print("Error: cannot save the pickle file")
            raise IOError
            
    def _convert_measurement_to_vector(self, timestep:int, measurement:list, target_num:int, source_num:int):
        """
            Convert the measurement to vector
        """
        for n in range(source_num):
            feat = []
            for i in range(target_num):
                feat.append(pos2vector(measurement[n][i], self.scenario, vtype='2D'))
                # feat_path = os.path.join(self.feat_dir, 'feat_{}_{}_{}.obj'.format(timestep, n, i))
                # np.save(feat_path, feat)
            feat_m = np.maximum.reduce([f for f in feat])
            feat_m_path = os.path.join(self.feat_dir, '{}_{}.npy'.format(timestep, n)) # feat file name: timestep_source.npy
            np.save(feat_m_path, feat_m)
                
# import yaml

# with open('D-CMCM/config/config.yaml', 'r') as f:
#     config = yaml.safe_load(f)

# print(config)
# print(config['settings']['duration'])

# # TODO: test class Generator with the config file
# generator = Generator(config)
# generator.generate_data()


# TODO: generate the data with the existing log and create the pickle file by using the script