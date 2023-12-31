import yaml
import os
import sys
import pickle
import numpy as np

"""
    1. Read config file
"""

try:
    with open("/vol/research/VS-Work/PW00391/D-CMCM/config/config.yaml", "r") as f:  # abs path of config file
        config = yaml.safe_load(f)
    sys.path.append(config["filesystem"]["root"])
    print("Config file read and root path added!")
except Exception as e:
    print(e)
    print("Error: Cannot read config file")
    sys.exit()

from utils.utils import CustomException

# check the file system
try:
    filesystem = list(config["filesystem"].values())
    root = filesystem[0]
    filesystem.remove(root)
    for sub_dir in filesystem:
        if not os.path.exists(root + sub_dir):
            raise CustomException(message="Filesystem: {} does not exist".format(root + sub_dir), error_code=0)
    print("Filesystem check pass!")
except CustomException as ce:
    print("Error code: {}, {}".format(ce.args[0], ce.args[1]))
    try:
        dir = ce.args[1].split(" ")[1]
        os.mkdir(dir)
        print("Directory {} created".format(dir))
    except CustomException(error_code=1, message="Cannot create directory {}".format(dir)) as ce:
        print("Error code: {}, {}".format(ce.args[0], ce.args[1]))
        sys.exit()

# generate the data
if config["training"]["generate_data"]:
    from utils.generate_data import Generator
    generator = Generator(config)
    generator.generate_data()
    print("Data generated")

# preprocess the data
from utils.preprocess import gt_preprocess, measurement_preprocess
if config["training"]["preprocess_data"]:
    with open(config["filesystem"]["root"] + config["filesystem"]["data_dir"] + "/GT.pkl", 'rb') as file:
        gt_obj = pickle.load(file)
    processed_gt = gt_preprocess(config, gt_obj)
    np.save(config["filesystem"]["root"] + config["filesystem"]["data_dir"] + "/GT.npy", processed_gt)
    print("GT processed and saved!")
    with open(config["filesystem"]["root"] + config["filesystem"]["data_dir"] + "/Measurement.pkl", 'rb') as file:
        measurement_obj = pickle.load(file)
    processed_measurement = measurement_preprocess(config, measurement_obj, flag='feat')
    np.save(config["filesystem"]["root"] + config["filesystem"]["data_dir"] + "/Measurement.npy", processed_measurement)
    print("Measurement processed and saved!")

# training
from script.training import train
train(config)
# print("Training finished!")


# measurement = np.load('/vol/research/VS-Work/PW00391/D-CMCM/datas/Measurement.npy')
# print(measurement.shape)

# import numpy as np
# feat = np.load("/vol/research/VS-Work/PW00391/D-CMCM/datas/feats/7_5.npy")
# print(feat.shape)
# import matplotlib.pyplot as plt
# plt.imshow(feat)
# plt.savefig("test.png")

# from utils.data import SingleDataset, DataModule
# import torch
# split = [i for i in range(7)]
# print(split)
# dataset = SingleDataset(config, split)
# data_module = DataModule(config)
# data_module.setup()
# print(len(data_module.test_set))

# from cores.model.DCMC import DCMC
# from cores.training.loss import CustomLoss
# from cores.training.metrics import eval
# model = DCMC(config)
# for batch_data, batch_label in data_module.val_dataloader():
#     predict = model.forward(batch_data)
#     loss = CustomLoss()
#     loss_value = loss(predict, batch_label)
#     correct, total = eval(predict, batch_label, threshold=20/512)
#     print(loss_value, correct, total)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# loss_value.backward()
# optimizer.step()

# print(predict.shape)

# idx = list(range(10))
# print(idx)
# import random
# random.shuffle(idx)
# print(idx)



