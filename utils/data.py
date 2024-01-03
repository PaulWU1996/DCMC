import numpy as np

from itertools import combinations, permutations

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class SingleDataset(Dataset):
    """
        SingleDataset: Dataset for single input (on single node)
    """
    def __init__(
        self,
        config,
        split: list
    ):
        super().__init__()

        self.config = config
        self.split = split
        self.chunks = {}

        idx = 0
        for timestep in split:
            for source_id in range(config["settings"]["source_num"]):
                feat_path = config["filesystem"]["root"] + config["filesystem"]["feat_dir"] + "/{}_{}.npy".format(timestep, source_id)
                gt_path = config["filesystem"]["root"] + config["filesystem"]["data_dir"] + "/GT.npy" # change back to GT.npy (simulate the tracker measurement -> prediction)
                self._append_pairs(idx=idx, timestep=timestep, source_id=source_id, feat_path=feat_path, gt_path=gt_path)
                idx += 1
            

    def _append_pairs(self, idx, timestep, source_id, feat_path, gt_path):
        """
            Append the pairs of the feat-gt pairs
        {
            'timestep': int
            'source_id': int
            'feat_path': str
            'gt_path': str
            'feat_loc': unavailable
            'gt_loc': unavailable
        }
        """
        self.chunks[idx] = {
            'timestep': timestep,
            'source_id': source_id,
            'feat_path': feat_path,
            'gt_path': gt_path,
            'feat_loc': None,
            'gt_loc': None
        }


    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        pair = self.chunks[idx]
        feat = np.load(pair['feat_path']).astype(np.float32)
        feat = feat[np.newaxis, :] # add one dimension for channel (1, 512, 512)
        gt = np.load(pair['gt_path'])[pair['timestep'], pair['source_id'],:].astype(np.float32)
        # gt = gt
        return feat, gt


class MultiDataset(Dataset):
    """
        MultiDataset: Dataset for multi input (on multi node)
    """
    def __init__(
        self,
        config,
        split: list
    ):
        super().__init__()

        self.config = config
        self.split = split
        self.chunks = {}

        available_sources = self._create_available_sources()

        idx = 0
        for timestep in split:
            if config["encoder"]["type"] == 'None' or config["encoder"]["type"] == 'Conformer':
                for source_list in available_sources:
                    feat_path_list = []
                    for source_id in source_list:
                        feat_path = config["filesystem"]["root"] + config["filesystem"]["feat_dir"] + "/{}_{}.npy".format(timestep, source_id)
                        feat_path_list.append(feat_path)
                    gt_path = config["filesystem"]["root"] + config["filesystem"]["data_dir"] + "/Measurement.npy"
                    self._append_pairs(idx=idx, timestep=timestep, source_id=source_id, feat_path=feat_path_list, gt_path=gt_path)
                    idx += 1
            elif config["encoder"]["type"] == 'CMConformer':
                for source_id in range(config["settings"]["source_num"]):
                    sources_list = available_sources[source_id]
                    for source_list in sources_list:
                        feat_path_list = []
                        for id in source_list:
                            feat_path = config["filesystem"]["root"] + config["filesystem"]["feat_dir"] + "/{}_{}.npy".format(timestep, id)
                            feat_path_list.append(feat_path)
                        gt_path = config["filesystem"]["root"] + config["filesystem"]["data_dir"] + "/Measurement.npy"
                        feat_path = config["filesystem"]["root"] + config["filesystem"]["feat_dir"] + "/{}_{}.npy".format(timestep, source_id)
                        self._append_pairs(idx=idx, timestep=timestep, source_id=source_id, feat_path=feat_path, gt_path=gt_path, second_feat_path=feat_path_list)
                        idx += 1
            else:
                raise NotImplementedError("Encoder type {} not implemented".format(config["encoder"]["type"]))    
            
    def _create_available_sources(self):
        """
            Create the available source list
        """
        full_source_list = list(range(self.config["settings"]["source_num"]))
        # None & Conformer | CMConformer
        if self.config["encoder"]["type"] == 'None' or self.config["encoder"]["type"] == 'Conformer':
            available_sources = list(combinations(full_source_list, self.config["settings"]["listener_max"]))
        elif self.config["encoder"]["type"] == 'CMConformer':
            available_sources = {}
            for source_id in range(len(full_source_list)):
                tmp_source_list = full_source_list.copy()
                tmp_source_list.remove(source_id)
                available_source_list = list(combinations(tmp_source_list, self.config["settings"]["listener_max"]-1))
                available_sources[source_id] = available_source_list
        else:
            raise NotImplementedError("Encoder type {} not implemented".format(self.config["encoder"]["type"]))
        return available_sources

    def _append_pairs(self, idx, timestep, source_id, feat_path, gt_path, second_feat_path=None):
        """
            Append the pairs of the feat-gt pairs
        {
            'timestep': int
            'source_id': int
            'feat_path': str
            'gt_path': str
            'feat_loc': unavailable
            'gt_loc': unavailable
            'second_feat_path': list= None
        }
        """
        self.chunks[idx] = {
            'timestep': timestep,
            'source_id': source_id,
            'feat_path': feat_path,
            'gt_path': gt_path,
            'feat_loc': None,
            'gt_loc': None,
            'second_feat_path': second_feat_path
        }


    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        pair = self.chunks[idx]

        if pair['second_feat_path'] is None:
            # Conformer or None Encoder
            feat = []
            for path in pair['feat_path']:
                feat.append(np.load(path).astype(np.float32))
            feat = np.array(feat)
            feat = np.sum(feat, axis=0)
            feat = feat / np.max(feat)
            feat = feat[np.newaxis, :] # 1, 512, 512
            second_feat = 0
        else:
            # CMConformer Encoder
            feat = np.load(pair['feat_path']).astype(np.float32)
            feat = feat[np.newaxis, :]
            # second_feat
            second_feat = []
            for path in pair['second_feat_path']:
                second_feat.append(np.load(path).astype(np.float32))
            second_feat = np.array(second_feat)
            second_feat = np.sum(second_feat, axis=0)
            second_feat = second_feat / np.max(second_feat)
            second_feat = second_feat[np.newaxis, :]
        gt = np.load(pair['gt_path'])[pair['timestep'], pair['source_id'],:].astype(np.float32)

        return [feat, second_feat], gt

class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        idx = list(range(config["settings"]["duration"]))
        split = config["training"]["split"]
        self.training_split = idx[:int(split[0]*len(idx))]
        self.validation_split = idx[int(split[0]*len(idx)):int(split[0]*len(idx))+int(split[1]*len(idx))]
        self.test_split = idx[-int(split[2]*len(idx)):]

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # setup the dataset instances
        if stage == 'fit' or stage is None:
            if self.config['settings']['listening']:
                # distributed settings
                self.train_set = MultiDataset(self.config, self.training_split)
                self.val_set = MultiDataset(self.config, self.validation_split)
            else:
                # single node settings
                self.train_set = SingleDataset(self.config, self.training_split)
                self.val_set = SingleDataset(self.config, self.validation_split)
        if stage == 'test' or stage is None:
            self.test_set = SingleDataset(self.config, self.test_split)

        return super().setup(stage)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.config["training"]["batch_size"], shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.config["training"]["batch_size"], shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.config["training"]["batch_size"], shuffle=False, num_workers=4)


# import yaml
# with open("/vol/research/VS-Work/PW00391/D-CMCM/config/config.yaml", "r") as f:  # abs path of 
#         config = yaml.safe_load(f)

# data = DataModule(config)
# data.setup()
# for batch, label in data.train_dataloader():
#     print(batch.shape, label.shape)
#     break
# feat, second_feat, gt = data.train_dataloader.__iter__()#data.train_set.__getitem__(0)
# print(len(data.train_set))
