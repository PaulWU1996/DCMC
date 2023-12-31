import numpy as np

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


# class MultiDataset(Dataset):

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