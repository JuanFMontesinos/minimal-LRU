from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import torchtext
import torchvision
from einops.layers.torch import Rearrange, Reduce
from PIL import Image  # Only used for Pathfinder
from .base import default_data_path, SequenceDataset

from aiwizard_core.dataframe import io, data
class SombreroDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir,l_max):
        evaluations, measurements = io.load_dataset(data_dir, multiprocessing=False, normalize_energy=True,extract_segments=True)
        classes = {x:None for x in evaluations.event_group.unique().tolist()}
        classes.pop('ar')
        evaluations,measurements = \
            data.purge_dataset(evaluations,measurements, 
                            classes = list(classes.keys()),
                            purge_variotherms=False,purge_isotherms=True, purge_drifts=False)
        evaluations,measurements = \
            data.HFwrtHeatRateResampler(evaluations,measurements,0.2).homogeneous_resampling()
        # Remove measurements with no evaluation
        for key in list(measurements.keys()):
            if key not in evaluations.measurement.to_list():
                measurements.pop(key)
                print(f"Measurement {key} with no evaluation removed from the dataset")
                
        # Removing measurements larger than 5k
        measurements = {k: v for k, v in measurements.items() if v.shape[0] <= l_max}
                                                                       

        self.evaluations = evaluations
        self.measurements = measurements
        self.keys = list(self.measurements.keys())
        print(f'Sombrero dataset at {data_dir} loaded with {len(self)} measurements and {len(self.evaluations)} evaluations')

    def __len__(self):
        return len(self.measurements)

    def __getitem__(self, idx):
        key = self.keys[idx]
        measurement = self.measurements[key]
        hf_key = 'heatflow' if 'heatflow' in measurement else 'heatflow*weight^-1'
        evaluation = self.evaluations[self.evaluations.measurement==key]
        hf = measurement[hf_key].to_numpy()
        if hf.max() == hf.min():
            pass
        else:
            hf = (hf - hf.min()) / (hf.max() - hf.min())
        hf = torch.from_numpy(hf).float()
        gt = torch.zeros_like(hf,dtype=torch.long)
        for row in evaluation.itertuples():
            gt[row.event_start:row.event_end] = 1
        return {"input_ids": hf, "Target": gt}
    
class SombreroEventDetection(SequenceDataset):
    _name_ = "sombrero_events"
    d_output = 2
    l_output = 0
    @property       
    def init_defaults(self):
        return {
            "l_max": 3000,
        }
    def load_dataset(self,stage:str):
        assert stage in ["train", "val", "test"], f"Invalid stage {stage}"
        path = self.data_dir.with_name(f"{stage}")
        return SombreroDataset(path,self.l_max)
    
    def load_datasets(self):
        datasets = {stage: self.load_dataset(stage) for stage in ["train", "val", "test"]}
        return datasets
    
    def setup(self, stage=None):
        dataset = self.load_datasets()
        self.dataset_train, self.dataset_val, self.dataset_test = (
            dataset["train"],
            dataset["val"],
            dataset["test"],
        )

        def collate_batch(batch):
            xs, ys = zip(*[(data["input_ids"], data["Target"]) for data in batch])
            # Added zeros to the length for start of mask
            lengths = torch.tensor([[0, len(x)] for x in xs])
            xs = nn.utils.rnn.pad_sequence(xs, batch_first=True)
            ys = nn.utils.rnn.pad_sequence(ys, batch_first=True)
            return xs, ys, {"lengths": lengths}

        self._collate_fn = collate_batch