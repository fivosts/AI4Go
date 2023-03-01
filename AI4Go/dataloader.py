"""
Pytorch dataloading
"""
import torch
import pathlib
import json

class TraceDataloader(torch.utils.data.Dataloader):
  def __init__(self):
    self.data = []
    return

  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, idx: int):
    raise NotImplementedError

class JSONTraces(TraceDataloader):
  def __init__(self, json_base_path: pathlib.Path):
    super(JSONTraces, self).__init__()
    for file in json_base_path.iterdir():
      if file.suffix == ".json":
        with open(file, 'r') as inf:
          self.data.append(json.load(inf))
    return

  def compute_dataset(self) -> None:
    raise NotImplementedError

class ByteTensor(TraceDataloader):
  def __init__(self, byte_base_path: pathlib.Path):
    super(ByteTensor, self).__init__()
    for file in byte_base_path.iterdir():
      if file.suffix == ".trace":
        with open(file, 'r') as inf:
          self.data.append(inf.read())
    return

  def compute_dataset(self) -> None:
    raise NotImplementedError
