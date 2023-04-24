"""
Pytorch dataloading
"""
import torch
import pathlib
import json
import glob
import numpy as np

class TraceDataloader(torch.utils.data.Dataset):
  def __init__(self):
    self.data = {
      'pass': [],
      'fail': [],
    }
    self.val_data = {
      'pass': [],
      'fail': [],
    }
    self.counters = {
      'pass': 0,
      'fail': 0,
    }
    self.val_counters = {
      'pass': 0,
      'fail': 0,
    }
    self.val_split = 0.7
    self.is_train = True
    return

  def __len__(self) -> int:
    if self.is_train:
      return len(self.data['pass']) + len(self.data['fail'])
    else:
      return len(self.val_data['pass']) + len(self.val_data['fail'])

  def __getitem__(self, idx: int):
    raise NotImplementedError

class JSONTraces(TraceDataloader):
  def __init__(self, json_base_path: pathlib.Path, sequence_length: int):
    super(JSONTraces, self).__init__()
    self.compute_dataset(json_base_path, sequence_length)
    return

  def __getitem__(self, idx: int):
    if self.is_train:
      data = self.data
      counters = self.counters
    else:
      data = self.val_data
      counters = self.val_counters
    if idx < 0:
      if -idx > len(self):
        raise ValueError
      idx = len(self) + idx
    if idx >= counters['pass']:
      return data['fail'][idx - counters['pass']]
    else:
      return data['pass'][idx]

  def compute_dataset(self, json_base_path, sequence_length) -> None:
    self.max_seq_len = 0
    self.vocab_size = 0
    seq_lens = []
    rand = np.random.RandomState()
    for label in {"pass", "fail"}:
      for file in (json_base_path / label).iterdir():
        if file.suffix == ".json":
          with open(file, 'r') as inf:
            try:
              js = json.load(inf)
              vector = self.json_to_vec(js)[:sequence_length]
              if vector:
                self.max_seq_len = max(self.max_seq_len, len(vector))
                seq_lens.append(len(vector))
                self.vocab_size = max(self.vocab_size, max(vector))
                tensor = torch.LongTensor(vector)
                if label == 'pass':
                  target = [1]
                else:
                  target = [0]
                if rand.random() > self.val_split:
                  self.data[label].append(
                    {
                      'inputs': tensor,
                      'target': torch.LongTensor(target),
                    }
                  )
                else:
                  self.val_data[label].append(
                    {
                      'inputs': tensor,
                      'target': torch.LongTensor(target),
                    }
                  )
            except RecursionError as e:
              pass
      self.counters[label] = len(self.data[label])
      self.val_counters[label] = len(self.val_data[label])
    if len(vector) > sequence_length:
      raise ValueError("Sequence length not large enough. Use at least {}".format(self.max_seq_len))
    self.pad_idx = self.vocab_size
    self.vocab_size += 1
    for k in self.data.keys():
      for idx, _ in enumerate(self.data[k]):
        if len(self.data[k][idx]['inputs']) < sequence_length:
          self.data[k][idx]['inputs'] = torch.cat(
            (self.data[k][idx]['inputs'],
            torch.LongTensor([self.pad_idx]*(sequence_length - len(self.data[k][idx]['inputs'])))),
            -1,
            # self.data[k][idx]['inputs']
          )
    for k in self.val_data.keys():
      for idx, _ in enumerate(self.val_data[k]):
        if len(self.val_data[k][idx]['inputs']) < sequence_length:
          self.val_data[k][idx]['inputs'] = torch.cat(
            (self.val_data[k][idx]['inputs'],
            torch.LongTensor([self.pad_idx]*(sequence_length - len(self.val_data[k][idx]['inputs'])))),
            -1,
            # self.val_data[k][idx]['inputs']
          )
    return

  def json_to_vec(self, json):
    events = []
    for event in json['events']:
      for k in {"Off", "Type", "Ts", "P", "G", "StkID"}:
        events += [int(i) for i in str(event[k])]
      if event["Stk"] is not None:
        for stack in event["Stk"]:
          events += [int(ord(c)) for c in stack["Fn"]]
        events += event["Args"]
    return events

class ByteTraces(TraceDataloader):
  def __init__(self, byte_base_path: pathlib.Path):
    super(ByteTraces, self).__init__()
    for file in byte_base_path.iterdir():
      if file.suffix == ".trace":
        with open(file, 'r') as inf:
          self.data.append(inf.read())
    return

  def compute_dataset(self) -> None:
    raise NotImplementedError
