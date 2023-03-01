"""
Train dispatching file.
"""
import logging
import pathlib
import torch
import model
import dataloader

from absl import flags

FLAGS = flags.FLAGS

class Trainer(object):
  def __init__(self, workspace_path: pathlib.Path):
    self.workspace_path = workspace_path
    ## Set model type
    try:
      self.model = {
        "CNN": lambda: model.CNN4Go,
        "Transformer": lambda: model.Transformer4Go,
      }[FLAGS.model]
    except KeyError as e:
      raise KeyError("You need to specify the model type. Received {}".format(FLAGS.model))

    ## Set corpus path
    self.corpus_path = pathlib.Path(FLAGS.corpus_base_path).resolve()
    if not self.corpus_path.exists():
      raise FileNotFoundError("Corpuse base path {} not found".format(self.corpus_path))

    ## Set dataloader
    try:
      self.dataloader = {
        "json": lambda: dataloader.JSONTraces(self.corpus_path),
        "bytes": lambda: dataloader.ByteTraces(self.corpus_path),
      }[FLAGS.corpus_type]
    except KeyError as e:
      raise KeyError("You need to set the right type for corpus. Valid choices are 'json' and 'bytes'. Received {}".format(FLAGS.corpus_type))

    ## Create workspace.
    self.hash = crypto.sha256_str(FLAGS.model + FLAGS.corpus_type + self.model.hash)
    self.model_dir = self.workspace_path / self.hash
    if not self.model_dir.exists():
      (self.model_dir / "checkpoint").mkdir(exist_ok = False, parents = True)
      (self.model_dir / "logs").mkdir(exist_ok = False, parents = True)
    return

  def Train(self):
    return

  def Sample(self):
    return
