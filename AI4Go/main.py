"""
Main file
"""
import pathlib
from absl import app, flags

import trainer

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "workspace_path",
  None,
  "Set base workspace path to use."
)

flags.DEFINE_string(
  "corpus_base_path",
  None,
  "Set base folder where all data files are located."
)

flags.DEFINE_string(
  "corpus_type",
  None,
  "Select corpus type. Options are 'byte' and 'json'"
)

flags.DEFINE_string(
  "model",
  None,
  "Select model architecture to use for training."
)

def main(*args, **kwargs) -> None:
  if FLAGS.workspace_path is None:
    raise ValueError("You have to set workspace path")
  p = pathlib.Path(FLAGS.workspace_path)
  p.mkdir(exist_ok = True, parents = True)
  t = trainer.Trainer(p)
  t.Train()
  return

if __name__ == "__main__":
  app.run(main)
  exit(0)
