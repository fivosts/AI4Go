"""
Train dispatching file.
"""
import logging
import pathlib
import torch
import model
import optimizer
import dataset
from collections import OrderedDict

from AI4Go.util import crypto

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer(
  "batch_size",
  None,
  "Model batch size."
)

flags.DEFINE_integer(
  "num_train_steps",
  None,
  "Set number of training steps"
)

flags.DEFINE_integer(
  "num_warmup_steps",
  None,
  "Set number of warmup steps"
)

flags.DEFINE_integer(
  "steps_per_epoch",
  None,
  "Set steps per epoch."
)

flags.DEFINE_float(
  "learning_rate",
  None,
  "Set training learning rate"
)

flags.DEFINE_float(
  "temperature",
  None,
  "Set sampling temperature"
)

class Trainer(object):
  def __init__(self, workspace_path: pathlib.Path):
    self.workspace_path = workspace_path
    ## Set pytorch.
    self.num_gpus = None
    if torch.cuda.is_available():
      self.num_gpus = torch.cuda.device_count()
    else:
      raise ValueError("I hereby forbid CPU training. Go use GPUs.")

    ## Set model type
    try:
      m = {
        "CNN": lambda: model.CNN4Go,
        "Transformer": lambda: model.Transformer4Go,
      }[FLAGS.model]
    except KeyError as e:
      raise KeyError("You need to specify the model type. Received {}".format(FLAGS.model))
    self.model = torch.DataParallel(self.model)

    ## Scheduler and optimizer
    self.optimizer, self.scheduler = optimizer.create_optimizer_and_scheduler(
      model           = m,
      num_train_steps = FLAGS.num_train_steps,
      warmup_steps    = FLAGS.num_warmup_steps,
      learning_rate   = FLAGS.learning_rate,
    )

    ## Set corpus path
    self.corpus_path = pathlib.Path(FLAGS.corpus_base_path).resolve()
    if not self.corpus_path.exists():
      raise FileNotFoundError("Corpuse base path {} not found".format(self.corpus_path))

    ## Set dataloader
    try:
      self.dataset = {
        "json": lambda: dataset.JSONTraces(self.corpus_path),
        "bytes": lambda: dataset.ByteTraces(self.corpus_path),
      }[FLAGS.corpus_type]
      self.train_sampler = torch.utils.data.RandomSampler(dataset, replacement = False)
      self.dataloader = torch.utils.data.dataloader.DataLoader(
        dataset = self.dataset,
        batch_size = FLAGS.batch_size,
        sampler = self.train_sampler,
        num_workers = 0,
        drop_last = False
      )
    except KeyError as e:
      raise KeyError("You need to set the right type for corpus. Valid choices are 'json' and 'bytes'. Received {}".format(FLAGS.corpus_type))

    ## Create workspace.
    self.hash = crypto.sha256_str(
      FLAGS.model +
      FLAGS.corpus_type +
      FLAGS.batch_size +
      FLAGS.num_train_steps +
      FLAGS.learning_rate +
      self.model.hash
    )
    self.model_dir = self.workspace_path / self.hash
    if not self.model_dir.exists():
      (self.model_dir / "checkpoint").mkdir(exist_ok = False, parents = True)
      (self.model_dir / "logs").mkdir(exist_ok = False, parents = True)
    self.current_step = 0
    return

  def Train(self):
    """
    Train.
    """
    self.current_step = self.loadCheckpoint(self.model, self.scheduler, self.optimizer)
    if self.current_step < FLAGS.num_train_steps:
      self.model.zero_grad()
      self.model.train()
      loss_fn = None
      loader = iter(self.dataloader)
      for epoch in range(self.current_step, FLAGS.num_train_steps, FLAGS.steps_per_epoch):
        ## New epoch
        epoch_loss = []
        if self.current_step > epoch * FLAGS.steps_per_epoch:
          continue
        for batch_step in range(0, FLAGS.steps_per_epoch, FLAGS.batch_size):
          ## get batch
          try:
            inputs = next(loader)
          except StopIteration:
            loader = None
          outputs = self.model(inputs)
          loss = loss_fn(outputs).mean()
          loss.backward()
          epoch_loss.append(loss.item())
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
          self.optimizer.step()
          self.scheduler.step()
          self.current_step += FLAGS.batch_size
        logging.info("Epoch {} Average Loss: {}".format(epoch, sum(epoch_loss) / len(epoch_loss)))
        self.saveCheckpoint(self.current_step, self.model, self.scheduler, self.optimizer)
    logging.info("Model has been trained for {} steps".format(FLAGS.num_train_steps))
    return

  def Sample(self):
    return

  def loadCheckpoint(self, model, scheduler, optimizer) -> int:
    """
    Load model checkpoint. Loads either most recent epoch, or selected checkpoint through FLAGS.
    """
    if not (self.ckpt_path / "checkpoint.meta").exists():
      return -1

    with open(self.ckpt_path / "checkpoint.meta", 'r') as mf:
      key     = "train_step"
      get_step  = lambda x: int(x.replace("\n", "").replace("{}: ".format(key), ""))
      lines     = mf.readlines()
      entries   = set({get_step(x) for x in lines if key in x})
    ckpt_step = max(entries)
    ckpt_comp = lambda x: self.ckpt_path / "{}-{}.pt".format(x, ckpt_step)

    if isinstance(model, self.torch.nn.DataParallel):
      try:
        model.module.load_state_dict(self.torch.load(ckpt_comp("model")))
      except RuntimeError:
        """
        Pytorch doesn't love loading a DataParallel checkpoint
        to a simple model. So, the following hack is needed
        to remove the 'module.' prefix from state keys.

        OR it might as well need the opposite. Transitioning from
        single to multiple GPUs will mean that 'module.' prefix is missing
        """
        new_state_dict = OrderedDict()
        for k, v in self.torch.load(ckpt_comp("model")).items():
          if k[:7] == 'module.':
            name = k[7:] # remove `module.`
          else:
            name = 'module.' + k # Add 'module.'
          new_state_dict[name] = v
        model.module.load_state_dict(new_state_dict)
    else:
      try:
        model.load_state_dict(self.torch.load(ckpt_comp("model")))
      except RuntimeError:
        """
        Pytorch doesn't love loading a DataParallel checkpoint
        to a simple model. So, the following hack is needed
        to remove the 'module.' prefix from state keys.

        OR it might as well need the opposite. Transitioning from
        single to multiple GPUs will mean that 'module.' prefix is missing
        """
        new_state_dict = OrderedDict()
        for k, v in self.torch.load(ckpt_comp("model"), map_location = lambda storage, loc: storage).items():
          if k[:7] == 'module.':
            name = k[7:] # remove `module.`
          else:
            name = 'module.' + k # Add 'module.'
          new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    if optimizer is not None and scheduler is not None and ckpt_step > 0:
      optimizer.load_state_dict(self.torch.load(ckpt_comp("optimizer"), map_location=self.pytorch.device))
      scheduler.load_state_dict(self.torch.load(ckpt_comp("scheduler"), map_location=self.pytorch.device))
    model.eval()
    return ckpt_step

  def saveCheckpoint(self, step, model, scheduler, optimizer) -> None:
    """
    Saves model, scheduler, optimizer checkpoints per epoch.
    """
    ckpt_comp = lambda x: self.ckpt_path / "{}-{}.pt".format(x, step)

    if isinstance(model, self.torch.nn.DataParallel):
      self.torch.save(model.module.state_dict(), ckpt_comp("model"))
    else:
      self.torch.save(model.state_dict(), ckpt_comp("model"))
    self.torch.save(optimizer.state_dict(), ckpt_comp("optimizer"))
    self.torch.save(scheduler.state_dict(), ckpt_comp("scheduler"))

    with open(self.ckpt_path / "checkpoint.meta", 'a') as mf:
      mf.write("train_step: {}\n".format(step))
    return
