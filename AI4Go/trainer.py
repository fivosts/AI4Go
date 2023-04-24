"""
Train dispatching file.
"""
import logging
import pathlib
import tqdm
import copy
import torch
import model
import optimizer
import dataset
from collections import OrderedDict

from util import crypto

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

    ## Set corpus path
    self.corpus_path = pathlib.Path(FLAGS.corpus_base_path).resolve()
    if not self.corpus_path.exists():
      raise FileNotFoundError("Corpuse base path {} not found".format(self.corpus_path))

    ## Set dataloader
    try:
      self.dataset = {
        "json": lambda: dataset.JSONTraces(self.corpus_path, FLAGS.sequence_length),
        "bytes": lambda: dataset.ByteTraces(self.corpus_path),
      }[FLAGS.corpus_type]()
      self.val_dataset = copy.deepcopy(self.dataset)
      self.val_dataset.is_train = False
      print(len(self.dataset))
      self.train_sampler = torch.utils.data.RandomSampler(self.dataset, replacement = False)
      self.val_sampler = torch.utils.data.RandomSampler(self.val_dataset, replacement = False)
      self.dataloader = torch.utils.data.dataloader.DataLoader(
        dataset = self.dataset,
        batch_size = FLAGS.batch_size,
        sampler = self.train_sampler,
        num_workers = 0,
        drop_last = False
      )
      self.val_dataloader = torch.utils.data.dataloader.DataLoader(
        dataset = self.val_dataset,
        batch_size = FLAGS.batch_size,
        sampler = self.val_sampler,
        num_workers = 0,
        drop_last = False
      )
    except KeyError as e:
      raise KeyError("You need to set the right type for corpus. Valid choices are 'json' and 'bytes'. Received {}".format(FLAGS.corpus_type))

    ## Set model type
    try:
      m = {
        "CNN": model.CNN4Go,
        "Transformer": model.Transformer4Go,
      }[FLAGS.model](
        vocab_size              = self.dataset.vocab_size,
        embedding_size          = FLAGS.embedding_size,
        padding_idx             = self.dataset.pad_idx,
        sequence_length         = FLAGS.sequence_length,
        dropout_prob            = FLAGS.dropout_prob,
        num_attention_heads     = FLAGS.num_attention_heads,
        layer_norm_eps          = FLAGS.layer_norm_eps,
        transformer_feedforward = FLAGS.transformer_feedforward,
        num_transformer_layers  = FLAGS.num_transformer_layers,
      )
    except KeyError as e:
      raise KeyError("You need to specify the model type. Received {}".format(FLAGS.model))
    self.model = torch.nn.DataParallel(m)

    ## Scheduler and optimizer
    self.optimizer, self.scheduler = optimizer.create_optimizer_and_scheduler(
      model           = m,
      num_train_steps = FLAGS.num_train_steps,
      warmup_steps    = FLAGS.num_warmup_steps,
      learning_rate   = FLAGS.learning_rate,
    )

    ## Create workspace.
    self.hash = crypto.sha256_str(
      str(FLAGS.model) +
      str(FLAGS.corpus_type) +
      str(FLAGS.batch_size) +
      str(FLAGS.num_train_steps) +
      str(FLAGS.learning_rate) +
      str(FLAGS.embedding_size) +
      str(FLAGS.sequence_length) +
      str(FLAGS.dropout_prob) +
      str(FLAGS.num_attention_heads) +
      str(FLAGS.layer_norm_eps) +
      str(FLAGS.transformer_feedforward) +
      str(FLAGS.num_transformer_layers)
    )
    self.model_dir = self.workspace_path / self.hash
    self.ckpt_path = self.model_dir / "checkpoint"
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
      loss_fn = torch.nn.CrossEntropyLoss()
      loader = iter(self.dataloader)
      for epoch in tqdm.tqdm(range(self.current_step, FLAGS.num_train_steps, FLAGS.steps_per_epoch), total = FLAGS.num_train_steps // FLAGS.steps_per_epoch, desc =  "Epoch"):
        ## New epoch
        epoch_loss = []
        if self.current_step > epoch * FLAGS.steps_per_epoch:
          continue
        for batch_step in tqdm.tqdm(range(0, FLAGS.steps_per_epoch, FLAGS.batch_size), total = FLAGS.steps_per_epoch // FLAGS.batch_size, desc = "Batch"):
          ## get batch
          try:
            inputs = next(loader)
          except StopIteration:
            loader = iter(self.dataloader)
          outputs = self.model(inputs['inputs'].to('cuda'))
          # print(outputs['output_logits'])
          # print(outputs['output_logits'].shape)
          # print(inputs['target'].squeeze(1))
          # print(inputs['target'].squeeze(1).shape)
          loss = loss_fn(outputs['output_logits'], inputs['target'].squeeze(1).to('cuda')).mean()
          loss.backward()
          epoch_loss.append(loss.item())
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
          self.optimizer.step()
          self.scheduler.step()
          self.current_step += FLAGS.batch_size
        logging.info("Epoch {} Average Loss: {}".format(epoch, sum(epoch_loss) / len(epoch_loss)))
        self.saveCheckpoint(self.current_step, self.model, self.scheduler, self.optimizer)
        self.Validate()
    logging.info("Model has been trained for {} steps".format(FLAGS.num_train_steps))
    return

  def Validate(self):
    with torch.no_grad():
      accuracy = [0, 0]
      self.model.eval()
      for inputs in self.val_dataloader:
        outputs = self.model(inputs['inputs'].to('cuda'))
        output_labels = torch.argmax(torch.softmax(outputs['output_logits'], dim = -1).cpu(), dim = -1)
        accuracy[0] += sum(output_labels == inputs['target'].reshape(-1,))
        accuracy[1] += output_labels.size(0)
      logging.info("Epoch val accuracy: {}".format(accuracy[0] / accuracy[1]))
      self.model.train()
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

    if isinstance(model, torch.nn.DataParallel):
      try:
        model.module.load_state_dict(torch.load(ckpt_comp("model")))
      except RuntimeError:
        """
        Pytorch doesn't love loading a DataParallel checkpoint
        to a simple model. So, the following hack is needed
        to remove the 'module.' prefix from state keys.

        OR it might as well need the opposite. Transitioning from
        single to multiple GPUs will mean that 'module.' prefix is missing
        """
        new_state_dict = OrderedDict()
        for k, v in torch.load(ckpt_comp("model")).items():
          if k[:7] == 'module.':
            name = k[7:] # remove `module.`
          else:
            name = 'module.' + k # Add 'module.'
          new_state_dict[name] = v
        model.module.load_state_dict(new_state_dict)
    else:
      try:
        model.load_state_dict(torch.load(ckpt_comp("model")))
      except RuntimeError:
        """
        Pytorch doesn't love loading a DataParallel checkpoint
        to a simple model. So, the following hack is needed
        to remove the 'module.' prefix from state keys.

        OR it might as well need the opposite. Transitioning from
        single to multiple GPUs will mean that 'module.' prefix is missing
        """
        new_state_dict = OrderedDict()
        for k, v in torch.load(ckpt_comp("model"), map_location = lambda storage, loc: storage).items():
          if k[:7] == 'module.':
            name = k[7:] # remove `module.`
          else:
            name = 'module.' + k # Add 'module.'
          new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    if optimizer is not None and scheduler is not None and ckpt_step > 0:
      optimizer.load_state_dict(torch.load(ckpt_comp("optimizer"), map_location='cuda'))
      scheduler.load_state_dict(torch.load(ckpt_comp("scheduler"), map_location='cuda'))
    model.eval()
    return ckpt_step

  def saveCheckpoint(self, step, model, scheduler, optimizer) -> None:
    """
    Saves model, scheduler, optimizer checkpoints per epoch.
    """
    ckpt_comp = lambda x: self.ckpt_path / "{}-{}.pt".format(x, step)

    if isinstance(model, torch.nn.DataParallel):
      torch.save(model.module.state_dict(), ckpt_comp("model"))
    else:
      torch.save(model.state_dict(), ckpt_comp("model"))
    torch.save(optimizer.state_dict(), ckpt_comp("optimizer"))
    torch.save(scheduler.state_dict(), ckpt_comp("scheduler"))

    with open(self.ckpt_path / "checkpoint.meta", 'a') as mf:
      mf.write("train_step: {}\n".format(step))
    return
