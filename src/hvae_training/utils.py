import os

import logging
from datetime import datetime

import csv
import types
import torch
import inspect
import pandas as pd
import json
from types import SimpleNamespace

def get_cuda_usage(GBFactor=1024**3):
    _mem_alloc=torch.cuda.memory_allocated(0)/GBFactor
    _mem_res=torch.cuda.memory_reserved(0)/GBFactor
    _mem_res_max=torch.cuda.max_memory_reserved(0)/GBFactor
    return f"GPU U/R/MR: {_mem_alloc:.2f}/{_mem_res:.2f}/{_mem_res_max:.2f}"

def save_model(model,optimizer,total_steps,time_now,args,log_msg,model_tag="best_loss"):
    ckpt = (model.state_dict(), optimizer.state_dict(), total_steps)
    fname = f"model.{time_now}.total_steps.ckpt.{model_tag}"
    fname = os.path.join(args.save_dir, fname)
    torch.save(ckpt, fname)
    log_msg(f"Saved best: {fname} {total_steps}")
    save_args(args,ofile=os.path.join(args.save_dir,'arguments_{}_{}.csv'.format(args.log_name,time_now)))


def format_arg_value(arg_value):
    if isinstance(arg_value, types.ModuleType):
        return str(arg_value)
    elif inspect.isclass(arg_value) or inspect.isfunction(arg_value) or inspect.ismethod(arg_value):
        return arg_value.__qualname__
    elif inspect.isbuiltin(arg_value):
        return arg_value.__name__
    else:
        return repr(arg_value)

def save_args(args,ofile='arguments.csv'):
    with open(ofile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for arg_name, arg_value in vars(args).items():
            formatted_value = format_arg_value(arg_value)
            writer.writerow([arg_name, formatted_value])

def load_namespace_from_json(filepath: str,extra_args={}) -> SimpleNamespace:
    """Load JSON into a SimpleNamespace."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    for k,v in extra_args.items():
        data[k] = v
    return SimpleNamespace(**data)

def load_model_args_to_obj(idf_file,atom_vocab="",vocab_proc=""):
  class  ModelLoadArgs:
    pass
  obj = ModelLoadArgs()
  #
  _df = pd.read_csv(idf_file,header=None,index_col=0)
  _ = _df.to_dict()[1]
  for k,v in _.items():
    if "object" not in v:
      _v = eval(v)
      if type(_v)==str and len(_v)==0:
         continue
      is_alpha = False
      if "None"==str(_v):
        continue
      if type(_v) not in [int,float]:
        for _c in _v:
          if _c.isalpha():
            is_alpha = True
            break
        if not is_alpha: 
          _v = eval(_v)
          if type(_v)==tuple:
            _v = ",".join([str(_) for _ in _v])
      setattr(obj, k, _v)
  setattr(obj, 'vocab', vocab_proc)
  setattr(obj, 'atom_vocab', atom_vocab)
  return obj

def setup_logger(log_name: str, start_time_obj: datetime = None):
    """
    Sets up a logger and returns a log_msg function for use in scripts.

    Args:
        log_name (str): Base name (or path) for the log file.
        start_time_obj (datetime, optional): Start time for elapsed time tracking.

    Returns:
        log_msg (function): Callable to log messages with optional elapsed time.
        start_time_obj (datetime): The start time used.
        log_file (str): The log file path.
    """
    if start_time_obj is None:
        start_time_obj = datetime.now()

    time_now = start_time_obj.strftime("%Y%m%d-%H%M%S")
    parent_dir = os.path.dirname(log_name)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    log_file = f"{log_name}_{time_now}.log"
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    def log_msg(msg, ref_time=start_time_obj):
        time_now_obj = datetime.now()
        current_time_str = time_now_obj.strftime("%Y%m%d-%H%M%S")
        if ref_time is not None:
            delta_time = time_now_obj - ref_time
            total_seconds = int(delta_time.total_seconds())
            days, remainder = divmod(total_seconds, 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, seconds = divmod(remainder, 60)
            delta_time_str = f"{days:02}:{hours:02}:{minutes:02}:{seconds:02}"
            _m = f"{delta_time_str}: {msg}."
        else:
            _m = f"{current_time_str}: {msg}."
        print(_m)
        logging.info(_m)

    return log_msg, start_time_obj, log_file

