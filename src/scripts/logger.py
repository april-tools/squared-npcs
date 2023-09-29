import os
from typing import Optional, Dict, Any, Union, List

import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from PIL import Image as pillow


class Logger:
    def __init__(
            self,
            verbose: bool,
            *,
            checkpoint_path: Optional[str] = None,
            tboard_path: Optional[str] = None,
            wandb_path: Optional[str] = None,
            wandb_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path
        self._tboard_writer: Optional[SummaryWriter] = None

        if tboard_path:
            self._setup_tboard(tboard_path)
        if wandb_path:
            if wandb_kwargs is None:
                wandb_kwargs = dict()
            self._setup_wandb(wandb_path, **wandb_kwargs)

    @property
    def has_graphical_endpoint(self) -> bool:
        return self._tboard_writer is not None or wandb.run

    def info(self, m: str):
        if self.verbose:
            print(m)

    def _setup_tboard(self, path: str):
        self._tboard_writer = SummaryWriter(log_dir=path)

    def _setup_wandb(
            self,
            path: str,
            project: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None,
            group: Optional[str] = None,
            name: Optional[str] = None,
            online: bool = True
    ):
        if wandb.run is None:
            wandb.init(
                project=project,
                name=name,
                dir=path,
                group=group,
                config=config,
                mode='online' if online else 'offline'
            )

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        if self._tboard_writer is not None:
            self._tboard_writer.add_scalar(tag, value, global_step=step)
        if wandb.run:
            wandb.log({tag: value}, step=step)

    def log_image(
            self,
            tag: str,
            value: Union[np.ndarray, torch.Tensor],
            step: Optional[int] = None,
            dataformats: str = "CHW"
    ):
        if self._tboard_writer is not None:
            self._tboard_writer.add_image(tag, value, global_step=step, dataformats=dataformats)
        if wandb.run:
            if isinstance(value, torch.Tensor):
                value = value.permute(1, 2, 0)
            else:
                value = value.transpose([1, 2, 0])
            image = wandb.Image(value)
            wandb.log({tag: image}, step=step)

    def log_hparams(
            self,
            hparam_dict: Dict[str, Any],
            metric_dict: Dict[str, Any],
            hparam_domain_discrete: Optional[Dict[str, List[Any]]] = None,
            run_name: Optional[str] = None
    ):
        if self._tboard_writer is not None:
            self._tboard_writer.add_hparams(
                hparam_dict,
                metric_dict,
                hparam_domain_discrete=hparam_domain_discrete,
                run_name=run_name)
        if wandb.run:
            wandb.run.summary.update(metric_dict)

    def close(self):
        if self._tboard_writer is not None:
            self._tboard_writer.close()
        if wandb.run:
            wandb.finish(quiet=True)

    def save_checkpoint(self, data: Dict[str, Any], filepath: str):
        if self.checkpoint_path:
            torch.save(data, os.path.join(self.checkpoint_path, filepath))

    def save_image(self, data: np.ndarray, filepath: str):
        if self.checkpoint_path:
            pillow.fromarray(data.transpose([1, 2, 0])).save(os.path.join(self.checkpoint_path, filepath))

    def save_array(self, array: np.ndarray, filepath: str):
        if self.checkpoint_path:
            np.save(os.path.join(self.checkpoint_path, filepath), array)
