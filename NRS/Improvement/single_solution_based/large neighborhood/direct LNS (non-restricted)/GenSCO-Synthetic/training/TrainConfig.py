from dataclasses import dataclass
import typing as tp
from .init_optimizer import init_optimizer


@dataclass
class TrainConfig:
    num_nodes: int | None = None

    num_steps: int = 10 ** 6
    num_warmup_steps: int = 0
    batch_size: int = 1024
    peak_lr: float = 2e-5
    end_lr: float = 1e-6
    weight_decay: float = 1e-4
    clip_norm: float | None = None
    seed: int = 42

    noise_type: tp.Literal['shuffle', 'randperm'] = 'randperm'
    target_disruption: tuple[int, int] | None | tp.Literal['default'] = 'default'

    optimizer_type: tp.Literal['adamw', 'muon', 'muon_decay'] = 'adamw'

    
    def __post_init__(self):
        if self.target_disruption == 'default':
            self.target_disruption = (int(self.num_nodes * 0.4), int(self.num_nodes * 0.4) + 1)
        if isinstance(self.target_disruption, list):
            self.target_disruption = tuple(self.target_disruption)

    def init_optimizer(self):
        return init_optimizer(
            self.num_steps, self.num_warmup_steps,
            self.peak_lr, 1e-6, self.end_lr,
            self.weight_decay, self.clip_norm,
            optimizer_type=self.optimizer_type,
        )

