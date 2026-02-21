from tensorboardX import SummaryWriter
from queue import Queue
import os
from helpers import parse_bool
import numpy as np


RESUME_ALLOWED = os.getenv('RESUME_LOG', default='false')
RESUME_ALLOWED = parse_bool(RESUME_ALLOWED)


class Logger:
    def __init__(self, logdir: str, step: int = 0):
        resume = RESUME_ALLOWED and step > 0
        if not resume:
            assert not os.path.exists(logdir)
            os.makedirs(logdir)
        self.writer = SummaryWriter(logdir, purge_step=step + 1 if resume else None)
        self.queue = Queue(-1)
        self.step = step

    def __call__(self, **kwargs):
        self.queue.put(kwargs)
        if self.queue.qsize() > 20:
            metrics: dict = self.queue.get()
            self.step += 1
            for k, v in metrics.items():
                self.writer.add_scalar(k, np.array(v).mean(), self.step)

    def flush(self):
        while not self.queue.empty():
            metrics: dict = self.queue.get()
            self.step += 1
            for k, v in metrics.items():
                self.writer.add_scalar(k, np.array(v).mean(), self.step)
        self.writer.flush()
