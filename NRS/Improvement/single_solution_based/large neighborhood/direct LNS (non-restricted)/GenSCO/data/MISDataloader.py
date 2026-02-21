import numpy as np
import tensorflow as tf


class MISDataloader:
    def __init__(
        self,
        dataset: dict[str, np.ndarray],
        batch_size: int,
        seed: int | None = None,
    ):
        labels = dataset['labels']
        if 'ids' in dataset.keys():
            labels = labels[dataset['ids']]
        self.labels = labels
        self.edges = dataset['edges']
        self.num_nodes = dataset['num_nodes']
        del dataset

        self.num_instances = self.edges.shape[0]
        self.batch_size = batch_size

    def __iter__(self):
        return self
    
    def __next__(self):
        indices = np.random.randint(0, self.num_instances, size=[self.batch_size], dtype=np.int32)
        return tuple(map(lambda x: x[indices], (self.num_nodes, self.edges, self.labels)))
    
    @property
    def num_nodes_padded(self) -> int:
        return self.num_nodes.max().item()
    
