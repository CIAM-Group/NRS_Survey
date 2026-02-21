import numpy as np
import argparse
from functools import partial


def remove_self_loop(edges: np.ndarray):
    return edges[edges[:, 0] != edges[:, 1]]


def remove_replicated(edges: np.ndarray, num_nodes: int):
    edges = np.sort(edges, axis=-1)
    edge_ids = edges[:, 0] * num_nodes + edges[:, 1]
    edge_ids.sort()
    should_keep = edge_ids != np.roll(edge_ids, 1)
    edge_ids = edge_ids[should_keep]
    return np.stack(np.divmod(edge_ids, num_nodes), axis=-1)


def parse_instance(content: str, replicated_format: bool):
    content = content.strip()
    edge_str, label_str = map(lambda x: x.strip(), content.split('label'))
    labels = list(map(lambda x: bool(int(x.strip())), label_str.split(' ')))
    
    num_nodes = len(labels)
    labels = np.array(labels, dtype='bool')
    edges = list(map(lambda x: int(x.strip()), edge_str.split(' ')))
    if replicated_format:
        edges = np.array(edges, dtype=np.int32).reshape(-1, 2, 2)
        assert (np.sort(edges[:, 0], axis=-1) == np.sort(edges[:, 1], axis=-1)).all()
        edges = edges[:, 0]     # remove replicated
    else:
        edges = np.array(edges, dtype=np.int32).reshape(-1, 2)
        
        edges = remove_self_loop(edges)
        edges = remove_replicated(edges, num_nodes)
        
        whether_reverse = np.random.binomial(1, 0.5, size=[edges.shape[0]])
        edges = np.where(
            whether_reverse[:, None],
            np.flip(edges, axis=-1),
            edges,
        )
    
    return edges, labels
    

def parse_single(src_path: str, replicated_format: bool):
    with open(src_path, 'r') as f:
        lines = f.readlines()
    
    instances = list(map(partial(parse_instance, replicated_format=replicated_format), lines))
    return instances


def parse_all(src_paths: str, replicated_format: bool):
    instances = []
    for src_path in src_paths:
        instances += parse_single(src_path, replicated_format=replicated_format)
    return instances
    
    
def convert_instances(instances: list[tuple[np.ndarray, np.ndarray]]):
    edge_list = [x[0] for x in instances]
    label_list = [x[1] for x in instances]
    
    num_nodes_list = [label.size for label in label_list]
    
    num_nodes = np.array(num_nodes_list, dtype=np.int32)
    num_nodes_max = num_nodes.max().item()
    
    num_edges_max = max(e.shape[0] for e in edge_list)
    
    edge_list = list(map(
        lambda e: np.pad(e, pad_width=[(0, num_edges_max - e.shape[0]), (0, 0)], mode='constant'),
        edge_list,
    ))
    label_list = list(map(
        lambda l: np.pad(l, pad_width=[(0, num_nodes_max - l.shape[0])], mode='constant'),
        label_list,
    ))
    
    edges = np.stack(edge_list, axis=0)
    labels = np.stack(label_list, axis=0)
    
    return num_nodes, edges, labels


def parse_convert_and_save(src_paths: str, save_path: str, replicated_format: bool):
    instances = parse_all(src_paths, replicated_format=replicated_format)
    num_nodes, edges, labels = convert_instances(instances)
    np.savez_compressed(save_path, num_nodes=num_nodes, edges=edges, labels=labels)
    
    
if __name__ == '__main__':
    np.random.seed(4861792)
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--src_paths', nargs='+', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--replicated_format', type=lambda x: False if x.lower() in ['0', 'false'] else True, required=True)
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()

    if not args.overwrite:
        import os
        assert not os.path.exists(args.save_path)
        
    parse_convert_and_save(args.src_paths, args.save_path, args.replicated_format)
        