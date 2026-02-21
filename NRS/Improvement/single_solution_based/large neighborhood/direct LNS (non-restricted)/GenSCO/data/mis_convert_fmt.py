import numpy as np
import networkx
from pathlib import Path
import pickle


def convert_single(graph: networkx.Graph):
    num_nodes = graph.number_of_nodes()
    edges = np.array(graph.edges(), dtype=np.int32)
    return num_nodes, edges


def find_graph_paths(directory):
    p = Path(directory)
    paths = []
    for file in p.glob('*.gpickle'):
        paths.append(str(file))
    return paths


def get_id(path: str):
    import re
    pattern = r'(\d+)'
    match = re.findall(pattern, path)
    id = match[-1]
    return int(id)


def convert(directory: str, save_path: str):
    graph_paths = find_graph_paths(directory)
    graph_paths.sort()
    num_nodes_list = []
    edges_list = []
    id_list = []
    for path in graph_paths:
        with open(path, 'rb') as f:
            graph = pickle.load(f)
        num_nodes, edges = convert_single(graph)
        num_nodes_list.append(num_nodes)
        edges_list.append(edges)
        id_list.append(get_id(path))
        
    num_nodes = np.array(num_nodes_list, dtype=np.int32)
    num_edges_max = max(e.shape[0] for e in edges_list)
    edges_list = list(map(
        lambda e: np.pad(e, pad_width=[(0, num_edges_max - e.shape[0]), (0, 0)], mode='constant'),
        edges_list,
    ))
    ids = np.array(id_list, dtype=np.int32)
    edges = np.stack(edges_list, axis=0)
    np.savez_compressed(save_path, ids=ids, num_nodes=num_nodes, edges=edges)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()

    if not args.overwrite:
        import os
        assert not os.path.exists(args.save_path)

    convert(args.src_path, args.save_path)
