from annoy import AnnoyIndex
import os

def create_annoy_index(f, metric):
    return AnnoyIndex(f, metric)

def add_item_to_annoy_index(index, item_id, vector):
    index.add_item(item_id, vector)

def build_annoy_index(index, n_trees, n_jobs=-1):
    index.build(n_trees, n_jobs)

def save_annoy_index(index, filename):
    index.save(filename)

def load_annoy_index(f, metric, filename):
    index = AnnoyIndex(f, metric)
    index.load(filename)
    return index

def unload_annoy_index(index):
    index.unload()

def query_annoy_index(index, vector, n, search_k=-1, include_distances=False):
    return index.get_nns_by_vector(vector, n, search_k, include_distances)

def get_item_vector(index, item_id):
    return index.get_item_vector(item_id)

def get_distance(index, i, j):
    return index.get_distance(i, j)

def get_n_items(index):
    return index.get_n_items()

def get_n_trees(index):
    return index.get_n_trees()

def on_disk_build(index, filename):
    index.on_disk_build(filename)

def set_seed(index, seed):
    index.set_seed(seed)
