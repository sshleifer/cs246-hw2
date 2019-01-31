import re
import sys
from pyspark import SparkConf, SparkContext
import shutil


PATH_C1 = 'q2/data/c1.txt'
PATH_C2 = 'q2/data/c2.txt'
DATA_PATH = 'q2/data/data.txt'
MAX_ITER = 20
K = 10
import numpy as np
import pickle


def euclid_dist(a,b):
    return np.sqrt(np.sum((a - b)**2))


def l1_dist(a,b):
    return np.sum(np.abs(a - b))


def add_dist_to_values(x) -> tuple:
    pt = x[0][0]
    centroid = x[1]
    uid = x[0][1]
    dist = euclid_dist(pt, centroid)
    return ((uid), [pt, centroid, dist])


def closer_cluster_chooser(c1, c2):
    return c1 if c1[-1] < c2[-1] else c2


def splint(row):
    return np.array([float(x) for x in row.split()])



def save_text_file(rdd, path, overwrite=True):
    if overwrite:
        shutil.rmtree(path, ignore_errors=True)
    rdd.repartition(1).saveAsTextFile(path)



def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def get_center(cluster_pts):
    """Average of all cluster members"""
    return np.array(list(cluster_pts)).mean(axis=0)


from tqdm import *

def run_kmeans(sc, centroids_path, dist_fn=euclid_dist):

    points = sc.textFile(DATA_PATH).map(splint)
    orig_centroids = sc.textFile(centroids_path).map(splint).collect()
    centroids = orig_centroids
    orig_shape = np.array(centroids).shape
    costs = []
    global cost
    cost = sc.accumulator(0)

    is_euclid = (dist_fn.__name__ == 'euclid_dist')

    def find_closest_cluster(pt, clusters, dist_fn):
        best_dist = np.inf
        best_id = -1  # SENTINEL
        for i, centroid in enumerate(clusters):
            dist = dist_fn(pt, centroid)
            if dist <= best_dist:
                best_id = i
                best_dist = dist
        assert best_id >= 0
        global cost
        if dist_fn.__name__ == 'euclid_dist':
            cost += (best_dist ** 2)
        else:
            cost += best_dist
        return (best_id, pt)

    for k in tqdm_notebook(range(MAX_ITER)):
        cost = sc.accumulator(0)
        assert cost.value == 0, 'Accumulator did not reset'
        cluster_assigner = lambda pt: find_closest_cluster(pt, centroids, dist_fn)
        candidate_assignments = points.map(cluster_assigner)
        gb = candidate_assignments.groupByKey()
        assert np.array(gb.mapValues(len).values().collect()).sum() == 4601
        new_centroids = gb.mapValues(get_center).values().collect()
        if is_euclid and  k > 0 and cost.value > costs[-1]:  # cant check before here
            raise ValueError(f'cost increased from {costs[-1]:.1f} to {cost.value: .1f} for {dist_fn.__name__}')
        costs.append(cost.value)
        new_shape = np.array(centroids).shape
        if new_shape != orig_shape: raise ValueError(f'new shape: {new_shape}, orig_shape: {orig_shape} at iter {k}')

        centroids = new_centroids

    return costs, centroids
#
# DESIRED_SHAPE = 4601
# #save_text_file(costs, 'costs')
# pickle_save(centroids, 'centroids.pkl')

if __name__ == '__main__':
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    run_kmeans(PATH_C1, euclid_dist)
    run_kmeans(PATH_C2, euclid_dist)
    sc.stop()



