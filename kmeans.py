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


conf = SparkConf()
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

def euclid_dist(a,b):
    #if isinstance(a, list): a,b = np.array(a), np.array(b)
    return np.sqrt(np.sum((a - b)**2))


def l1_dist(a,b):
    #if isinstance(a, list): a, b = np.array(a), np.array(b)
    return np.sum(np.abs(a - b))


def add_dist_to_values(x) -> tuple:
    pt = x[0][0]
    centroid = x[1]
    uid = x[0][1]
    dist = euclid_dist(pt, centroid)
    return ((uid), [pt, centroid, dist])



from pyspark.accumulators import AccumulatorParam
class VectorAccumulatorParam(AccumulatorParam):
     def zero(self, value):
         return [0.0] * len(value)
     def addInPlace(self, val1, val2):
         val1[val2] +=1
         return val1

cluster_size = sc.accumulator([0] * K, VectorAccumulatorParam)
def closer_cluster_chooser(c1, c2):
    return c1 if c1[-1] < c2[-1] else c2


def splint(row):
    return np.array([float(x) for x in row.split()])


def find_closest_cluster(pt, clusters, dist_fn):
    best_dist = np.inf
    best_id = -1 # SENTINEL
    for i, centroid in enumerate(clusters):
        dist = dist_fn(pt, centroid)
        if dist <= best_dist:
            best_id = i
            best_dist = dist
    assert best_id >= 0
    global cost
    cost += dist
    return (best_id ,pt)


# def pt_sum(a, b):
#     assert isinstance(a, list)
#     return np.array(a) + np.array(b)


def save_text_file(rdd, path, overwrite=True):
    if overwrite:
        shutil.rmtree(path, ignore_errors=True)
    rdd.repartition(1).saveAsTextFile(path)


import pickle


def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


#def


points = sc.textFile(DATA_PATH).map(splint)
centroids = sc.textFile(PATH_C1).map(splint).cache().collect()
orig_shape = np.array(centroids).shape
print(f'ORIG shape: {orig_shape}')
costs = []
for k in range(MAX_ITER):
    cost = sc.accumulator(0)
    cluster_assigner = lambda pt: find_closest_cluster(pt, centroids, euclid_dist)
    candidate_assignments = points.map(cluster_assigner)
    gb = candidate_assignments.groupByKey()
    new_centroids = gb.mapValues(lambda x: np.array(list(x)).mean(axis=1)).values().collect()
    new_shape = np.array(centroids).shape
    print()
    if new_shape != orig_shape:
        print(f'new shape: {new_shape}, orig_shape: {orig_shape} at iter {k}')
        break

    centroids = new_centroids

    print('K', k)

print(f'costs: {costs}')
DESIRED_SHAPE = 4601
#save_text_file(costs, 'costs')
pickle_save(centroids, 'centroids.pkl')



sc.stop()
