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



def closer_cluster_chooser(c1, c2):
    return c1 if c1[-1] < c2[-1] else c2


def find_closest_cluster(pt, clusters, dist_fn):
    best_dist = np.inf
    best_id = -1 # SENTINEL
    for i, centroid in enumerate(clusters):
        dist = dist_fn(pt, centroid)
        if dist <= best_dist:
            best_id = i
            best_dist = dist
    assert best_id >= 0
    return best_id

conf = SparkConf()
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
def splint(row):
    return np.array([float(x) for x in row.split()])

centroids = sc.textFile(PATH_C1).map(splint).cache()
points = sc.textFile(DATA_PATH).map(splint).zipWithUniqueId().cache()
if False:  # pickling error
    cluster_assigner = lambda pt: find_closest_cluster(pt, centroids, euclid_dist)
    candidate_assignments = points.map(cluster_assigner)
    ass = candidate_assignments.take(5).collect()





DESIRED_SHAPE = 4601

candidate_assignments = points.cartesian(centroids)
with_dists = candidate_assignments.map(add_dist_to_values)
ass = with_dists.take(2)



best_cluster = with_dists.reduceByKey(closer_cluster_chooser)
best_cluster.take(2)

def centroid_repr(x):
    return x[1]

crepr = best_cluster.map(centroid_repr)
new_cost = crepr.reduceByKey(lambda x: x[-1].sum())

new_c_attempt = crepr.reduceByKey(lambda x: x[-2].mean())



# cost fn
#

# ass[0]
# closest_centroid
# can like groupbykey.reduce top recompute centroids



shutil.rmtree('cluster_assignments', ignore_errors=True)
candidate_assignments.saveAsTextFile('cluster_assignments')


shutil.rmtree('centroids', ignore_errors=True)
centroids.saveAsTextFile('centroids')
sc.stop()
