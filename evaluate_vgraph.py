import os
import subprocess
import filecmp
from tqdm import tqdm
import numpy as np
import pickle as pkl
from src.graph.utils import load_vgraph_db, load_target_db
from src.matching.triplet_match import *
from multiprocessing import Pool,Process, Queue, SimpleQueue
import time

def decision_function(cvg_score, pvg_score, nvg_score):
    return cvg_score >= CVG_THRESH and pvg_score >= PVG_THRESH and pvg_score > nvg_score

def consume(work):
    (target_id, vg, t_trips) = work
    cvg_score, pvg_score, nvg_score = triplet_match_exact(vg, t_trips)
    return (target_id, vg, cvg_score, pvg_score, nvg_score)
