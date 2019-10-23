import os
import filecmp
from tqdm import tqdm
import numpy as np
import pickle as pkl
from src.graph.utils import load_vgraph_db, load_target_db
from src.matching.triplet_match import *

from multiprocessing import Pool,Process, Queue, SimpleQueue
import time

def decision_function(cvg_score, pvg_score, nvg_score):
    return cvg_score > CVG_THRESH and pvg_score > PVG_THRESH and pvg_score > nvg_score

def consume(work):
    (target_id, vg, t_trips, match_mode) = work
    if match_mode == 'exact':
        cvg_score, pvg_score, nvg_score = triplet_match_exact(vg, t_trips)
    else: # is 'approx'
        cvg_score, pvg_score, nvg_score = triplet_match_approx(vg, t_trips)
    return (target_id, vg, cvg_score, pvg_score, nvg_score)



PRINT_UNK=True
# This script will run a set of evaluations to test how well vGraph is doing.
# Additionally it will check to see if any other tools exist to compare with(if they are installed).
# The general workflow goes as follows:
#   - we use the files  downloaded from the code_miner to test
#   - we see how well techniques perform against the vuln/patch files
#   - we see how well techniques perform against the before/after files
#   - we see how well techniques perform against before/after files that have code modifications from the vuln/patch files

# In Each case, we consider it a TP if the technique flags on a known-vulnerable function with the appropriate CVE #
# We consider it a FP if the technique flags on a known-patched function with the appropriate CVE # 
# In the case where a technique reports on a function from a different CVE #, we compare the fuction name, 
# and timestamp.  If it's the same function, from an earlier timetsamp than the CVE was published, we consider it a TP.
# If it's the same function from a later timestamp, we consider it a FP.
# Any other case we report it as unknown and require manual investigation.
def generate_ground_truth(target_graphs):
    NUM_VULN=0
    NUM_PATCH=0
    NUM_BEFORE=0
    NUM_AFTER=0
   
    for g in target_graphs:
        repo,cve,t,hash,f,_,_=g['path'].split('/')[-7:]
        func=g['base_name'] 
        
        
        if t == 'vuln':
            NUM_VULN += 1
        elif t == 'patch':
            NUM_PATCH += 1
        elif t == 'before':
            NUM_BEFORE += 1
        elif t == 'after':
            NUM_AFTER += 1


    print("Done! Ground truth stats:")
    print("NUM_VULN: %d" % NUM_VULN)
    print("NUM_PATCH: %d" % NUM_PATCH)
    print("NUM_BEFORE: %d" % NUM_BEFORE)
    print("NUM_AFTER: %d" % NUM_AFTER)
    print("TOT_VULN: %d" % (NUM_VULN + NUM_BEFORE))
    print("TOT_NOT_VULN: %d" % (NUM_PATCH + NUM_AFTER))
    return NUM_VULN, NUM_PATCH, NUM_BEFORE, NUM_AFTER

def get_hits_multi(vgraph_db, target_db):
    skipped=0
    #submitted=0
    # start our worker procs
    #in_queue = Queue() # one input queue
    #out_queue = Queue()
    #workers = []
    #for i in range(4):
    #    oq = Queue()
    #    p = Process(target=consumer, args=(in_queue, oq,))
    #    p.daemon = True
    #    p.start()
    #    workers.append((p,oq))
    print("Filtering based on characteristic vector...")
    work=[]
    pbar = tqdm(total=len(target_db)*len(vgraph_db))
    for i, tg in enumerate(target_db):
        t_trips = tg['triples']
        t_vec = np.array(tg['vec'])
        tg['hits'] = [] # place to put hits
        for vg in vgraph_db:
            vg_vec = np.array(vg['vec']) 
            # simple mean squared error calculation as a filter
            if np.square(vg_vec-t_vec).mean() < 500.:
                work.append((i,vg, t_trips, MATCH_MODE)) 
                #in_queue.put((i,vg, t_trips, MATCH_MODE)) 
                #submitted += 1
            else:
                skipped+=1
            pbar.update(1)
    pbar.close()
    print("Work size: %d" % len(work))
    print("Applying pool...")
    p = Pool(NUM_PROCS)
    res = p.map(consume, work)
    print("done..")

    for r in res:
        (target_id, vg, cvg_score, pvg_score, nvg_score) = r
        if decision_function(cvg_score, pvg_score, nvg_score):
            target_db[target_id]['hits'].append(vg)


def get_hits(vgraph_db, target_db):
    skipped=0
    pbar = tqdm(total=len(target_db)*len(vgraph_db))
    for i, tg in enumerate(target_db):
        t_trips = tg['triples']
        t_vec = np.array(tg['vec'])
        tg['hits'] = [] # place to store our hits
        for vg in vgraph_db:
            vg_vec = np.array(vg['vec'])
            if np.square(vg_vec-t_vec).mean() < 500.:
                if MATCH_MODE is 'exact':
                    cvg_score, pvg_score, nvg_score = triplet_match_exact(vg, t_trips)
                else: # is 'approx'
                    cvg_score, pvg_score, nvg_score = triplet_match_approx(vg, t_trips)
                #if cvg_score > CVG_THRESH and pvg_score > PVG_THRESH and nvg_score < NVG_THRESH:
                # .. what happens if we try just pvg/nvg?
                if decision_functioN(cvg_score, pvg_score, nvg_score):
                    # we have a hit
                    tg['hits'].append(vg)
            else:
                skipped+=1
            pbar.update(1)
    pbar.close()
    print("Num skipped from graph vector filter: ", skipped)


def eval_vgraph(vgraph_db, target_db, gt):
    # Loop through target graphs.  Get any vGraph hits and evaluate for truthiness
    # Now we score
    print("Scoring results...")

    # Score all:
    TP=0.
    FP=0.
    TN=0.
    FN=0.
    UNK=0
    for tg in target_db:
        if len(tg['hits']) == 0: # nothing hit on this target
            if '/patch/' in tg['path'] or '/after/' in tg['path']:
                TN +=1
            else: # something should have hit
                FN += 1
        else:
            # something hit
            for vg in tg['hits']:
                if vg['cve'] in tg['path'] and ('/vuln/' in tg['path'] or '/before/' in tg['path']):
                    TP += 1
                elif vg['cve'] in tg['path'] and ('/patch/' in tg['path'] or '/after/' in tg['path']):
                    FP += 1
                else:
                    # Need to check
                    tg_name = tg['path'].split('/')[-1][:-len('.gpickle')]
                    vg_name = vg['func']
                    tg_time = int(tg['path'].split('/')[-4].split('_')[1])
                    vg_time = int(vg['hsh'].split('_')[1])
                    if tg_name == vg_name and tg_time < vg_time:
                        TP += 1
                    else:
                        UNK += 1
                        if(PRINT_UNK):
                            print("UNK: %s %s/%s/%s/%s/%s)" % (tg['path'], vg['repo'], vg['cve'], vg['hsh'],vg['file'], vg['func']))

             
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2*(P*R)/(P+R)
    print("All Score:")
    print("TP\tFP\tTN\tFN\tUNK\tP\tR\tF1")   
    print("%d\t%d\t%d\t%d\t%d\t%02f\t%02f\t%02f"%(TP,FP,TN,FN,UNK,P,R,F1))

    # Score train:
    TP=0.
    FP=0.
    TN=0.
    FN=0.
    UNK=0
    for tg in target_db:
        if not ('/vuln/' in tg['path'] or '/patch/' in tg['path']):
            continue # only vuln/patch
        if len(tg['hits']) == 0: # nothing hit on this target
            if '/patch/' in tg['path'] or '/after/' in tg['path']:
                TN +=1
            else: # something should have hit
                FN += 1
        else:
            # something hit
            for vg in tg['hits']:
                if vg['cve'] in tg['path'] and ('/vuln/' in tg['path'] or '/before/' in tg['path']):
                    TP += 1
                elif vg['cve'] in tg['path'] and ('/patch/' in tg['path'] or '/after/' in tg['path']):
                    FP += 1
                else:
                    # Need to check
                    tg_name = tg['path'].split('/')[-1][:-len('.gpickle')]
                    vg_name = vg['func']
                    tg_time = int(tg['path'].split('/')[-4].split('_')[1])
                    vg_time = int(vg['hsh'].split('_')[1])
                    if tg_name == vg_name and tg_time < vg_time:
                        TP += 1
                    else:
                        UNK += 1
                        if(PRINT_UNK):
                            print("UNK: %s %s/%s/%s/%s/%s)" % (tg['path'], vg['repo'], vg['cve'], vg['hsh'],vg['file'], vg['func']))

    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2*(P*R)/(P+R)
    print("Train Score:")
    print("TP\tFP\tTN\tFN\tUNK\tP\tR\tF1")   
    print("%d\t%d\t%d\t%d\t%d\t%02f\t%02f\t%02f"%(TP,FP,TN,FN,UNK,P,R,F1))

    # score test
    TP=0.
    FP=0.
    TN=0.
    FN=0.
    UNK=0
    for tg in target_db:
        if not ('/before/' in tg['path'] or '/after/' in tg['path']):
            continue # only before/after
        if len(tg['hits']) == 0: # nothing hit on this target
            if '/patch/' in tg['path'] or '/after/' in tg['path']:
                TN +=1
            else: # something should have hit
                FN += 1
        else:
            # something hit
            for vg in tg['hits']:
                if vg['cve'] in tg['path'] and ('/vuln/' in tg['path'] or '/before/' in tg['path']):
                    TP += 1
                elif vg['cve'] in tg['path'] and ('/patch/' in tg['path'] or '/after/' in tg['path']):
                    FP += 1
                else:
                    # Need to check
                    tg_name = tg['path'].split('/')[-1][:-len('.gpickle')]
                    vg_name = vg['func']
                    tg_time = int(tg['path'].split('/')[-4].split('_')[1])
                    vg_time = int(vg['hsh'].split('_')[1])
                    if tg_name == vg_name and tg_time < vg_time:
                        TP += 1
                    else:
                        UNK += 1
                        if(PRINT_UNK):
                            print("UNK: %s %s/%s/%s/%s/%s)" % (tg['path'], vg['repo'], vg['cve'], vg['hsh'],vg['file'], vg['func']))


    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2*(P*R)/(P+R)
    print("Test Score:")
    print("TP\tFP\tTN\tFN\tUNK\tP\tR\tF1")   
    print("%d\t%d\t%d\t%d\t%d\t%02f\t%02f\t%02f"%(TP,FP,TN,FN,UNK,P,R,F1))
  
    # score test modified
    TP=0.
    FP=0.
    TN=0.
    FN=0.
    UNK=0
    for tg in target_db:
        if not ('/before/' in tg['path'] or '/after/' in tg['path']):
            continue # only before/after

        #print(tg['path'])
        #data/vuln_patch_graph_db/ffmpeg/CVE-2013-0868/after/2793b218bdd59db51f1f5c960c508fea7190310e_1407122995/huffyuvdec.c/graph/decode_frame.gpickle
        d = '/'.join(tg['path'].split('/')[0:4]) # root/repo/CVE
        func = tg['path'].split('/')[-1] # function.gpickle

        if '/before/' in tg['path']: # should be vuln
            is_same = False
            for (root,dirs,files) in os.walk(d):
                if func in files and '/vuln/' in root:
                   # This is orig vuln file.  lets check for differences
                   before_src = tg['path'].replace('/graph/', '/code/')
                   before_src = before_src.replace('.gpickle','.c')
                   vuln_src = (root + '/' + func).replace('/graph/','/code/')
                   vuln_src = vuln_src.replace('.gpickle','.c')
                   if filecmp.cmp(before_src, vuln_src):
                       # found match
                       is_same = True
                       break
            if is_same:
                continue # skip this one
        else: # after patch so should be patched
            is_same = False
            for (root,dirs,files) in os.walk(d):
                if func in files and '/patch/' in root:
                   # This is orig vuln file.  lets check for differences
                   after_src = tg['path'].replace('/graph/', '/code/')
                   after_src = after_src.replace('.gpickle','.c')
                   patch_src = (root + '/' + func).replace('/graph/','/code/')
                   patch_src = patch_src.replace('.gpickle','.c')
                   if filecmp.cmp(after_src, patch_src):
                       is_same = True
                       break
            if is_same:
                continue # skip this

        # If we make it here, this is either a before/after target graph which has
        # source code thats modified from the vuln/patch file used to generate vGraph

        # score it
        if len(tg['hits']) == 0: # nothing hit on this target
            if '/patch/' in tg['path'] or '/after/' in tg['path']:
                TN +=1
            else: # something should have hit
                FN += 1
        else:
            # something hit
            for vg in tg['hits']:
                if vg['cve'] in tg['path'] and ('/vuln/' in tg['path'] or '/before/' in tg['path']):
                    TP += 1
                elif vg['cve'] in tg['path'] and ('/patch/' in tg['path'] or '/after/' in tg['path']):
                    FP += 1
                else:
                    # Need to check
                    tg_name = tg['path'].split('/')[-1][:-len('.gpickle')]
                    vg_name = vg['func']
                    tg_time = int(tg['path'].split('/')[-4].split('_')[1])
                    vg_time = int(vg['hsh'].split('_')[1])
                    if tg_name == vg_name and tg_time < vg_time:
                        TP += 1
                    else:
                        UNK += 1
                        if(PRINT_UNK):
                            print("UNK: %s %s/%s/%s/%s/%s)" % (tg['path'], vg['repo'], vg['cve'], vg['hsh'],vg['file'], vg['func']))



    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2*(P*R)/(P+R)
    print("Test Modified")
    print("TP\tFP\tTN\tFN\tUNK\tP\tR\tF1")
    print("%d\t%d\t%d\t%d\t%d\t%02f\t%02f\t%02f"%(TP,FP,TN,FN,UNK,P,R,F1))




PRINT_UNK=True # used for debugging of unkown results
CVG_THRESH=50 # not currently matching CVG.  only used for characeristic vector
PVG_THRESH=50
NVG_THRESH=50 # don't use this either.  just check pvg > nvg
NUM_PROCS=40
MATCH_MODE='exact'
#MATCH_MODE='approx'

vgraph_db = load_vgraph_db('data/vgraph_db')
target_db = load_target_db('data/vuln_patch_graph_db')
gt = generate_ground_truth(target_db)
exit()
start_time = time.time()
get_hits_multi(vgraph_db, target_db)
elapsed = time.time() - start_time
print("Time to get hits: ", elapsed)
# Great, now we can actually go about evaluating the different approaches.
eval_vgraph(vgraph_db, target_db, gt)

