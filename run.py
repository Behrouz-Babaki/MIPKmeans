#!/usr/bin/python
# -*- coding: utf-8 -*-

#     Copyright (C) 2017  Behrouz Babaki
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, division
import numpy as np
from mipkmeans import mipkmeans, ccmodel, l2_distance
import argparse
import time

def read_data(datafile):
    data = []
    with open(datafile, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('%'): continue
            if line != '':
                d = [float(i) for i in line.split()]
                data.append(d)
    data = np.array(data)
    return data

def read_constraints(consfile):
    ml, cl = [], []
    with open(consfile, 'r') as f:
        for line in f:
            line = line.strip()
            if line != '':
                line = line.split()
                constraint = (int(line[0]), int(line[1]))
                c = int(line[2])
                if c == 1:
                    ml.append(constraint)
                if c == -1:
                    cl.append(constraint)
    return ml, cl

def run(data, ml, cl, 
        lb, ub, k, n_rep, 
        init, conv, laziness,
        max_iter, tolerance):

    mip_model = ccmodel(data, k, 
                        ml, cl, 
                        lb, ub,
                        laziness)    
                        
    best_clusters = None
    best_score = None
    for i in range(n_rep):
        clusters, centers = mipkmeans(data, k, 
                                      mip_model, 
                                      initialization=init,
                                      convergence_test=conv,
                                      max_iter=max_iter,
                                      tol=tolerance)
        if not clusters:
            return None, None
            
        score = sum(l2_distance(data[j], centers[clusters[j]]) 
                    for j in range(len(data)))
        if best_score is None or score < best_score:
            best_score = score
            best_clusters = clusters
            
    return best_clusters, best_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MIP-Kmeans algorithm')
    parser.add_argument('dfile', help='data file')
    parser.add_argument('cfile', help='constraint file')
    parser.add_argument('k', type=int, help='number of clusters')
    parser.add_argument('--labeled', help='the first column is class label', action='store_true')
    parser.add_argument('--measure', choices=('ARI', 'NMI', 'ALL'), default=None)
    parser.add_argument('--lb', type=int, help='upper-bound on cluster size', default=None)
    parser.add_argument('--ub', type=int, help='lower-bound on cluster size', default=None)
    parser.add_argument('--ofile', help='file to store the clustering', default=None)
    parser.add_argument('--sfile', help='file to store the summary', default=None)
    parser.add_argument('--n_rep', help='number of times to repeat the algorithm', 
                        default=1, type=int)
    parser.add_argument('--init', help='initialization method', 
                        choices=('random','kmpp'), default='kmpp')
    parser.add_argument('--convergence', help='convergence criterion', 
                        choices=('label', 'shift'), default='label')
    parser.add_argument('--m_iter', help='maximum number of iterations of the main loop', 
                        default=None, type=int)
    parser.add_argument('--tol', help='tolerance for deciding on convergence', 
                        default=1e-4, type=float)
    parser.add_argument('--constraint_laziness', 
                        help='whether to first add the constraints or only if they are violated',
                        choices=(0, 1, 2, 3), default=0, type=int)
    args = parser.parse_args()
    
    if args.measure is not None and not args.labeled:
        print('Class labels are needed for evaluation of clustering')
        exit(1)

    start_time = time.time()
    data = read_data(args.dfile)

    if args.labeled:
        labels = data[:, 0]
        data = data[:, 1:]        

    ml, cl = read_constraints(args.cfile)    
    clusters, score = run(data, ml, cl,
                          args.lb, args.ub,
                          args.k, args.n_rep,
                          args.init, args.convergence,
                          args.constraint_laziness,
                          args.m_iter, args.tol)
    runtime = time.time() - start_time

    if args.ofile is not None:
        with open(args.ofile, 'w') as f:
            if clusters is not None:
                for cluster in clusters:
                    f.write('%d\n' %cluster)
                    

    if args.measure is not None:        
        measure_names = []
        for name in ('ARI', 'NMI'):
            if args.measure == name or args.measure == 'ALL':
                measure_names.append(name)
                
        measures = dict()
        if clusters is None:
            measures = {name: None for name in measure_names}
        else:
            if 'ARI' in measure_names:
                from sklearn.metrics import adjusted_rand_score
                measures['ARI'] = adjusted_rand_score(clusters, labels)
            if 'NMI' in measure_names:
                from sklearn.metrics import normalized_mutual_info_score
                measures['NMI'] = normalized_mutual_info_score(clusters, labels)            
                
    if args.sfile is not None:
        with open(args.sfile, 'w') as f:
            print('objective: ', file=f, end='')
            if score is not None:
                print('%f'%score, file=f)
            else:
                print('None', file=f)
            print('runtime: %f'%runtime, file=f)
            print('average runtime: %f'%(runtime/args.n_rep), file=f)
            for measure in measures:
                if measures[measure] is None:
                    print('%s: None'%measure, file=f)
                else:
                    print('%s: %f'%(measure, measures[measure]), file=f)
            
              
    

