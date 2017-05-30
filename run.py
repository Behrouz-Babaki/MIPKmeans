#!/usr/bin/env python3.5
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

from __future__ import print_function
import sys
from mipkmeans import mipkmeans, l2_distance
import argparse

def read_data(datafile):
    data = []
    with open(datafile, 'r') as f:
        for line in f:
            line = line.strip()
            if line != '':
                d = [float(i) for i in line.split()]
                data.append(d)
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

def run(datafile, consfile, k, n_rep, 
        init, conv, lazy,
        max_iter, tolerance):
    data = read_data(datafile)
    ml, cl = read_constraints(consfile)
    
    best_clusters = None
    best_score = None    
    for i in range(n_rep):
        clusters, centers = mipkmeans(data, k, ml, cl,
                                    initialization=init,
                                    convergence_test=conv,
                                    constraint_laziness=lazy,
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
    parser.add_argument('--ofile', help='file to store the output', default=None)
    parser.add_argument('--sfile', help='file to which to append the summary', default=None)
    parser.add_argument('--n_rep', help='number of times to repeat the algorithm', 
                        default=10, type=int)
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

    clusters, score = run(args.dfile, args.cfile, 
                          args.k, args.n_rep,
                          args.init, args.convergence,
                          args.constraint_laziness,
                          args.m_iter, args.tol)

    if args.ofile is not None and clusters is not None:
        with open(args.ofile, 'w') as f:
            for cluster in clusters:
                f.write('%d\n' %cluster)
                
    if args.sfile is not None:
        with open(args.sfile, 'a') as f:
            if score is not None:
                print('%s %s %d %f'%(args.dfile, args.cfile, args.k, score), file=f)
            else:
                print('%s %s %d None'%(args.dfile, args.cfile, args.k), file=f)
              
    

