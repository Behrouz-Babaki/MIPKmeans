# coding: utf-8

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

from __future__ import division
import numpy as np
from gurobipy import Model, GRB, LinExpr
from collections import OrderedDict
import random

def l2_distance(a, b):
    return np.square(np.linalg.norm(a-b))

class ccmodel(object):
    def __init__(self, data, k, 
                 ml, cl,
                 lb=None, 
                 ub=None,
                 lazy=0,
                 timeout=None, 
                 verbosity=0):
        self.timeout = timeout
        self.lazy = lazy
        self.data = data
        self.ml = ml
        self.cl = cl
        self.lb = lb
        self.ub = ub
        self.n = len(data)
        self.k = k
        
        self.model = Model('cc')
        self.model.params.OutputFlag = verbosity
        self.model.params.UpdateMode = 1
        
        self.vars = dict()
        self.create_model()
                
    def create_model(self):
        # create variables
        for i in range(self.n):
            for j in range(self.k):
                self.vars[(i, j)] = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
        
        # create the constraints
        # Each instance is assigned to exactly one cluster
        for i in range(self.n):
            expr = LinExpr([1]*self.k, [self.vars[(i, j)] for j in range(self.k)])
            self.model.addConstr(expr, GRB.EQUAL, 1.0, 'c%d'%i)
        
        # Each cluster contains at least one instance
        if self.lb is None:
            for j in range(self.k):
                expr = LinExpr([1]*self.n, [self.vars[(i, j)] for i in range(self.n)])
                c = self.model.addConstr(expr, GRB.GREATER_EQUAL, 1.0, 's%d'%i)
                c.lazy = self.lazy
        
        # must-link constraints
        for i, (first, second) in enumerate(self.ml):
            for j in range(self.k):
                c = self.model.addConstr(self.vars[(first, j)] == self.vars[(second, j)], 'ml%d:%d'%(i, j))
                c.lazy = self.lazy
        
        # can-not-link constraints
        for i , (first, second) in enumerate(self.cl):
            for j in range(self.k):
                c = self.model.addConstr(self.vars[(first, j)] + self.vars[(second, j)] <= 1.0, 'cl%d:%d'%(i, j))
                c.lazy = self.lazy
                
        # cardinality constraints
        if self.lb is not None or self.ub is not None:
            for j in range(self.k):
                expr = LinExpr([1]*self.n, [self.vars[(i, j)] for i in range(self.n)])
                if self.lb is not None:
                    c = self.model.addConstr(expr, GRB.GREATER_EQUAL, self.lb, 'lb%d'%j)
                    c.lazy = self.lazy
                if self.ub is not None:
                    c = self.model.addConstr(expr, GRB.LESS_EQUAL, self.ub, 'ub%d'%j)
                    c.lazy = self.lazy
        
        self.model.update()
        
    def set_objective(self, centroids):
        distances = [[-1] * self.k for i in range(self.n)]
        for i in range(self.n):
            for j in range(self.k):
                distances[i][j] = l2_distance(self.data[i], centroids[j])        
        
        # create the objective function
        obj_coefs = []
        obj_vars = []
        for i in range(self.n):
            for j in range(self.k):
                obj_coefs.append(distances[i][j])
                obj_vars.append(self.vars[(i, j)])
        self.model.setObjective(LinExpr(obj_coefs, obj_vars), GRB.MINIMIZE)
        self.model.update()
        
    
    def solve(self, centroids):
        self.model.reset()
        self.set_objective(centroids)
        
        if self.timeout:
            self.model.TimeLimit = self.timeout
            
        self.model.optimize()
        
        self.status = self.model.Status
        self.opt_time = self.model.Runtime
        clusters = None
        if self.status == GRB.OPTIMAL:
            clusters= [-1 for i in range(self.n)]
            for i in range(self.n):
                for j in range(self.k):
                    if self.vars[(i, j)].x > 0.5:
                        clusters[i] = j
        return clusters

# taken from scikit-learn (https://goo.gl/1RYPP5)
def tolerance(tol, dataset):
    n = len(dataset)
    dim = len(dataset[0])
    averages = [sum(dataset[i][d] for i in range(n))/float(n) for d in range(dim)]
    variances = [sum((dataset[i][d]-averages[d])**2 for i in range(n))/float(n) for d in range(dim)]
    return tol * sum(variances) / dim
    
def closest_clusters(centers, datapoint):
    distances = np.square(np.linalg.norm(centers - datapoint, axis=1))
    return np.argsort(distances), distances

def initialize_centers(dataset, k, method='random'):
    if method == 'random':
        ids = list(range(len(dataset)))
        random.shuffle(ids)
        return [dataset[i] for i in ids[:k]]        
    
    elif method == 'kmpp':
        chances = [1] * len(dataset)
        centers = []
        
        for _ in range(k):
            chances = [x/sum(chances) for x in chances]        
            r = random.random()
            acc = 0.0
            for index, chance in enumerate(chances):
                if acc + chance >= r:
                    break
                acc += chance
            centers.append(dataset[index])
            
            for index, point in enumerate(dataset):
                cids, distances = closest_clusters(centers, point)
                chances[index] = distances[cids[0]]
        
        centers = np.array(centers)        
        return centers    

def compute_centers(clusters, dataset, k, canonical=False):
    if canonical:
        # canonical labeling of clusters
        ids = list(OrderedDict.fromkeys(clusters))
        c_to_id = dict()
        for j, c in enumerate(ids):
            c_to_id[c] = j
        for j, c in enumerate(clusters):
            clusters[j] = c_to_id[c]
     
    dim = len(dataset[0])
    centers = np.zeros((k, dim))
    counts = np.zeros(k)
    for j, c in enumerate(clusters):
        centers[c] += dataset[j]
        counts[c] += 1
    for j in range(k):
        centers[j] /= counts[j]
    return clusters, centers


def mipkmeans(dataset, k, 
              ml=[], cl=[],
              lb=None, ub=None, 
              initialization='kmpp',
              convergence_test='label',
              constraint_laziness=0,
              max_iter=300, tol=1e-4):

    mip_model = ccmodel(dataset, k, ml, cl, lb, ub,
                        lazy=constraint_laziness)
    tol = tolerance(tol, dataset)
    canonical_labeling = (convergence_test == 'label')
    centers = initialize_centers(dataset, k, method=initialization)
    clusters = [-1] * len(dataset)
    
    converged = False
    iter_counter = 0
    while not converged:
        # solve the assignment subproblem
        clusters_ = mip_model.solve(centers)
        if not clusters_:
            return None, None
            
        # compute the new centers
        clusters_, centers_ = compute_centers(clusters_, dataset, 
                                             k, canonical_labeling)

        # check the convergence                                             
        if max_iter is not None and iter_counter > max_iter:
            converged = True
        elif convergence_test == 'label':
            converged = True
            i = 0
            while converged and i < len(dataset):
                if clusters[i] != clusters_[i]:
                    converged = False
                i += 1
        elif convergence_test == 'shift':
            shift = np.sum(np.square(np.linalg.norm(centers - centers_)))
            if shift <= tol:
                converged = True
        
        # store the current values of clusters and centers   
        clusters = clusters_
        centers = centers_
        iter_counter += 1
        
    return clusters, centers
