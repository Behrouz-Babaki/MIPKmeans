# MIPKmeans
A python implementation of MIP-Kmeans algorithm

## Usage

```
usage: run.py [-h] [--ofile OFILE] [--sfile SFILE] [--n_rep N_REP]
              [--init {random,kmpp}] [--convergence {label,shift}]
              [--m_iter M_ITER] [--tol TOL] [--constraint_laziness {0,1,2,3}]
              dfile cfile k

Run MIP-Kmeans algorithm

positional arguments:
  dfile                 data file
  cfile                 constraint file
  k                     number of clusters

optional arguments:
  -h, --help            show this help message and exit
  --ofile OFILE         file to store the output
  --sfile SFILE         file to which to append the summary
  --n_rep N_REP         number of times to repeat the algorithm
  --init {random,kmpp}  initialization method
  --convergence {label,shift}
                        convergence criterion
  --m_iter M_ITER       maximum number of iterations of the main loop
  --tol TOL             tolerance for deciding on convergence
  --constraint_laziness {0,1,2,3}
                        whether to first add the constraints or only if they
                        are violated
```

To see a run of the algorithm on example data and constraints, run the script `test.sh`.

## Dependencise

The program uses [Gurbio](http://www.gurobi.com/) solver to solve the *assignment subproblem*. You should also install the python interface to gurobi (the `gurobipy` module).  