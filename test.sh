#!/bin/bash

n=100
k=3

for i in $(ls data/iris/constraints/iris*cons); do
    echo "$i"
    cmd="./run.py data/iris/iris.data ${i} ${k} --n_rep ${n} --sfile ./iris.out"
    ${cmd}
done


