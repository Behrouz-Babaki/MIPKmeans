#!/bin/bash

n=100
k=3

output_dir="test_output"

if [ ! -d ${output_dir} ];
then
    mkdir ${output_dir}
fi

for i in $(ls data/iris/constraints/iris*cons); do
    n_cons=$(echo $i | grep -o -E '[0-9]+')
    echo ${n_cons}
    cmd="./run.py data/iris/iris.data ${i} ${k} --n_rep ${n} --sfile ${output_dir}/iris.${n_cons}.out"
    ${cmd}
done


