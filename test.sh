#!/bin/bash

n=100
k=3

output_dir="test_output"

if [ ! -d ${output_dir} ];
then
    mkdir ${output_dir}
fi

cmd="./run.py data/datasets/Iris.txt data/constraints/Iris/1.25.25.txt ${k} --n_rep ${n} --labeled --measure ALL --sfile ${output_dir}/iris.1.25.25.out"

${cmd}



