#!/bin/bash

cores=100

rm -rf data/multiprocessing/*

for (( c=0; c<$cores; c++))
do
	python mr_prho_samplers.py $c &
done

python monitor.py 100

python combine.py "still-not-in-use"

