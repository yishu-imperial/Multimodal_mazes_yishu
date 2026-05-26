#!/bin/bash
# Submit 50 × 2 array jobs = 300 agents total
# Each array job has 3 subjobs (random, small_world, modular)

for i in {1..50}
do
    qsub scripts/job_array_conserve.pbs
    qsub scripts/job_array_no_conserve.pbs
done
