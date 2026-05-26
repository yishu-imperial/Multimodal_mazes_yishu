#!/bin/bash
# Submit 50 × 2 array jobs = 300 agents total (Task Msc)
# Each array job has 3 subjobs (random, small_world, modular)

for i in {1..50}
do
    qsub scripts/job_array_msc_conserve.pbs
    qsub scripts/job_array_msc_no_conserve.pbs
done
