#!/bin/bash

## Specifies the interpreting shell for the job.
#$ -S /bin/bash

## Specifies that all environment variables active within the qsub utility be exported to the context of the job.
#$ -V

## Specifies the parallel environment
#$ -pe smp 4

## Execute the job from the current working directory.
#$ -cwd 

## The  name  of  the  job.
#$ -N OMP_prac

##send an email when the job ends
#$ -m e

##email addrees notification
#$ -M XXXXXX@alumnes.udl.cat

##Passes an environment variable to the job
#$ -v OMP_NUM_THREADS=4

## Join outputs
#$ -j y

## In this line you have to write the command that will execute your application.
mpiexec -np 4 ./convo images/im04.ppm kernel/kernel25x25_random.txt ex.ppm 1 > ex.txt

