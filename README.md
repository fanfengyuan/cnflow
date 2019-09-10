# CNFLOW
A pipeline for best cambricon models.

## Usage

### 1. compile
```
root@localhost:/share/projects/github/cnflow# make
```
Dependent library: cnrt, opencv, glog

### 2. run
```
root@localhost:/share/projects/github/cnflow# mkdir -p bin lib


root@localhost:/share/projects/github/cnflow# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/lib/

root@localhost:/share/projects/github/cnflow# ./bin/test_flow
sec: 8847065 us
qps: 1130.318360
```

## Build your own pipeline
- Modify cnflow.cpp and cnflow.h to build your own pipeline.
