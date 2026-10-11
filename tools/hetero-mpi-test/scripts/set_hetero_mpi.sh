#!/bin/bash

export LD_LIBRARY_PATH=$HOME/lib/ucx/build_het/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/lib/openmpi/build_het/lib:$LD_LIBRARY_PATH 
export PATH=$HOME/lib/openmpi/build/bin:$PATH
export LD_LIBRARY_PATH=$HOME/lib/missing_libs:$LD_LIBRARY_PATH