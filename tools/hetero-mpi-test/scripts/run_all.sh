#!/bin/bash
# run_all.sh - Run all hetero-mpi tests

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

echo "========================================="
echo "  Running All Hetero-MPI Tests"
echo "========================================="
echo

# 1. SendRecv Test
echo ">>> [1/3] Running SendRecv Test..."
echo "==========================================="
cd "$PROJECT_ROOT/tests/sendrecv"
./run_hetero.sh
echo
echo "SendRecv Test Complete!"
echo

# 2. AllReduce Test
echo ">>> [2/3] Running AllReduce Test..."
echo "==========================================="
cd "$PROJECT_ROOT/tests/collective"
./run_hetero_allreduce.sh
echo
echo "AllReduce Test Complete!"
echo

# 3. AllGather Test
echo ">>> [3/3] Running AllGather Test..."
echo "==========================================="
cd "$PROJECT_ROOT/tests/collective"
./run_hetero_allgather.sh
echo
echo "AllGather Test Complete!"
echo

echo "========================================="
echo "  All Tests Completed Successfully!"
echo "========================================="
