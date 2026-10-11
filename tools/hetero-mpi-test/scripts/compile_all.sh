#!/bin/bash
# compile_all.sh - Build all test binaries

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

echo "=== Building all hetero-mpi tests ==="
echo

echo ">>> Building SendRecv tests..."
cd "$PROJECT_ROOT/tests/sendrecv" && ./compile_separate.sh
echo

echo ">>> Building Collective tests..."
cd "$PROJECT_ROOT/tests/collective" && ./compile_separate.sh
echo

echo "=== All builds complete ==="
