#!/bin/bash

# Build script for odepack
# Usage: ./build.sh [OPTIONS]
#   -m, --mpreal     Enable MPFR multi-precision support
#   -d, --debug      Enable debug build
#   -j N             Number of parallel jobs (default: nproc)
#   -h, --help       Show this help message

set -e

# Defaults
MPREAL=OFF
DEBUG=OFF
JOBS=$(nproc)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mpreal)
            MPREAL=ON
            shift
            ;;
        -d|--debug)
            DEBUG=ON
            shift
            ;;
        -j)
            JOBS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./build.sh [OPTIONS]"
            echo "  -m, --mpreal     Enable MPFR multi-precision support"
            echo "  -d, --debug      Enable debug build"
            echo "  -j N             Number of parallel jobs (default: $(nproc))"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== odepack build ==="
echo "MPREAL: $MPREAL"
echo "DEBUG:  $DEBUG"
echo "JOBS:   $JOBS"
echo ""

# Clean up previous build
echo "Cleaning up previous build..."
rm -rf build/ dist/ *.egg-info/ _skbuild/ __pycache__/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.so" -delete 2>/dev/null || true

# Configure
echo "Configuring..."
cmake -B build \
    -DBUILD_PYTHON=ON \
    -DMPREAL=$MPREAL \
    -DDEBUG=$DEBUG

# Build
echo "Building with $JOBS jobs..."
cmake --build build -j "$JOBS"

echo ""
echo "=== Build complete ==="
echo "Python modules built in: build/python/"
