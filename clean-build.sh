#!/usr/bin/env bash
# Remove disposable build artifacts (~10GB)
# The JAR is in ~/.m2 and on GitHub — these are fully reproducible
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

rm -rf engines/pytorch/pytorch-native/pytorch
rm -rf engines/pytorch/pytorch-native/libtorch
rm -rf engines/pytorch/pytorch-native/build_libtorch
rm -rf engines/pytorch/pytorch-native/build

echo "Build artifacts cleaned"
