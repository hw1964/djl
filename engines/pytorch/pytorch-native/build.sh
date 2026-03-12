#!/usr/bin/env bash
set -ex

WORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export WORK_DIR

NUM_PROC=1
if [[ -n $(command -v nproc) ]]; then
    NUM_PROC=$(nproc)
elif [[ -n $(command -v sysctl) ]]; then
    NUM_PROC=$(sysctl -n hw.ncpu)
fi

PLATFORM=$(uname | tr '[:upper:]' '[:lower:]')

VERSION=$1
FLAVOR=$2
AARCH64_CXX11ABI="-cxx11"
CXX11ABI="-cxx11-abi"
if [[ $3 == "precxx11" ]]; then
    CXX11ABI=""
    AARCH64_CXX11ABI=""
fi
ARCH=$4

# Download or build libtorch
if [[ ! -d "libtorch" ]]; then
    if [[ $PLATFORM == 'linux' ]]; then
        if [[ ! "$FLAVOR" =~ ^(cpu|cu117|cu121|cu124|cu128|cu131)$ ]]; then
            echo "$FLAVOR is not supported."
            exit 1
        fi
        if [[ $ARCH == 'aarch64' && "$FLAVOR" = cu* ]]; then
            # Build from source for aarch64 CUDA
            rm -rf pytorch
            git clone --recursive https://github.com/pytorch/pytorch.git pytorch
            cd pytorch
            git checkout v${VERSION}
            git submodule sync
            git submodule update --init --recursive --jobs 8

            # --- PATCHES (only for CUDA 13.x) ---
            # PyTorch 2.7.1 + CUDA 12.8 is an officially supported combination
            # and needs no patches. Patches 1-3 are only needed for CUDA 13.x
            # where cuFFT enums were removed and cudaGraphNodeGetDependentNodes
            # changed to a 4-arg signature.
            CUDA_MAJOR=$("${CUDA_HOME:-/usr/local/cuda}"/bin/nvcc --version | grep -oP 'release \K[0-9]+')
            if [[ "$CUDA_MAJOR" -ge 13 ]]; then
                echo "=== Applying CUDA 13.x patches ==="

                # PATCH 1: Add cufft.h include
                sed -i '1i #include <cufft.h>' aten/src/ATen/native/cuda/CuFFTUtils.h

                # PATCH 2: Remove cuFFT enums dropped in CUDA 13.x
                sed -i '/case CUFFT_INCOMPLETE_PARAMETER_LIST:/,/return "CUFFT_INCOMPLETE_PARAMETER_LIST";/d' \
                    aten/src/ATen/native/cuda/CuFFTUtils.h
                sed -i '/case CUFFT_PARSE_ERROR:/,/return "CUFFT_PARSE_ERROR";/d' \
                    aten/src/ATen/native/cuda/CuFFTUtils.h
                sed -i '/case CUFFT_LICENSE_ERROR:/,/return "CUFFT_LICENSE_ERROR";/d' \
                    aten/src/ATen/native/cuda/CuFFTUtils.h

                # PATCH 3: Fix cudaGraphNodeGetDependentNodes 4-arg signature
                python3 -c "
import re, sys
path = 'third_party/cudnn_frontend/include/cudnn_frontend_shim.h'
with open(path, 'r') as f:
    content = f.read()
patched = re.sub(
    r'(cudaGraphNodeGetDependentNodes,\s*node,\s*pDependentNodes,)\s*(pNumDependentNodes\))',
    r'\1 nullptr, \2',
    content
)
if patched == content:
    print('⚠️  Patch 3 SKIPPED - pattern not found', file=sys.stderr)
else:
    with open(path, 'w') as f:
        f.write(patched)
    print('✅ Patch 3 OK')
"
            else
                echo "=== CUDA $CUDA_MAJOR detected — no patches needed ==="
            fi

            # --- CUDA 12.8 toolkit ---
            # We use CUDA 12.8 (the latest version officially supported by PyTorch 2.7.x)
            # even though the NGC container ships CUDA 13.1. CUDA 13.1 bundles CCCL 3.x
            # which removed cub::TransformInputIterator and other APIs that PyTorch 2.7.x
            # depends on. The host driver (13.x) is backwards compatible with 12.8-built
            # binaries at runtime.
            export CUDA_HOME=/usr/local/cuda-12.8
            export PATH=${CUDA_HOME}/bin:${PATH}
            export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

            CUB_DIR=${CUDA_HOME}/targets/sbsa-linux/include

            echo "=== CUB verification ==="
            ls "${CUB_DIR}/cub/cub.cuh" \
                && echo "✅ CUB found at ${CUB_DIR}" \
                || { echo "❌ CUB not found at ${CUB_DIR} — check CUDA 12.8 installation"; exit 1; }
            export TORCH_CUDA_ARCH_LIST="9.0;12.0+PTX"
            export MAX_JOBS=$NUM_PROC
            export USE_CUDA=1
            export USE_CUDNN=1
            export BUILD_CAFFE2=1

            apt-get update
            apt-get install -y libopenblas-dev libblas-dev liblapack-dev gfortran ninja-build

            pip install -r requirements.txt

            mkdir -p ../build_libtorch
            cd ../build_libtorch

            cmake -G Ninja ../pytorch \
                -DCMAKE_BUILD_TYPE=Release \
                -DPYTHON_EXECUTABLE=$(which python3) \
                -DBUILD_PYTHON=OFF \
                -DBUILD_SHARED_LIBS=ON \
                -DUSE_CUDA=ON \
                -DUSE_CUDNN=ON \
                -DUSE_MKLDNN=ON \
                -DUSE_OPENMP=ON \
                -DUSE_FBGEMM=OFF \
                -DUSE_KINETO=OFF \
                -DUSE_NCCL=OFF \
                -DUSE_TENSORPIPE=OFF \
                -DUSE_GLOO=OFF \
                -DUSE_MPI=OFF \
                -DUSE_SYSTEM_NCCL=OFF \
                -DUSE_ROCM=OFF \
                -DUSE_OPENCV=OFF \
                -DUSE_FFMPEG=OFF \
                -DUSE_LEVELDB=OFF \
                -DUSE_LMDB=OFF \
                -DUSE_NUMPY=OFF \
                -DUSE_ROS=OFF \
                -DUSE_ZSTD=OFF \
                -DUSE_QNNPACK=OFF \
                -DUSE_PYTORCH_QNNPACK=OFF \
                -DUSE_XNNPACK=OFF \
                -DUSE_VALGRIND=OFF \
                -DUSE_TBB=OFF \
                -DUSE_NUMA=OFF \
                -DUSE_DEPLOY=OFF \
                -DUSE_NNPACK=OFF \
                -DBLAS=OpenBLAS \
                -DUSE_OPENCL=OFF \
                -DUSE_RPC=OFF \
                -DUSE_TENSORRT=OFF \
                -DUSE_ARM=ON \
                -DCUB_INCLUDE_DIR="${CUB_DIR}" \
                -DCMAKE_INSTALL_PREFIX=../libtorch
            # CUB points to CUDA 13.1 CCCL — same version Thrust uses, no conflict.
            # Do NOT replace with a standalone cloned CUB.

            ninja -j $NUM_PROC
            ninja install
            cd ..
            rm -rf build_libtorch

        elif [[ $ARCH == 'aarch64' ]]; then
            if [[ "$VERSION" =~ ^(2.[7-9].*)$ ]]; then
                curl -s "https://djl-ai.s3.amazonaws.com/publish/pytorch/${VERSION}/libtorch-linux-aarch64-${VERSION}.zip" | jar xv >/dev/null
            else
                curl -s "https://djl-ai.s3.amazonaws.com/publish/pytorch/${VERSION}/libtorch${AARCH64_CXX11ABI}-shared-with-deps-${VERSION}-aarch64.zip" | jar xv >/dev/null
            fi
        else
            curl -s "https://download.pytorch.org/libtorch/${FLAVOR}/libtorch${CXX11ABI}-shared-with-deps-${VERSION}%2B${FLAVOR}.zip" | jar xv >/dev/null
        fi
    elif [[ $PLATFORM == 'darwin' ]]; then
        if [[ "$VERSION" =~ ^(2.[2-9].*)$ ]]; then
            if [[ $ARCH == 'aarch64' ]]; then
                curl -s "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${VERSION}.zip" | jar xv >/dev/null
            else
                curl -s "https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-${VERSION}.zip" | jar xv >/dev/null
            fi
        else
            if [[ $ARCH == 'aarch64' ]]; then
                curl -s "https://djl-ai.s3.amazonaws.com/publish/pytorch/${VERSION}/libtorch-macos-${VERSION}-aarch64.zip" | jar xv >/dev/null
            else
                curl -s "https://download.pytorch.org/libtorch/cpu/libtorch-macos-${VERSION}.zip" | jar xv >/dev/null
            fi
        fi
    else
        echo "$PLATFORM is not supported."
        exit 1
    fi
fi

if [[ "$VERSION" == "1.13.1" || "$VERSION" == "2.0.1" || "$VERSION" =~ ^(2.[1-6].*)$ ]]; then
    # V1_13_X enables c10/util/variant.h which was removed in PyTorch 2.7+
    PT_VERSION=V1_13_X
fi

if [[ "$FLAVOR" = cu* ]]; then
    USE_CUDA=1
fi

pushd .
rm -rf build
mkdir build && cd build
mkdir classes
javac -sourcepath ../../pytorch-engine/src/main/java/ \
    ../../pytorch-engine/src/main/java/ai/djl/pytorch/jni/PyTorchLibrary.java \
    -h include -d classes

if [[ -d ../libtorch ]]; then
    CMAKE_PREFIX_PATH=../libtorch
else
    CMAKE_PREFIX_PATH=/usr/local/lib/python3.12/dist-packages/torch
fi

cmake -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
    -DPT_VERSION="${PT_VERSION}" \
    -DUSE_CUDA="$USE_CUDA" ..

cmake --build . --config Release -- -j "${NUM_PROC}"

if [[ "$FLAVOR" = cu* ]]; then
    # avoid link with libcudart.so.11.0
    sed -i -r "s|/usr/local/cuda(.{5})?/lib64/lib(cudart|nvrtc).so||g" CMakeFiles/djl_torch.dir/link.txt
    rm libdjl_torch.so
    . CMakeFiles/djl_torch.dir/link.txt
fi

if [[ $PLATFORM == 'darwin' ]]; then
    install_name_tool -add_rpath @loader_path libdjl_torch.dylib
fi
popd
