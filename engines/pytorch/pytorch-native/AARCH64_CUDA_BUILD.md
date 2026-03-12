# Building DJL PyTorch Native for linux-aarch64 + CUDA

Build instructions for DJL's PyTorch native JNI library on ARM64 Linux with CUDA GPU support.
Successfully tested on NVIDIA DGX Spark and GIGABYTE AI TOP Atom (Grace Blackwell GB10).

## Why?

Official DJL releases only ship pre-built PyTorch native libraries for:
- `linux-x86_64` (CPU and CUDA)
- `linux-aarch64` (CPU only)

There are **no official aarch64 + CUDA binaries**. This build compiles libtorch from source
on aarch64 and produces a JAR that enables GPU-accelerated PyTorch inference on ARM systems
with NVIDIA GPUs.

## Hardware tested

| Property | Value |
|---|---|
| System | NVIDIA DGX Spark / GIGABYTE AI TOP Atom |
| SoC | NVIDIA Grace Blackwell GB10 |
| CPU | ARM Neoverse V2 (aarch64) |
| GPU | NVIDIA Blackwell — compute capability 12.1 (sm_121) |
| OS | Ubuntu 24.04 LTS aarch64 |

## Prerequisites

- **Docker** with NVIDIA GPU support (`--gpus all`)
- **Base image**: `nvcr.io/nvidia/pytorch:25.12-py3` (or similar NGC PyTorch container)
- **CUDA 12.8 toolkit** (installed inside the container alongside the NGC container's CUDA 13.1)
- **OpenJDK 21** (`apt-get install -y openjdk-21-jdk-headless`)
- **Build tools**: cmake, ninja-build, libopenblas-dev

### Why CUDA 12.8, not 13.1?

PyTorch 2.7.1 officially supports CUDA 11.8, 12.6, and 12.8. The NGC container ships
CUDA 13.1, which bundles CCCL 3.x — this version removed `cub::TransformInputIterator`
and other APIs that PyTorch 2.7.x depends on.

CUDA 12.8 is installed alongside 13.1 and used for compilation. The host's CUDA 13.x driver
is backwards compatible with 12.8-built binaries at runtime.

```bash
# Install CUDA 12.8 toolkit (one-time, inside the container)
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y --no-install-recommends cuda-toolkit-12-8
```

## Build instructions

### Docker setup

```bash
docker run --gpus all -it \
  -v ~/djl:/build/djl \
  -v ~/.m2:/root/.m2 \
  nvcr.io/nvidia/pytorch:25.12-py3 bash
```

### Inside the container

```bash
cd /build/djl/engines/pytorch/pytorch-native

export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
export PATH=$JAVA_HOME/bin:$PATH

# Stage 1+2: Clone PyTorch, build libtorch from source, build JNI
# This takes several hours on first run (compiles ~2485 CUDA objects)
./gradlew clean compileJNI -Paarch64 -Pcuda=cu128 --no-configuration-cache

# Stage 3: Package into a JAR
./gradlew packageCustomAarch64Cuda -Pcuda=cu128 --no-configuration-cache

# Stage 4: Install to local Maven cache
DEST=~/.m2/repository/ai/djl/pytorch/pytorch-native-cu128/2.7.1
mkdir -p "$DEST"
cp build/libs/pytorch-native-cu128-2.7.1-linux-aarch64.jar "$DEST/"
# Also copy the POM
cat > "$DEST/pytorch-native-cu128-2.7.1.pom" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>ai.djl.pytorch</groupId>
  <artifactId>pytorch-native-cu128</artifactId>
  <version>2.7.1</version>
</project>
EOF
```

### What `build.sh` does for aarch64 + CUDA

1. Sets `CUDA_HOME=/usr/local/cuda-12.8` **first** (before any version detection)
2. Clones PyTorch v2.7.1 from source with all submodules
3. Applies **Patch A** (CUDA < 13): removes the extra `nullptr` arg from `cudaGraphNodeGetDependentNodes` in `cudnn_frontend_shim.h` — the bundled cudnn_frontend uses the 4-arg API but CUDA 12.8 declares the 3-arg version
4. Applies **Patches B/C** (CUDA 13+ only): cuFFT enum removals
5. Configures with CMake + Ninja, targeting `sm_90` and `sm_120+PTX` (Blackwell)
6. Compiles libtorch (~2485 objects, several hours)
7. Installs to `libtorch/`
8. Compiles the DJL JNI bridge (`libdjl_torch.so`) against the built libtorch

### Why `+PTX` in `TORCH_CUDA_ARCH_LIST`?

The GB10 GPU is compute capability sm_121. Without `+PTX`, the JAR contains only sm_120
SASS code. Inside the NGC Docker container this works (NGC provides a compat stack), but
on bare metal `cuda-compat-12-8` ships no libraries. The `+PTX` flag embeds PTX intermediate
code for sm_120, which the CUDA runtime JIT-compiles for sm_121 at first launch.

## Using the JAR

### Maven dependency (use a profile to avoid errors on non-aarch64 machines)

```xml
<profiles>
    <profile>
        <id>aarch64-cuda</id>
        <dependencies>
            <dependency>
                <groupId>ai.djl.pytorch</groupId>
                <artifactId>pytorch-native-cu128</artifactId>
                <version>2.7.1</version>
                <classifier>linux-aarch64</classifier>
                <scope>runtime</scope>
            </dependency>
        </dependencies>
    </profile>
</profiles>
```

Activate only on the aarch64+CUDA machine: `mvn exec:java -Paarch64-cuda ...`

### Running

```bash
rm -rf ~/.djl.ai/pytorch/
mvn exec:java -Paarch64-cuda \
  -Dexec.mainClass="your.MainClass" \
  -Dai.djl.default_engine=PyTorch
```

DJL auto-extracts all `.so` files from the JAR into `~/.djl.ai/pytorch/2.7.1-cu128-linux-aarch64/`
including `0.36.0-libdjl_torch.so` (renamed at package time to match DJL 0.36.0 from Maven Central).

## Running tests

The existing DJL test suite can verify the build. Set environment variables to point at
the custom-built libraries:

```bash
export PYTORCH_LIBRARY_PATH=/build/djl/engines/pytorch/pytorch-native/libtorch/lib
export PYTORCH_VERSION=2.7.1
export PYTORCH_FLAVOR=cu128

# Copy JNI lib to expected location
DJL_VERSION=0.36.0
mkdir -p engines/pytorch/pytorch-native/jnilib/${DJL_VERSION}/linux-aarch64/cu128
cp engines/pytorch/pytorch-native/build/libdjl_torch.so \
   engines/pytorch/pytorch-native/jnilib/${DJL_VERSION}/linux-aarch64/cu128/

# Also make it available on the classpath
mkdir -p engines/pytorch/pytorch-jni/build/classes/java/main/jnilib/linux-aarch64/cu128
cp engines/pytorch/pytorch-native/build/libdjl_torch.so \
   engines/pytorch/pytorch-jni/build/classes/java/main/jnilib/linux-aarch64/cu128/

# Run tests (from repo root)
./gradlew :engines:pytorch:pytorch-engine:test \
    --no-configuration-cache \
    --tests "ai.djl.pytorch.integration.PtNDArrayTest" \
    -x compileJNI
```

Expected output:
```
testStringTensor PASSED   (CPU - string tensor creation)
testLargeTensor  PASSED   (GPU - large tensor allocation on Blackwell)
```

## Known issues

- **JAVA_HOME**: On Ubuntu 24.04 aarch64, the path is `/usr/lib/jvm/java-21-openjdk-arm64` (not `aarch64`)
- **PT_VERSION flag**: Do not pass `-DPT_VERSION=V1_13_X` to cmake for PyTorch 2.7+. The `c10/util/variant.h` header was removed; `std::variant` is used instead.
- **Gradle config cache**: Use `--no-configuration-cache` for **both** `compileJNI` and `packageCustomAarch64Cuda`. Without it, Gradle reuses cached config and skips `build.sh` entirely.
- **CUDA 13.1**: PyTorch 2.7.1 does not support CUDA 13.x. Use 12.8.
- **CUDA_HOME ordering**: `CUDA_HOME=/usr/local/cuda-12.8` must be set **before** `nvcc --version` is called for CUDA_MAJOR detection. The default `/usr/local/cuda` symlink points to CUDA 13 in the NGC container.
- **cudnn_frontend nullptr patch**: The bundled cudnn_frontend uses a 4-arg `cudaGraphNodeGetDependentNodes`. CUDA 12.8 has the 3-arg version. Patch A removes the extra `nullptr` arg for CUDA < 13.
- **cmake relative path**: If `CMAKE_PREFIX_PATH=../libtorch` fails to find `TorchConfig.cmake`, use the absolute path: `/build/djl/engines/pytorch/pytorch-native/libtorch`.
- **Bare metal vs Docker**: The first build's JAR (without `+PTX`) only works inside the NGC Docker container. The current JAR (with `+PTX`) works on bare metal too.

## Build artifacts

The built JAR (`pytorch-native-cu128-2.7.1-linux-aarch64.jar`, ~488MB) is available as a
GitHub Release asset on this fork at:
`https://github.com/hw1964/djl/releases/tag/v2.7.1-aarch64-cu128`

It contains:

- `libc10.so`, `libc10_cuda.so` — PyTorch core libraries
- `libtorch.so`, `libtorch_cpu.so`, `libtorch_cuda.so` — PyTorch runtime
- `libcaffe2_nvrtc.so` — NVRTC integration
- `0.36.0-libdjl_torch.so` — DJL JNI bridge (renamed to match DJL 0.36.0)
- `pytorch.properties` — version/flavor metadata
