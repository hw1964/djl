# DJL PyTorch Native — linux-aarch64 CUDA Build
## Context Document for Claude Code CLI
**Hardware**: NVIDIA DGX Spark / GIGABYTE AI TOP ATOM (Grace Blackwell GB10, sm_90+)
**Goal**: Build Deep Java Library (DJL) with PyTorch Native support for `linux-aarch64` + CUDA
**Reason**: No official pre-built binaries exist for `linux-aarch64 + CUDA`. Official DJL only ships CPU aarch64 and CUDA x86_64.
**Updated**: March 2026

---

## 1. Hardware & Environment

| Property | Value |
|---|---|
| System | NVIDIA DGX Spark / GIGABYTE AI TOP ATOM |
| SoC | NVIDIA Grace Blackwell GB10 |
| CPU | ARM Neoverse (aarch64 / arm64) |
| GPU | Blackwell GB10 — CUDA compute capability 12.1 (sm_120) |
| OS | Ubuntu 24.04 LTS aarch64 |
| NGC container CUDA | 13.1 (from NGC container, NOT used for build) |
| Build CUDA | **12.8** (installed alongside 13.1, used for compilation) |
| Host CUDA driver | 13.0 (backwards compatible with 12.8 binaries) |
| Base Docker image | `nvcr.io/nvidia/pytorch:25.12-py3` |
| Java | OpenJDK 21 (`/usr/lib/jvm/java-21-openjdk-arm64`) |
| Python | 3.12 |
| Build tool | Ninja (preferred over make for parallelism) |
| CPU cores available | 20 (Grace CPU) |
| DJL version | 0.37.0-SNAPSHOT |
| PyTorch version | 2.7.1 |
| CUB version | 2.7.0 (bundled with CUDA 12.8) |

---

## 2. Key Decision: CUDA 12.8 instead of 13.1

The NGC container ships CUDA 13.1, but **PyTorch 2.7.1 only officially supports CUDA 11.8, 12.6, and 12.8**. CUDA 13.1 bundles CCCL 3.x which removed `cub::TransformInputIterator` and other APIs that PyTorch 2.7.x depends on.

**Solution**: Install CUDA 12.8 toolkit alongside 13.1 in the container and point the build at it. The host's CUDA 13.x driver is backwards compatible with 12.8-built binaries at runtime.

```bash
# Install CUDA 12.8 toolkit (one-time setup in container)
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y --no-install-recommends cuda-toolkit-12-8
```

CUDA 12.8 installs to `/usr/local/cuda-12.8/`. The build uses `CUDA_HOME=/usr/local/cuda-12.8`.

**With CUDA 12.8, PyTorch 2.7.1 compiles without ANY patches** — no source modifications needed.

---

## 3. Repository Structure

```
djl/                                         <- fork of https://github.com/hw1964/djl
└── engines/
    └── pytorch/
        └── pytorch-native/
            ├── build.sh                     <- main build script (MODIFIED)
            ├── build.gradle.kts             <- Gradle build (MODIFIED)
            ├── CMakeLists.txt               <- JNI CMake config (MODIFIED)
            ├── src/main/native/             <- DJL JNI C++ source
            ├── pytorch/                     <- cloned pytorch source (generated during build)
            ├── libtorch/                    <- installed libtorch (generated during build)
            └── build/                       <- JNI build directory (contains libdjl_torch.so)
```

**Git branch**: `feature/aarch64-cuda-blackwell`
**Fork**: `hw1964/djl`
**Base branch**: `master`

---

## 4. Build Strategy

1. **Clones PyTorch from source** (`github.com/pytorch/pytorch`) at v2.7.1
2. **Compiles libtorch** using CMake + Ninja with CUDA 12.8 targeting `sm_90` and `sm_120+PTX` (Blackwell)
3. **Compiles DJL JNI** (`libdjl_torch.so`) against the freshly built libtorch
4. **Packages a custom JAR** (`pytorch-native-cu128-2.7.1-linux-aarch64.jar`) via Gradle
5. **Installs to local Maven cache** for use in the Java project

---

## 5. Build Commands

### Full build from scratch
```bash
cd /build/djl/engines/pytorch/pytorch-native

export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
export PATH=$JAVA_HOME/bin:$PATH

# Stage 1+2: clone PyTorch, build libtorch, build JNI
./gradlew clean compileJNI -Paarch64 -Pcuda=cu128 --no-configuration-cache

# Stage 3: package JAR
./gradlew packageCustomAarch64Cuda -Pcuda=cu128 --no-configuration-cache

# Stage 4: install to local Maven
# (publishToMavenLocal doesn't cover custom task, so install manually)
DEST=~/.m2/repository/ai/djl/pytorch/pytorch-native-cu128/2.7.1
mkdir -p "$DEST"
cp build/libs/pytorch-native-cu128-2.7.1-linux-aarch64.jar "$DEST/"
```

### Resume after libtorch compile (if build.sh was interrupted)
```bash
# Resume ninja build
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
cd build_libtorch && ninja -j 20 && ninja install && cd ..

# Then build JNI manually
cd /build/djl/engines/pytorch/pytorch-native
rm -rf build && mkdir build && cd build && mkdir classes
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
export PATH=$JAVA_HOME/bin:/usr/local/cuda-12.8/bin:$PATH
javac -sourcepath ../../pytorch-engine/src/main/java/ \
    ../../pytorch-engine/src/main/java/ai/djl/pytorch/jni/PyTorchLibrary.java \
    -h include -d classes
cmake -DCMAKE_PREFIX_PATH=/build/djl/engines/pytorch/pytorch-native/libtorch \
    -DUSE_CUDA=1 ..
cmake --build . --config Release -- -j 20
```

---

## 6. Key Files & What Was Changed

### 6.1 `build.sh`

- CUDA 12.8 toolkit path: `CUDA_HOME=/usr/local/cuda-12.8`
- CUB from CUDA 12.8 standard path: `/usr/local/cuda-12.8/targets/sbsa-linux/include`
- Patches are **conditional on CUDA major version** — only applied for CUDA 13+
- `PT_VERSION=V1_13_X` only set for PyTorch <= 2.6.x (not 2.7+, because `c10/util/variant.h` was removed)

### 6.2 `build.gradle.kts`

- Custom task `packageCustomAarch64Cuda` packages libtorch + JNI into a JAR
- Sources `.so` files from `libtorch/lib/` and `build/libdjl_torch.so`
- Renames `libdjl_torch.so` → `{djlRuntimeVersion}-libdjl_torch.so` (default `0.36.0`) so DJL auto-resolves it without manual extraction. Override with `-PdjlRuntimeVersion=X.Y.Z` if needed.
- Default flavor changed to `cu128`

### 6.3 `CMakeLists.txt`

- `cmake_policy(SET CMP0146 OLD)` for FindCUDA deprecation
- JNI built without `-DPT_VERSION=V1_13_X` for PyTorch 2.7+ (avoids `c10/util/variant.h` issue)

---

## 7. Patches

**IMPORTANT**: `CUDA_HOME=/usr/local/cuda-12.8` must be exported **before** the `CUDA_MAJOR` detection so `nvcc --version` returns 12, not 13 (the NGC container default symlink). This is already correct in `build.sh`.

| # | Patch | File | Root Cause | When needed |
|---|---|---|---|---|
| A | Remove `nullptr` from `cudaGraphNodeGetDependentNodes` call | `cudnn_frontend_shim.h` | The bundled cudnn_frontend submodule uses the 4-arg API; CUDA 12.8 declares the 3-arg version | **CUDA < 13** |
| B | Add `#include <cufft.h>` | `CuFFTUtils.h` | Missing header | CUDA 13+ |
| C | Remove 3 dropped cuFFT enums | `CuFFTUtils.h` | Enums removed in CUDA 13.x | CUDA 13+ |

**Note**: CUDA 13.x also has the `TransformInputIterator` removal (CCCL 3.x). That would require additional patches or PyTorch 2.10+ which has native CCCL 3.x support.

---

## 8. Java Integration

The native JAR is only available in the local Maven repo on the aarch64+CUDA build machine.
Use a **Maven profile** so the dependency doesn't cause errors on other machines/IDEs.

**`pom.xml`** — add to `<profiles>` section:
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

On other machines, the profile is inactive and the dependency is ignored — no IDE errors.

**Run**:
```bash
rm -rf ~/.djl.ai/pytorch/
mvn clean
mvn exec:java -Paarch64-cuda \
  -Dexec.mainClass="your.MainClass" \
  -Dai.djl.default_engine=PyTorch \
  -Dai.djl.logging.level=debug
```

---

## 9. Docker Workflow

**Base image**: `nvcr.io/nvidia/pytorch:25.12-py3`

**Start script** (`start-docker.sh` on host):
```bash
#!/usr/bin/env bash
docker run --gpus all -it \
  -v ~/djl:/build/djl \
  -v /home/harald/.m2:/root/.m2 \
  -v /home/harald/.djl.ai:/root/.djl.ai \
  -v /home/harald/.local/bin/claude:/usr/local/bin/claude:ro \
  -v /home/harald/.claude:/root/.claude:ro \
  -v /home/harald/.claude.json:/root/.claude.json:ro \
  djl-build-blackwell:latest bash
```

**Second terminal into the same running container**:
```bash
docker exec -it $(docker ps -q --filter ancestor=djl-build-blackwell:latest) bash
```

---

## 10. Known Issues & Gotchas

- **JAVA_HOME**: Path is `/usr/lib/jvm/java-21-openjdk-arm64` (NOT `aarch64`)
- **OpenBLAS required** — MKL is x86 only; use `-DBLAS=OpenBLAS`
- **`USE_FBGEMM=OFF`** — FBGEMM is x86 only
- **`TORCH_CUDA_ARCH_LIST="9.0;12.0+PTX"`** — `12.0+PTX` includes PTX intermediate code so the CUDA runtime can JIT-compile for sm_121 (GB10) at first run. Without `+PTX`, sm_120 SASS runs inside Docker (NGC compat stack) but fails on bare metal where `cuda-compat-12-8` ships no libs.
- **PT_VERSION=V1_13_X must NOT be set for PyTorch 2.7+** — it enables `#include <c10/util/variant.h>` which was removed; `std::variant` is used instead
- **Gradle config cache** — use `--no-configuration-cache` for BOTH `compileJNI` and `packageCustomAarch64Cuda` tasks, otherwise Gradle reuses cached config and skips `build.sh` entirely (libtorch won't be built)
- **`build.sh` JNI cmake path** — `CMAKE_PREFIX_PATH=../libtorch` (relative) sometimes fails; use absolute path `/build/djl/engines/pytorch/pytorch-native/libtorch` if needed
- **publishToMavenLocal** doesn't publish the custom JAR task output — install the JAR manually via `cp`
- **Don't build with CUDA 13.1** — PyTorch 2.7.1 doesn't support it (CCCL 3.x removed APIs). Use 12.8.
- **Two Ninja builds simultaneously** — don't do this, they conflict

---

## 11. Build Progress & Status

| Milestone | Status |
|---|---|
| Docker container setup (NGC base) | ✅ Done |
| CUDA 12.8 toolkit installed | ✅ Done |
| DJL fork + feature branch | ✅ Done |
| `build.sh` updated for CUDA 12.8 | ✅ Done |
| `build.gradle.kts` custom task | ✅ Done |
| `CMakeLists.txt` adjustments | ✅ Done |
| Full libtorch compile (2485/2485) | ✅ Done |
| DJL JNI compile (`libdjl_torch.so`) | ✅ Done |
| Gradle JAR packaging | ✅ Done |
| Local Maven install | ✅ Done |
| Java verification test | ✅ Done (CPU + GPU tests passed) |
| Git commit + push | ✅ Done (`feature/aarch64-cuda-blackwell`) |
| GitHub Release + JAR upload | ✅ Done (`v2.7.1-aarch64-cu128`) |
| Rebuilt with `+PTX` for bare-metal sm_121 | ✅ Done |
| JNI lib renamed to `0.36.0-libdjl_torch.so` in JAR | ✅ Done |

**Build artifact**: `pytorch-native-cu128-2.7.1-linux-aarch64.jar` (488MB — includes PTX code)
**Local Maven**: `~/.m2/repository/ai/djl/pytorch/pytorch-native-cu128/2.7.1/`
**GitHub Release**: `https://github.com/hw1964/djl/releases/tag/v2.7.1-aarch64-cu128`

---

## 12. Useful Commands Reference

```bash
# Check CUDA versions installed
ls -d /usr/local/cuda-*
/usr/local/cuda-12.8/bin/nvcc --version

# Verify CUB version in CUDA 12.8
grep '#define CUB_VERSION' /usr/local/cuda-12.8/targets/sbsa-linux/include/cub/version.cuh

# Check libtorch after install
ls -lh libtorch/lib/*.so | head -20

# Verify the JNI .so
file build/libdjl_torch.so

# Check JAR contents
jar tf build/libs/pytorch-native-cu128-2.7.1-linux-aarch64.jar | grep "native/lib"

# Clear DJL cache and retest
rm -rf ~/.djl.ai/pytorch/ && mvn exec:java -Dexec.mainClass="your.MainClass"

# Second terminal into running container
docker exec -it $(docker ps -q --filter ancestor=djl-build-blackwell:latest) bash
```

---

## 13. Next Steps

1. **Open upstream PR** to `deepjavalibrary/djl` with aarch64 CUDA build support (optional)
