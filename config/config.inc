#!/bin/bash

# set CC and CXX
if [ -n "$CC" ]; then
  SET_CC="-DCMAKE_C_COMPILER=${CC}"
fi

if [ -n "$CXX" ]; then
  SET_CXX="-DCMAKE_CXX_COMPILER=${CXX}"
fi

# set FLAGS
if [ -n "$CC_FLAGS" ]; then
  SET_CC_FLAGS="CC_FLAGS=${CC_FLAGS}"
fi

if [ -n "$NVCC_FLAGS" ]; then
  SET_NVCC_FLAGS="NVCC_FLAGS=${NVCC_FLAGS}"
fi

if [ -n "$LD_FLAGS" ]; then
  SET_LD_FLAGS="LD_FLAGS=${LD_FLAGS}"
fi

#set installation dir
if [ -n "$INSTALL_DIR" ]; then
  SET_INSTALL_DIR="-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}"
fi

# set build type
if [ -n "$BUILD_TYPE" ]; then
  SET_BUILD="-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
fi

# set CUDA Arch
if [ -n "$FF_CUDA_ARCH" ]; then
  SET_CUDA_ARCH="-DFF_CUDA_ARCH=${FF_CUDA_ARCH}"
fi

# set CUDA dir
if [ -n "$CUDA_DIR" ]; then
  SET_CUDA="-DCUDA_PATH=${CUDA_DIR}"
  CUDA_PATH="${CUDA_DIR}/lib64/stubs"
  SET_CUDA_LIB_PATH="CUDA_PATH=${CUDA_PATH}"
fi

# set cudnn dir
if [ -n "$CUDNN_DIR" ]; then
  SET_CUDNN="-DCUDNN_PATH=${CUDNN_DIR}"
fi

# enable Python
if [ "$FF_USE_PYTHON" = "ON" ]; then
  SET_PYTHON="-DFF_USE_PYTHON=ON"
elif [ "$FF_USE_PYTHON" = "OFF" ]; then
  SET_PYTHON="-DFF_USE_PYTHON=OFF"
else
  SET_PYTHON="-DFF_USE_PYTHON=ON"
fi

# enable NCCL
if [ "$FF_USE_NCCL" = "ON" ]; then
  SET_NCCL="-DFF_USE_NCCL=ON"
elif [ "$FF_USE_NCCL" = "OFF" ]; then
  SET_NCCL="-DFF_USE_NCCL=OFF"
else
  SET_NCCL="-DFF_USE_NCCL=ON"
fi

# enable GASNET
if [ "$FF_USE_GASNET" = "ON" ]; then
  SET_GASNET="-DFF_USE_GASNET=ON"
  if [ "$FF_GASNET_CONDUIT" = "ibv" ]; then
    SET_GASNET+=" -DFF_GASNET_CONDUIT=ibv"
  elif [ "$FF_GASNET_CONDUIT" = "mpi" ]; then
    SET_GASNET+=" -DFF_GASNET_CONDUIT=mpi"
  elif [ "$FF_GASNET_CONDUIT" = "udp" ]; then
    SET_GASNET+=" -DFF_GASNET_CONDUIT=udp"
  elif [ "$FF_GASNET_CONDUIT" = "ucx" ]; then
    SET_GASNET+=" -DFF_GASNET_CONDUIT=ucx"
  fi
elif [ "$FF_USE_GASNET" = "OFF" ]; then
  SET_GASNET="-DFF_USE_GASNET=OFF"
else
  SET_GASNET="-DFF_USE_GASNET=ON"
fi

# build C++ examples
if [ "$FF_BUILD_ALL_EXAMPLES" = "ON" ]; then
  SET_EXAMPLES="-DFF_BUILD_ALL_EXAMPLES=ON"
elif [ "$FF_BUILD_ALL_EXAMPLES" = "OFF" ]; then
  SET_EXAMPLES="-DFF_BUILD_ALL_EXAMPLES=OFF"
else
  SET_EXAMPLES="-DFF_BUILD_ALL_EXAMPLES=ON"
fi

# enable C++ unit tests
if [ "$FF_BUILD_UNIT_TESTS" = "ON" ]; then
  SET_BUILD_UNIT_TESTS="-DFF_BUILD_UNIT_TESTS=ON"
elif [ "$FF_BUILD_UNIT_TESTS" = "OFF" ]; then
  SET_BUILD_UNIT_TESTS="-DFF_BUILD_UNIT_TESTS=OFF"
else
  SET_BUILD_UNIT_TESTS="-DFF_BUILD_UNIT_TESTS=OFF"
fi

# build using pre-compiled libraries, where available
if [ "$FF_USE_PREBUILT_LEGION" = "ON" ]; then
  SET_USE_PREBUILT_LEGION="-DFF_USE_PREBUILT_LEGION=ON"
elif [ "$FF_USE_PREBUILT_LEGION" = "OFF" ]; then
  SET_USE_PREBUILT_LEGION="-DFF_USE_PREBUILT_LEGION=OFF"
else
  SET_USE_PREBUILT_LEGION="-DFF_USE_PREBUILT_LEGION=ON"
fi
if [ "$FF_USE_PREBUILT_NCCL" = "ON" ]; then
  SET_USE_PREBUILT_NCCL="-DFF_USE_PREBUILT_NCCL=ON"
elif [ "$FF_USE_PREBUILT_NCCL" = "OFF" ]; then
  SET_USE_PREBUILT_NCCL="-DFF_USE_PREBUILT_NCCL=OFF"
else
  SET_USE_PREBUILT_NCCL="-DFF_USE_PREBUILT_NCCL=ON"
fi
if [ "$FF_USE_ALL_PREBUILT_LIBRARIES" = "ON" ]; then
  SET_USE_ALL_PREBUILT_LIBRARIES="-DFF_USE_ALL_PREBUILT_LIBRARIES=ON"
elif [ "$FF_USE_ALL_PREBUILT_LIBRARIES" = "OFF" ]; then
  SET_USE_ALL_PREBUILT_LIBRARIES="-DFF_USE_ALL_PREBUILT_LIBRARIES=OFF"
else
  SET_USE_ALL_PREBUILT_LIBRARIES="-DFF_USE_ALL_PREBUILT_LIBRARIES=OFF"
fi


# enable avx2
if [ "$FF_USE_AVX2" = "ON" ]; then
  SET_AVX2="-DFF_USE_AVX2=ON"
elif [ "$FF_USE_AVX2" = "OFF" ]; then
  SET_AVX2="-DFF_USE_AVX2=OFF"
else
  SET_AVX2="-DFF_USE_AVX2=OFF"
fi

#set max dims
if [ -n "$FF_MAX_DIM" ]; then
  SET_MAX_DIM="-DFF_MAX_DIM=${FF_MAX_DIM}"
fi

# set ROCM path
if [ -n "$ROCM_PATH" ]; then
  SET_ROCM_PATH="-DROCM_PATH=${ROCM_PATH}"
fi

# set GPU backend
if [ -n "$FF_GPU_BACKEND" ]; then
  SET_FF_GPU_BACKEND="-DFF_GPU_BACKEND=${FF_GPU_BACKEND}"

  # cmake does not play nicely with overrides via `set()` of CMAKE_CXX_COMPILER and friends
  # because it uses their values to setup the toolchain.
  # see: https://gitlab.kitware.com/cmake/community/-/wikis/FAQ#how-do-i-use-a-different-compiler
  #
  # Ideally we would use the values internally to the cmake script, e.g. HIP_HIPCC_EXECUTABLE,
  # to set these values but this is a sufficient compromise.
  if [ "$FF_GPU_BACKEND" = "hip_cuda" ] || [ "$FF_GPU_BACKEND" = "hip_rocm" ]; then
    if [ -n "$SET_CXX" ]; then
      echo "FF_GPU_BACKEND is set to ${FF_GPU_BACKEND}. Normally we would set the compiler and linker" 1>&2
      echo "to hipcc, but the compiler is already set to ${SET_CXX}". 1>&2
    else
      if [ "$FF_GPU_BACKEND" = "hip_cuda" ]; then
        # Configuring hipcc for nvidia:
        #
        # The platform hipcc targets is configured by the HIP_PLATFORM env var.
        # Ideally, as we could in the shell, we would call `HIP_PLATFORM=nvidia hipcc <...>`.
        # However, CMAKE_CXX_COMPILER doesn't allow configuration as such. Additionally,
        # cmake doesn't allow setting environment variables for target builds like make does
        # with exported variables.
        #
        # Instead, this file configures hipcc with HIP_PLATFORM and CUDA_PATH
        #
        # CMAKE requires CMAKE_CXX_COMPILER exists before cmake is called, so we can't
        # write out this file during build configuration.
        echo "HIP_PLATFORM=nvidia CUDA_PATH=${CUDA_DIR} ${ROCM_PATH}/bin/hipcc \$@" > "$(pwd)/nvidia_hipcc"
        chmod +x "$(pwd)/nvidia_hipcc"
        SET_CXX="-DCMAKE_CXX_COMPILER=$(pwd)/nvidia_hipcc -DCMAKE_CXX_LINKER=$(pwd)/nvidia_hipcc"
      else
        SET_CXX="-DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -DCMAKE_CXX_LINKER=/opt/rocm/bin/hipcc"
      fi
    fi
  fi
fi

CMAKE_FLAGS="-DCUDA_USE_STATIC_CUDA_RUNTIME=OFF ${SET_CC} ${SET_CXX} ${SET_INSTALL_DIR} ${SET_BUILD} ${SET_CUDA_ARCH} ${SET_CUDA} ${SET_CUDNN} ${SET_PYTHON} ${SET_NCCL} ${SET_GASNET} ${SET_EXAMPLES} ${SET_USE_PREBUILT_LEGION} ${SET_USE_PREBUILT_NCCL} ${SET_USE_ALL_PREBUILT_LIBRARIES} ${SET_BUILD_UNIT_TESTS} ${SET_AVX2} ${SET_MAX_DIM} ${SET_ROCM_PATH} ${SET_FF_GPU_BACKEND}"

function run_cmake() {
SRC_LOCATION=${SRC_LOCATION:=`dirname $0`/../}
CMAKE_COMMAND="${SET_CC_FLAGS} ${SET_NVCC_FLAGS} ${SET_LD_FLAGS} ${SET_CUDA_LIB_PATH} cmake ${CMAKE_FLAGS} $* ${SRC_LOCATION}"
echo $CMAKE_COMMAND
eval $CMAKE_COMMAND
}
