CXX_COMPILER="/opt/rocm/llvm/bin/clang++"

LLVM_INCLUDE_DIR="/opt/rocm/llvm/include"
LLVM_LIB_DIR=$(/opt/rocm/llvm/bin/llvm-config --libdir)
LLVM_LINK_LIBS=$(/opt/rocm/llvm/bin/llvm-config --libs)

BUILD_DIR=build

mkdir -p $BUILD_DIR

$CXX_COMPILER -std=c++14 -I$LLVM_INCLUDE_DIR -c BitVectorMapping.cpp -o $BUILD_DIR/BitVectorMapping.o

/opt/rocm/bin/hipcc -std=c++14 lva-gpu.cpp -I$LLVM_INCLUDE_DIR -Xlinker -L$LLVM_LIB_DIR -Xlinker $BUILD_DIR/BitVectorMapping.o -Xlinker $LLVM_LINK_LIBS -Xlinker -ltinfo -o $BUILD_DIR/lva-gpu -v
