#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "BitVectorMapping.h"

#include <hip/hip_runtime.h>

using namespace llvm;

#define getEffectiveIndex(rowWidth, i, j) (((i) * (rowWidth)) + (j))

// ===================== GPU_FIXED_POINT_STUFF_BEGIN ===========================

__global__ void handleSuccessors(int *InA, int *OutA,
                                 std::uint64_t bitVectorLen, int *successorArr,
                                 std::uint64_t successorArrWidth,
                                 bool isSecond) {
  std::uint64_t c = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  std::uint64_t successorArrI = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;

  if (successorArrI >= successorArrWidth || c >= bitVectorLen)
    return;

  int instIdInSuccessorArr =
      getEffectiveIndex(successorArrWidth, 0, successorArrI);
  int successorIdInSuccessorArr =
      getEffectiveIndex(successorArrWidth, 1, successorArrI);

  int instIdBV =
      getEffectiveIndex(bitVectorLen, successorArr[instIdInSuccessorArr], c);
  int successorInstIdBV = getEffectiveIndex(
      bitVectorLen, successorArr[successorIdInSuccessorArr], c);

  if (isSecond)
    OutA[instIdBV] |= InA[successorInstIdBV];
  else
    OutA[instIdBV] = InA[successorInstIdBV];
}

__global__ void applyTransferFunction(int *InA, int *OutA, int *GenA,
                                      int *KillA, int bitVectorLen,
                                      std::uint64_t instCount) {
  std::uint64_t c = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  std::uint64_t r = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;

  if (r >= instCount || c >= bitVectorLen)
    return;

  std::uint64_t instIdBV = getEffectiveIndex(bitVectorLen, r, c);
  InA[instIdBV] = GenA[instIdBV] | (OutA[instIdBV] & ~(KillA[instIdBV]));
}

// ======================= GPU_FIXED_POINT_STUFF_END ===========================

// ======================== LIVENESS_GEN_KILL_BEGIN ============================

static bool shouldConsider(const llvm::Value *V) {
  if (!V->hasName())
    return false;

  if (llvm::isa<llvm::Constant>(V))
    return false;

  if (llvm::isa<llvm::BasicBlock>(V))
    return false;

  return true;
}

static std::vector<Value *> getLivenessGen(Instruction *Inst) {
  std::vector<Value *> Gen;
  if (auto BI = dyn_cast<BranchInst>(Inst)) {
    // Only condition used in a (conditional) branch
    if (BI->isConditional())
      Gen.push_back(BI->getCondition());

    return Gen;
  }
  for (auto Op : Inst->operand_values()) {
    if (shouldConsider(Op)) {
      Gen.push_back(Op);
    }
  }
  return Gen;
}

static std::vector<Value *> getLivenessKill(Instruction *Inst) {
  std::vector<Value *> Kill;
  if (isa<BranchInst>(Inst))
    return Kill;

  if (isa<StoreInst>(Inst))
    return Kill;

  if (isa<ReturnInst>(Inst))
    return Kill;

  if (Value *V = dyn_cast<Value>(Inst)) {
    if (shouldConsider(V))
      Kill.push_back(V);
  }
  return Kill;
}

// ========================= LIVENESS_GEN_KILL_END =============================

// The IR must go through lowerswitch and instnamer passes before running this.
class LiveVariablesAnalysis {
private:
  std::map<Instruction *, BitVector> Gen, Kill;
  std::map<Instruction *, BitVector> In, Out;
  int *GenArr, *KillArr, *InArr, *OutArr;

  BitVectorMapping BVM;
  std::uint64_t instCount, bitVectorLength;

  std::vector<Instruction *> intToInstrMap;
  std::map<Instruction *, int> instrToIntMap;

  // We ensure that the CFG has atmost two successors.
  int *S1, *S2;
  std::uint64_t S1size, S2size;
  std::uint64_t roundRobinBound;

public:
  LiveVariablesAnalysis(BitVectorMapping BVM) {
    this->BVM = BVM;
    instCount = 0;
    bitVectorLength = 0;
  }

  std::vector<llvm::Instruction *> getSuccessors(llvm::Instruction *Inst) {
    std::vector<llvm::Instruction *> Result;
    llvm::BasicBlock *Parent = Inst->getParent();
    if (&Parent->back() == Inst) {
      for (llvm::BasicBlock *BB : successors(Parent)) {
        Result.push_back(&BB->front());
      }
    } else {
      Result.push_back(Inst->getNextNonDebugInstruction());
    }
    return Result;
  }

  void initializeSuccessorInfo(Function &F) {
    std::vector<std::pair<int, int>> Succs1, Succs2;
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      auto successors = getSuccessors(&*I);
      int instIdx = instrToIntMap[&*I];

      assert(successors.size() <= 2 &&
             "More than two successors not supported");

      if (successors.empty())
        continue;

      Succs1.push_back(std::make_pair(instIdx, instrToIntMap[successors[0]]));

      if (successors.size() == 2)
        Succs2.push_back(std::make_pair(instIdx, instrToIntMap[successors[1]]));
    }
    S1size = Succs1.size();
    S2size = Succs2.size();

    hipMallocManaged(&S1, 2 * Succs1.size() * sizeof(int));
    for (int j = 0; j < Succs1.size(); j++) {
      int idx = getEffectiveIndex(Succs1.size(), 0, j);
      S1[idx] = Succs1[j].first;
      idx = getEffectiveIndex(Succs1.size(), 1, j);
      S1[idx] = Succs1[j].second;
    }

    hipMallocManaged(&S2, 2 * Succs2.size() * sizeof(int));
    for (int j = 0; j < Succs2.size(); j++) {
      int idx = getEffectiveIndex(Succs2.size(), 0, j);
      S2[idx] = Succs2[j].first;
      idx = getEffectiveIndex(Succs2.size(), 1, j);
      S2[idx] = Succs2[j].second;
    }
  }

  void initializeBitVectorArrays(Function &F) {
    hipMallocManaged(&GenArr, instCount * bitVectorLength * sizeof(int));
    hipMallocManaged(&KillArr, instCount * bitVectorLength * sizeof(int));
    hipMallocManaged(&InArr, instCount * bitVectorLength * sizeof(int));
    hipMallocManaged(&OutArr, instCount * bitVectorLength * sizeof(int));

    int instIdx = 0;
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      for (int i = 0; i < bitVectorLength; i++) {
        int effectiveIdx = getEffectiveIndex(bitVectorLength, instIdx, i);
        GenArr[effectiveIdx] = (Gen[&*I].test(i)) ? 1 : 0;
        KillArr[effectiveIdx] = (Kill[&*I].test(i)) ? 1 : 0;
        InArr[effectiveIdx] = (In[&*I].test(i)) ? 1 : 0;
        OutArr[effectiveIdx] = (Out[&*I].test(i)) ? 1 : 0;
      }
      instIdx += 1;
    }
  }

  void printSuccessors() {
    std::cout << "First => \n";
    int idx;
    for (int i = 0; i < S1size; i++) {
      idx = getEffectiveIndex(S1size, 0, i);
      std::cout << S1[idx] << ' ';
    }
    std::cout << "\n";
    for (int i = 0; i < S1size; i++) {
      idx = getEffectiveIndex(S1size, 1, i);
      std::cout << S1[idx] << ' ';
    }
    std::cout << "\n";
    std::cout << "Second => \n";
    for (int i = 0; i < S2size; i++) {
      idx = getEffectiveIndex(S2size, 0, i);
      std::cout << S2[idx] << ' ';
    }
    std::cout << "\n";
    for (int i = 0; i < S2size; i++) {
      idx = getEffectiveIndex(S2size, 1, i);
      std::cout << S2[idx] << ' ';
    }
    std::cout << "\n";
  }

  void printDataFlowValues() {
    std::cout << "In => \n";
    for (int i = 0; i < instCount; i++) {
      for (int j = 0; j < bitVectorLength; j++) {
        int effectiveIdx = getEffectiveIndex(bitVectorLength, i, j);
        std::cout << InArr[effectiveIdx] << ' ';
      }
      std::cout << "\n";
    }
    std::cout << "Out => \n";
    for (int i = 0; i < instCount; i++) {
      for (int j = 0; j < bitVectorLength; j++) {
        int effectiveIdx = getEffectiveIndex(bitVectorLength, i, j);
        std::cout << OutArr[effectiveIdx] << ' ';
      }
      std::cout << "\n";
    }
    std::cout << "Gen => \n";
    for (int i = 0; i < instCount; i++) {
      for (int j = 0; j < bitVectorLength; j++) {
        int effectiveIdx = getEffectiveIndex(bitVectorLength, i, j);
        std::cout << GenArr[effectiveIdx] << ' ';
      }
      std::cout << "\n";
    }
    std::cout << "Kill => \n";
    for (int i = 0; i < instCount; i++) {
      for (int j = 0; j < bitVectorLength; j++) {
        int effectiveIdx = getEffectiveIndex(bitVectorLength, i, j);
        std::cout << KillArr[effectiveIdx] << ' ';
      }
      std::cout << "\n";
    }
  }

  void mapGlobalVariables(Module *M) {
    auto I = M->global_begin();
    auto E = M->global_end();
    for (I; I != E; ++I) {
      if (llvm::Value *V = dyn_cast<Value>(&*I))
        BVM.assignIndexToValue(V);
    }
  }

  void mapLocalVariables(Function &F) {
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      if (Value *V = dyn_cast<Value>(&(*I))) {
        if (BVM.shouldMap(V) && !BVM.getBitVectorIndexForValue(V))
          BVM.assignIndexToValue(V);
      }
      for (auto operand : I->operand_values()) {
        if (BVM.shouldMap(operand) && !BVM.getBitVectorIndexForValue(operand))
          BVM.assignIndexToValue(operand);
      }
    }
  }

  void computeFixedPointOnGPU() {
    for (std::uint64_t i = 0; i < roundRobinBound; i++) {
      dim3 blockSize(32, 32);
      int bx = (bitVectorLength + blockSize.x - 1) / blockSize.x;
      int by = (instCount + blockSize.y - 1) / blockSize.y;

      hipLaunchKernelGGL(applyTransferFunction, dim3(bx, by), blockSize, 0, 0,
                         InArr, OutArr, GenArr, KillArr, bitVectorLength,
                         instCount);

      by = (S1size + blockSize.y - 1) / blockSize.y;
      hipLaunchKernelGGL(handleSuccessors, dim3(bx, by), blockSize, 0, 0, InArr,
                         OutArr, bitVectorLength, S1, S1size,
                         /*isSecond=*/false);

      by = (S2size + blockSize.y - 1) / blockSize.y;
      hipLaunchKernelGGL(handleSuccessors, dim3(bx, by), blockSize, 0, 0, InArr,
                         OutArr, bitVectorLength, S2, S2size,
                         /*isSecond=*/true);
    }
    hipDeviceSynchronize();
  }

  void run(Function &F) {
    mapGlobalVariables(F.getParent());
    mapLocalVariables(F);
    BVM.printMapping();

    std::cout << BVM.getAddressToIndexMap().size() << '\n';

    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      intToInstrMap.push_back(&*I);
      instrToIntMap[&*I] = instCount;
      instCount += 1;
      Gen[&*I] = (BVM.createBitVectorWithTrue(getLivenessGen(&*I)));
      if (instCount == 1) {
        bitVectorLength = Gen[&*I].size();
      }
      Kill[&*I] = (BVM.createBitVectorWithTrue(getLivenessKill(&*I)));
      In[&*I] = (BVM.createBitVectorWithAllBitsZero());
      Out[&*I] = (BVM.createBitVectorWithAllBitsZero());
    }

    initializeBitVectorArrays(F);
    initializeSuccessorInfo(F);
    this->roundRobinBound =
        instCount * instCount; // placeholder value for testing in small
                               // programs. TODO: Compute actual bound later

    std::chrono::time_point<std::chrono::steady_clock> start, end;
    std::chrono::duration<float, std::ratio<1, 1>> duration;

    start = std::chrono::steady_clock::now();
    computeFixedPointOnGPU();
    end = std::chrono::steady_clock::now();

    duration = end - start;

    std::cout << "Solving took " << std::setprecision(10) << duration.count()
              << "s\n";
    std::cout.flush();

    std::cout << "Printing final data flow values\n";
    printDataFlowValues();

    // Cleanup
    hipFree(GenArr);
    hipFree(KillArr);
    hipFree(InArr);
    hipFree(OutArr);
    hipFree(S1);
    hipFree(S2);
    BVM.reset();
  }
};

int main(int argc, char *argv[]) {
  assert(argc == 2 && "Pass only one argument - the .ll file");

  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M(parseAssemblyFile(argv[1], Err, Ctx));

  bool isValid = verifyModule(*M);
  assert(!isValid && "Broken module!");

  for (auto FI = M->begin(); FI != M->end(); ++FI) {
    BitVectorMapping BVM;
    LiveVariablesAnalysis LA(BVM);
    LA.run(*FI);
  }
}
