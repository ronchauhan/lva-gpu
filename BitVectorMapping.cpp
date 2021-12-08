#include "BitVectorMapping.h"

#include "llvm/IR/Constant.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

bool BitVectorMapping::shouldMap(const llvm::Value *V) const {
  if (!V->hasName())
    return false;

  if (llvm::isa<llvm::Constant>(V))
    return false;

  if (llvm::isa<llvm::BasicBlock>(V))
    return false;

  if (llvm::isa<llvm::StoreInst>(V))
    return false;

  if (auto Inst = llvm::dyn_cast<llvm::Instruction>(V)) {
    if (Inst->isTerminator())
      return false;
  }
  return true;
}

llvm::Optional<uint64_t>
BitVectorMapping::getBitVectorIndexForValue(const llvm::Value *V) const {
  uint64_t address = (uint64_t) & (*V);
  auto it = addrIndexMap.find(address);
  if (it == addrIndexMap.end())
    return llvm::None;
  return it->second;
}

void BitVectorMapping::assignIndexToValue(const llvm::Value *V) {
  uint64_t address = (uint64_t) & (*V);
  auto it = addrIndexMap.find(address);
  if (it == addrIndexMap.end()) {
    addrIndexMap[address] = nextIndex;
    indexAddrMap[nextIndex] = address;
    ++nextIndex;
  }
}

// Create and return a bitvector where the bit corresponding to each value in
// <values> is set to 1. All other bits are set to 0.
llvm::BitVector BitVectorMapping::createBitVectorWithTrue(
    const std::vector<llvm::Value *> &values) const {
  llvm::BitVector BV(addrIndexMap.size(), false);
  for (auto value : values) {
    llvm::Optional<uint64_t> indexOrNone = getBitVectorIndexForValue(value);
    assert(indexOrNone && "value must be mapped!");
    BV[indexOrNone.getValue()] = true;
  }
  return BV;
}

void BitVectorMapping::printMapping() const {
  for (auto KV : addrIndexMap) {
    llvm::outs() << ((llvm::Value *)KV.first)->getName() << '\t' << KV.second
                 << '\n';
  }
}
