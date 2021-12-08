#ifndef BITVECTORMAPPING_H
#define BITVECTORMAPPING_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Optional.h"
#include "llvm/IR/Value.h"

#include <map>
#include <vector>

// This class is meant to hold a mapping from addresses to indices so that we
// can create a bitvector representing a set of llvm::Values of interest.
class BitVectorMapping {

  using AddressToIndexMap = std::map<uint64_t, uint64_t>;
  using IndexToAddressMap = std::map<uint64_t, uint64_t>;

  IndexToAddressMap indexAddrMap;
  AddressToIndexMap addrIndexMap;

  // Index to assign for the mapping
  uint64_t nextIndex;

public:
  BitVectorMapping() { nextIndex = 0; }

  void reset() {
    nextIndex = 0;
    addrIndexMap.clear();
    indexAddrMap.clear();
  }

  // This is specific to live variables analysis right now.
  bool shouldMap(const llvm::Value *V) const;

  const IndexToAddressMap &getIndexToAddressMap() { return indexAddrMap; }

  const AddressToIndexMap &getAddressToIndexMap() { return addrIndexMap; }

  bool assignIndexToValue(const llvm::Value *V);

  void printMapping() const;

  // Utility functions
  llvm::Optional<uint64_t>
  getBitVectorIndexForValue(const llvm::Value *V) const;

  // Create and return a bitvector where the bit corresponding to each value in
  // <values> is set to 1. All other bits are set to 0.
  llvm::BitVector
  createBitVectorWithTrue(const std::vector<llvm::Value *> &values) const;

  llvm::BitVector createBitVectorWithAllBitsZero() const {
    return llvm::BitVector(addrIndexMap.size(), false);
  }

  llvm::BitVector createBitVectorWithAllBitsOne() const {
    return llvm::BitVector(addrIndexMap.size(), true);
  }
};

#endif
