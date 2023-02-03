#ifndef _FLEXFLOW_OP_META_OPS_BINARY_OP_H
#define _FLEXFLOW_OP_META_OPS_BINARY_OP_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/op_params.h"

namespace FlexFlow {

class BinaryOpParams : public OpParamsInterface {
  bool is_valid(std::vector<ParallelTensorShape> const &) const override final;
  int num_outputs(std::vector<ParallelTensorShape> const &) const override final;

  virtual bool is_valid(ParallelTensorShape const &, ParallelTensorShape const &) const;
};

}

#endif 