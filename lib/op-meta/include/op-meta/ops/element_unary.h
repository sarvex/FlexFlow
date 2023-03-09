#ifndef _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H
#define _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H

#include "op-meta/ffconst.h"
#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

struct ElementUnaryAttrs : public UnaryOpAttrs {
public:
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  OperatorType op;
  /* bool inplace; */
  float scalar = 0.0;
};

bool operator==(ElementUnaryAttrs const &, ElementUnaryAttrs const &);
bool operator<(ElementUnaryAttrs const &, ElementUnaryAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::ElementUnaryAttrs, op, scalar);

namespace std {
template <>
struct hash<::FlexFlow::ElementUnaryAttrs> {
  size_t operator()(::FlexFlow::ElementUnaryAttrs const &) const;
};
} 

#endif 