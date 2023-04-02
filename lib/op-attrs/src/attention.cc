#include "op-attrs/ops/attention.h"
#include "utils/hash-utils.h"
#include <algorithm>
#include "utils/visitable_funcs.h"

namespace FlexFlow {

/* bool MultiHeadAttentionAttrs::is_valid(std::vector<ParallelTensorShape> const &inputs) const { */
/*   return (inputs.size() == 3 && std::all_of(inputs.begin(), inputs.end(), [](ParallelTensorShape const &s) { return s.is_valid(); })); */
/*   bool is_valid = true; */
/*   return is_valid; */
/* } */

bool operator==(MultiHeadAttentionAttrs const &lhs, MultiHeadAttentionAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(MultiHeadAttentionAttrs const &lhs, MultiHeadAttentionAttrs const &rhs) {
  return visit_lt(lhs, rhs);
}

int vProjSize(MultiHeadAttentionAttrs const &attrs) {
  return attrs.vdim;
}

int kProjSize(MultiHeadAttentionAttrs const &attrs) {
  return attrs.kdim;
}

int oProjSize(MultiHeadAttentionAttrs const &attrs) {
  return attrs.embed_dim;
}

}

namespace std {
using ::FlexFlow::MultiHeadAttentionAttrs;

size_t hash<MultiHeadAttentionAttrs>::operator()(
    MultiHeadAttentionAttrs const &attrs) const {
  return visit_hash(attrs);
} 
}