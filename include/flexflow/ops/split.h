#ifndef _FLEXFLOW_SPLIT_H
#define _FLEXFLOW_SPLIT_H

#include "flexflow/model.h"

namespace FlexFlow {

class Split : public Op {
public:
  Split(FFModel& model,
        const ParallelTensor input,
        const std::vector<int>& split,
        int axis,
        const char* name);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}

  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
#if defined (FF_USE_CUDA) || defined (FF_USE_HIP_CUDA)
  static void forward_kernel(float **out_ptrs,
                             float const *in_ptr,
                             Legion::coord_t const *out_blk_sizes,
                             Legion::coord_t in_blk_size,
                             Legion::coord_t num_blks,
                             int numOutputs,
                             cudaStream_t stream);
#else
  static void forward_kernel(float **out_ptrs,
                             float const *in_ptr,
                             Legion::coord_t const *out_blk_sizes,
                             Legion::coord_t in_blk_size,
                             Legion::coord_t num_blks,
                             int numOutputs,
                             hipStream_t stream);
#endif
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;

  size_t get_params_hash() const override;
public:
  int axis;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_SPLIT_H
