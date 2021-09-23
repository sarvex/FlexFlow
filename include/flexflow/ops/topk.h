#ifndef _FLEXFLOW_TOPK_H_
#define _FLEXFLOW_TOPK_H_

#include "flexflow/model.h"

namespace FlexFlow {

class TopKMeta : public OpMeta {
public:
  TopKMeta(FFHandler handle);
  bool sorted;
};

class TopK : public Op {
public:
  TopK(FFModel& model,
       const Tensor input,
       int k, bool sorted,
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
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
  static void forward_kernel(const TopKMeta* m,
                      const float* input_ptr,
                      float* output_ptr,
                      int* indices_ptr,
                      size_t batch_size, int length, int k,
                      bool sorted,
                      cudaStream_t stream);
  static void backward_kernel(const TopKMeta* m,
                       const float* out_grad_ptr,
                       const int* indices_ptr,
                       float* in_grad_ptr,
                       size_t batch_size, int length, int k,
		       cudaStream_t stream);
public:
  int k;
  bool sorted;
};

}; // namespace FlexFlow

#endif