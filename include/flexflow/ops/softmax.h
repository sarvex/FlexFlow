#ifndef _FLEXFLOW_SOFTMAX_H
#define _FLEXFLOW_SOFTMAX_H

#include "flexflow/model.h"

namespace FlexFlow {

class Softmax;

class SoftmaxMeta : public OpMeta {
public:
  SoftmaxMeta(FFHandler handle,
              const Softmax* softmax,
              const Legion::Domain& input_domain);
#if defined (FF_USE_CUDA) || defined (FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t inputTensor;
#else
  miopenTensorDescriptor_t inputTensor;
#endif
  bool profiling;
  int dim;
  char op_name[MAX_OPNAME];
};

class Softmax : public Op {
public:
  Softmax(FFModel& model,
          const ParallelTensor logit,
          int dim,
          const char* name);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  bool get_int_parameter(PMParameter, int*) const;
  void print_layer(const FFModel& model) {assert(0);}
  static Op* create_operator_from_layer(FFModel& model,
                                        const Layer* layer,
                                        const std::vector<ParallelTensor>& inputs);
  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  void init_meta(SoftmaxMeta *m,
                 Legion::Rect<2> const &input,
                 Legion::Rect<2> const &output) const;
  bool measure_operator_cost(Simulator* sim,
                             const ParallelConfig& pc,
                             CostMetrics& cost_metrics) const;
#if defined (FF_USE_CUDA) || defined (FF_USE_HIP_CUDA)
  static void forward_kernel(SoftmaxMeta const *m,
                             float const *input_ptr,
                             float *output_ptr,
                             cudaStream_t stream);
  static void backward_kernel(float *input_grad_ptr,
                              float const *output_grad_ptr,
                              size_t num_elements,
                              cudaStream_t stream);
#else
  static void forward_kernel(SoftmaxMeta const *m,
                             float const *input_ptr,
                             float *output_ptr,
                             hipStream_t stream);
  static void backward_kernel(float *input_grad_ptr,
                              float const *output_grad_ptr,
                              size_t num_elements,
                              hipStream_t stream);
#endif
  size_t get_params_hash() const override;
private:
  template<int NDIM>
  static void forward_task_with_dim(const Legion::Task *task,
                                    const std::vector<Legion::PhysicalRegion> &regions,
                                    Legion::Context ctx, Legion::Runtime *runtime);
  template<int NDIM>
  static void backward_task_with_dim(const Legion::Task *task,
                                     const std::vector<Legion::PhysicalRegion> &regions,
                                     Legion::Context ctx, Legion::Runtime *runtime);
public:
  int dim;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_SOFTMAX_H
