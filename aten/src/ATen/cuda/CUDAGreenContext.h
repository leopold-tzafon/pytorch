#pragma once
#include <ATen/cuda/CUDAEvent.h>
#include <cuda.h>

// Forward declare green context as opaque ptr
typedef struct CUgreenCtx_st* CUgreenCtx;

namespace at::cuda {

class TORCH_CUDA_CPP_API GreenContext {
 public:
  // Green context creation
  static std::unique_ptr<GreenContext> create(
      uint32_t num_sms,
      std::optional<uint32_t> device_id);
  ~GreenContext() noexcept;

  // Delete copy constructor and assignment
  GreenContext(const GreenContext&) = delete;
  GreenContext& operator=(const GreenContext&) = delete;

  // Make this context current
  void setContext();

  void popContext();

 private:
  GreenContext(uint32_t device_id, uint32_t num_sms);
  // Implement move operations
  GreenContext(GreenContext&& other) noexcept;
  GreenContext& operator=(GreenContext&& other) noexcept;

  int32_t device_id_ = -1;
  CUgreenCtx green_ctx_ = nullptr;
  CUcontext context_ = nullptr;
  cudaStream_t parent_stream_ = nullptr;
};
} // namespace at::cuda
