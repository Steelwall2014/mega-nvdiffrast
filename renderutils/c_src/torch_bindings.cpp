#include <future>
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <string>

#define CHECK_CUDA_ERROR(CUDA_CALL) { cudaError_t err = CUDA_CALL; AT_CUDA_CHECK(cudaGetLastError()); }
#define CHECK_DEVICE(...) do { TORCH_CHECK(at::cuda::check_device({__VA_ARGS__}), __func__, "(): Inputs " #__VA_ARGS__ " must reside on the same GPU device") } while(0)

#include "common.h"
#include "normal_tangent.h"
#include "virtual_shadow_mapping.h"

#define BLOCK_X 8
#define BLOCK_Y 8

#define calcMipLevelSize(w, h, i) make_int2((w >> (i)) > 1 ? (w >> (i)) : 1, (h >> (i)) > 1 ? (h >> (i)) : 1)
#define calcPageNum(wh, page_size) (((wh) < (page_size)) ? 1 : ((wh) / (page_size)))

//------------------------------------------------------------------------
// Tensor helpers

template<typename T>
auto prepareCudaArray(const std::vector<T>& InArray)
{
    // For some reason, cudaFree will cause the system crash without giving any message.
    // So we use libtorch to allocate memory instead.
    torch::TensorOptions Options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    torch::Tensor data = torch::from_blob((void*)InArray.data(), {int64_t(sizeof(T) * InArray.size())}, Options).cuda();
    return data;
}
template<bool Const=true, typename TPtr = std::conditional_t<Const, const float*, float*>>
auto prepareCudaTensorArray(const std::vector<torch::Tensor>& pages)
{
    std::vector<TPtr> mip_ptr;
    for (int i = 0; i < pages.size(); i++)
    {
        bool has_tensor = pages[i].defined() && pages[i].nbytes() && pages[i].is_cuda();
        mip_ptr.push_back(has_tensor ? (float*)pages[i].data_ptr() : NULL);
    }
    return prepareCudaArray(mip_ptr);
}

template<bool Const=true, typename TPtr = std::conditional_t<Const, const float*, float*>>
auto prepareCudaTensorArray(const std::vector<std::vector<torch::Tensor>>& pages)
{
    using TUniPtr = decltype(prepareCudaTensorArray<Const>(std::vector<torch::Tensor>()));
    std::vector<TUniPtr> mip_ptr;
    for (int mip = 0; mip < pages.size(); mip++)
    {
        mip_ptr.emplace_back(prepareCudaTensorArray<Const>(pages[mip]));
    }
    return mip_ptr;
}

static int calculateMaxMipLevel(int width, int height, int mipLevelLimit)
{

    if (mipLevelLimit == 0)
        return 0;

    int w = width;
    int h = height;

    int level = 0;
    while ((w|h) > 1)
    {
        // Current level.
        level += 1;

        // Downsample.
        if (w > 1) w >>= 1;
        if (h > 1) h >>= 1;

        if (mipLevelLimit >= 0 && level == mipLevelLimit)
            break;
    }

    return level;
}

//------------------------------------------------------------------------
// normal_tangent.cu

void CalculateNormalKernel(const NormalKernelParams p);
void CalculateNormalGradKernel(const NormalKernelParams p);
void CalculateTangentKernel(const TangentKernelParams p);
void CalculateTangentGradKernel(const TangentKernelParams p);

std::vector<torch::Tensor> calculate_normal_fwd(std::vector<torch::Tensor> ClusterPositions, std::vector<torch::Tensor> ClusterIndexes)
{
    CHECK_DEVICE(ClusterPositions);
    CHECK_DEVICE(ClusterIndexes);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(ClusterPositions[0]));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    std::vector<torch::Tensor> OutNormals;

    // Calculate normals for each cluster
    for (int i = 0; i < ClusterPositions.size(); ++i)
    {
        NormalKernelParams p{};
        p.Positions = ClusterPositions[i].data_ptr<float>();
        p.Indexes = ClusterIndexes[i].data_ptr<int>();
        p.NumTriangles = ClusterIndexes[i].size(0);
        torch::Tensor Normal = torch::zeros_like(ClusterPositions[i]);
        p.Normals = Normal.data_ptr<float>();

        void* args[] = { &p };
        dim3 blockSize(BLOCK_X * BLOCK_Y, 1, 1);
        dim3 gridSize(std::ceil(p.NumTriangles / (float)blockSize.x), 1, 1);
        CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)CalculateNormalKernel, gridSize, blockSize, args, 0, stream));
        OutNormals.push_back(Normal);
    }

    return OutNormals;
}

std::vector<torch::Tensor> calculate_normal_bwd(std::vector<torch::Tensor> ClusterPositions, std::vector<torch::Tensor> ClusterIndexes, std::vector<torch::Tensor> ClusterNormalGrads)
{
    CHECK_DEVICE(ClusterPositions);
    CHECK_DEVICE(ClusterIndexes);
    CHECK_DEVICE(ClusterNormalGrads);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(ClusterPositions[0]));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    std::vector<torch::Tensor> OutClusterPositionGrads;

    for (int i = 0; i < ClusterPositions.size(); ++i)
    {
        NormalKernelParams p{};
        p.Positions = ClusterPositions[i].data_ptr<float>();
        p.NormalsGrad = ClusterNormalGrads[i].data_ptr<float>();
        p.Indexes = ClusterIndexes[i].data_ptr<int>();
        p.NumTriangles = ClusterIndexes[i].size(0);
        torch::Tensor PositionGrad = torch::zeros_like(ClusterPositions[i]);
        p.PositionsGrad = PositionGrad.data_ptr<float>();

        void* args[] = { &p };
        dim3 blockSize(BLOCK_X * BLOCK_Y, 1, 1);
        dim3 gridSize(std::ceil(p.NumTriangles / (float)blockSize.x), 1, 1);
        CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)CalculateNormalGradKernel, gridSize, blockSize, args, 0, stream));
        OutClusterPositionGrads.push_back(PositionGrad);
    }

    return OutClusterPositionGrads;
}

std::vector<torch::Tensor> calculate_tangent_fwd(std::vector<torch::Tensor> ClusterPositions, std::vector<torch::Tensor> ClusterTexCoords, std::vector<torch::Tensor> ClusterPosIndexes, std::vector<torch::Tensor> ClusterUVIndexes)
{
    CHECK_DEVICE(ClusterPositions);
    CHECK_DEVICE(ClusterTexCoords);
    CHECK_DEVICE(ClusterPosIndexes);
    CHECK_DEVICE(ClusterUVIndexes);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(ClusterPositions[0]));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    std::vector<torch::Tensor> OutTangents;

    // Calculate tangents for each cluster
    for (int i = 0; i < ClusterPositions.size(); ++i)
    {
        TangentKernelParams p{};
        p.Positions = ClusterPositions[i].data_ptr<float>();
        p.TexCoords = ClusterTexCoords[i].data_ptr<float>();
        p.PosIndexes = ClusterPosIndexes[i].data_ptr<int>();
        p.UVIndexes = ClusterUVIndexes[i].data_ptr<int>();
        p.NumTriangles = ClusterPosIndexes[i].size(0);
        torch::Tensor Tangent = torch::zeros_like(ClusterPositions[i]);
        p.Tangents = Tangent.data_ptr<float>();
        p.ClusterIdx = i;

        void* args[] = { &p };
        dim3 blockSize(BLOCK_X * BLOCK_Y, 1, 1);
        dim3 gridSize(std::ceil(p.NumTriangles / (float)blockSize.x), 1, 1);
        CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)CalculateTangentKernel, gridSize, blockSize, args, 0, stream));
        OutTangents.push_back(Tangent);
    }

    return OutTangents;
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> calculate_tangent_bwd(
    std::vector<torch::Tensor> ClusterPositions, std::vector<torch::Tensor> ClusterTexCoords, 
    std::vector<torch::Tensor> ClusterPosIndexes, std::vector<torch::Tensor> ClusterUVIndexes, 
    std::vector<torch::Tensor> ClusterTangentGrads)
{
    CHECK_DEVICE(ClusterPositions);
    CHECK_DEVICE(ClusterTexCoords);
    CHECK_DEVICE(ClusterPosIndexes);
    CHECK_DEVICE(ClusterUVIndexes);
    CHECK_DEVICE(ClusterTangentGrads);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(ClusterPositions[0]));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    std::vector<torch::Tensor> OutClusterPositionGrads;
    std::vector<torch::Tensor> OutClusterTexCoordGrads;

    for (int i = 0; i < ClusterPositions.size(); ++i)
    {
        torch::Tensor PositionGrad = torch::zeros_like(ClusterPositions[i]);
        torch::Tensor TexCoordGrad = torch::zeros_like(ClusterTexCoords[i]);
        TangentKernelParams p{};
        p.Positions = ClusterPositions[i].data_ptr<float>();
        p.TexCoords = ClusterTexCoords[i].data_ptr<float>();
        p.PosIndexes = ClusterPosIndexes[i].data_ptr<int>();
        p.UVIndexes = ClusterUVIndexes[i].data_ptr<int>();
        p.TangentsGrad = ClusterTangentGrads[i].data_ptr<float>();
        p.NumTriangles = ClusterPosIndexes[i].size(0);
        p.PositionsGrad = PositionGrad.data_ptr<float>();
        p.TexCoordsGrad = TexCoordGrad.data_ptr<float>();
        p.ClusterIdx = i;

        void* args[] = { &p };
        dim3 blockSize(BLOCK_X * BLOCK_Y, 1, 1);
        dim3 gridSize(std::ceil(p.NumTriangles / (float)blockSize.x), 1, 1);
        CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)CalculateTangentGradKernel, gridSize, blockSize, args, 0, stream));
        OutClusterPositionGrads.push_back(PositionGrad);
        OutClusterTexCoordGrads.push_back(TexCoordGrad);
    }

    return { OutClusterPositionGrads, OutClusterTexCoordGrads };
}

void VirtualShadowMappingFeedbackNearest(const VirtualShadowMapFeedbackKernalParams p);
void VirtualShadowMappingFeedbackLinear(const VirtualShadowMapFeedbackKernalParams p);

std::tuple<std::vector<torch::Tensor>, torch::Tensor> virtual_shadow_map_feedback(torch::Tensor camera_pos, torch::Tensor gb_pos, torch::Tensor shadow_map_uv, int filter_mode, int vsm_height, int vsm_width, int page_size_x, int page_size_y, int max_mip_level, torch::Tensor mask, float first_level_extent)
{
    CHECK_DEVICE(camera_pos);
    CHECK_DEVICE(gb_pos);
    CHECK_DEVICE(shadow_map_uv);
    TORCH_CHECK(vsm_height>0 && (vsm_height & (vsm_height-1))==0, "virtual_shadow_map_feedback: VSM height must be power of two.");
    TORCH_CHECK(vsm_width>0 && (vsm_width & (vsm_width-1))==0, "virtual_shadow_map_feedback: VSM width must be power of two.");
    TORCH_CHECK(page_size_y>0 && (page_size_y & (page_size_y-1))==0, "virtual_shadow_map_feedback: Page Y must be power of two.");
    TORCH_CHECK(page_size_x>0 && (page_size_x & (page_size_x-1))==0, "virtual_shadow_map_feedback: Page X must be power of two.");

    TORCH_CHECK(gb_pos.sizes().size()==4 && gb_pos.size(0) > 0 && gb_pos.size(1) > 0 && gb_pos.size(2) > 0 && gb_pos.size(3) == 3, "gb_pos must have shape [minibatch_size, height, width, 3]");
    TORCH_CHECK(camera_pos.sizes().size()==2 && camera_pos.size(0) > 0 && camera_pos.size(1) == 3, "camera_pos must have shape [minibatch_size, 3]");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(gb_pos));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    max_mip_level = calculateMaxMipLevel(vsm_width, vsm_height, max_mip_level);
    
    VirtualShadowMapFeedbackKernalParams p{};
    p.gb_pos = gb_pos.data_ptr<float>();
    p.camera_pos = camera_pos.data_ptr<float>();
    if (mask.defined() && mask.numel() > 0)
        p.mask = mask.data_ptr<bool>();
    else
        p.mask = NULL;
    p.shadow_map_uv = shadow_map_uv.data_ptr<float>();
    p.max_mipmap_level = max_mip_level;
    p.n = gb_pos.size(0);
    p.imgHeight = gb_pos.size(1);
    p.imgWidth = gb_pos.size(2);
    p.vsmHeight = vsm_height;
    p.vsmWidth = vsm_width;
    p.page_size_x = page_size_x;
    p.page_size_y = page_size_y;
    p.first_level_extent = first_level_extent;

    TORCH_CHECK(p.vsmWidth % p.page_size_x == 0, "The width of VSM must be a multiple of page_size_x");
    TORCH_CHECK(p.vsmHeight % p.page_size_y == 0, "The height of VSM must be a multiple of page_size_y");
    TORCH_CHECK(shadow_map_uv.sizes().size()==4 && 
                shadow_map_uv.size(0) > 0 && 
                shadow_map_uv.size(1) == p.imgHeight && 
                shadow_map_uv.size(2) == p.imgWidth && 
                shadow_map_uv.size(3) == 2, "shadow_map_uv must have shape [minibatch_size, height, width, 2]");

    torch::Tensor vsm_mip_levels = torch::zeros({p.n, p.imgHeight, p.imgWidth}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    p.vsm_mip_levels = vsm_mip_levels.data_ptr<float>();

    std::vector<torch::Tensor> Out(max_mip_level+1);
    for (int mip = 0; mip <= max_mip_level; mip++)
    {
        int2 sz_mip = calcMipLevelSize(vsm_width, vsm_height, mip);
        int width_mip = sz_mip.x;
        int height_mip = sz_mip.y;
        int page_num_y_mip = calcPageNum(height_mip, page_size_y);
        int page_num_x_mip = calcPageNum(width_mip, page_size_x);
        int page_num_mip = page_num_y_mip * page_num_x_mip;
        int numPages = page_num_mip;
        p.num_pages[mip] = numPages;
        torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);
        torch::Tensor feedback = torch::zeros({p.n, numPages}, opts);
        Out[mip] = feedback;
        p.feedback[mip] = feedback.data_ptr<bool>();
    }

    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.imgWidth, p.imgHeight);
    dim3 gridSize  = getLaunchGridSize(blockSize, p.imgWidth, p.imgHeight, p.n);

    void* args[] = { &p };
    void* func_tbl[] = { 
        (void*)VirtualShadowMappingFeedbackNearest, 
        (void*)VirtualShadowMappingFeedbackLinear 
    };
    CHECK_CUDA_ERROR(cudaLaunchKernel(func_tbl[filter_mode], gridSize, blockSize, args, 0, stream));
    return {Out, vsm_mip_levels};
}

struct CppThread
{
    CppThread() = default;
    CppThread(std::future<void>&& t) : thread(std::move(t)) {}
    std::future<void> thread;
    void join()
    {
        thread.get();
    }
    bool done()
    {
        return thread.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }
};

std::unique_ptr<CppThread> async_add_(std::vector<torch::Tensor> input, std::vector<torch::Tensor> other)
{
    std::future<void> thread = std::async([input, other](){
        torch::NoGradGuard no_grad;
        for (int i = 0; i < input.size(); i++)
        {
            input[i].add_(other[i]);
        }
    });
    std::unique_ptr<CppThread> wrapper = std::make_unique<CppThread>(std::move(thread));
    return wrapper;
}

std::unique_ptr<CppThread> async_copy_(std::vector<torch::Tensor> self, std::vector<torch::Tensor> src)
{
    std::future<void> thread = std::async([self, src](){
        torch::NoGradGuard no_grad;
        for (int i = 0; i < src.size(); i++)
        {
            self[i].copy_(src[i], true);
        }
    });
    std::unique_ptr<CppThread> wrapper = std::make_unique<CppThread>(std::move(thread));
    return wrapper;
}

std::unique_ptr<CppThread> async_multi_tensor_adam(
    std::vector<torch::Tensor> params, 
    std::vector<torch::Tensor> grads, 
    std::vector<torch::Tensor> exp_avgs, 
    std::vector<torch::Tensor> exp_avg_sqs, 
    std::vector<int64_t> steps, 
    double beta1, double beta2, double lr, double eps)
{
    std::future<void> thread = std::async([params, grads, exp_avgs, exp_avg_sqs, steps, lr, beta1, beta2, eps](){
        torch::NoGradGuard no_grad;
        size_t num_params = params.size();
        for (size_t i = 0; i < num_params; i++)
        {
            auto& grad = grads[i];
            auto& p = params[i];
            auto& exp_avg = exp_avgs[i];
            auto& exp_avg_sq = exp_avg_sqs[i];
            auto& step = steps[i];

            double bias_correction1 = 1 - std::pow(beta1, step);
            double bias_correction2 = 1 - std::pow(beta2, step);
            
            exp_avg.mul_(beta1).add_(grad, 1 - beta1);
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1 - beta2);
        
            torch::Tensor denom = (exp_avg_sq.sqrt() / sqrt(bias_correction2)).add_(eps);
        
            double step_size = lr / bias_correction1;
            p.addcdiv_(exp_avg, denom, -step_size);
        }    
    });
    std::unique_ptr<CppThread> wrapper = std::make_unique<CppThread>(std::move(thread));
    return wrapper;
}

template<typename T>
struct CppFuture
{
    CppFuture() = default;
    CppFuture(std::future<T>&& t) : future(std::move(t)) {}
    T get()
    {
        return future.get();
    }
    void wait()
    {
        future.wait();
    }
    bool done()
    {
        return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }
    std::future<T> future;
};
using AsyncToCpuFuture = CppFuture<std::vector<torch::Tensor>>;
std::unique_ptr<AsyncToCpuFuture> async_to_cpu(std::vector<torch::Tensor> input, bool pin_memory)
{
    std::future<std::vector<torch::Tensor>> future = std::async([input, pin_memory]() mutable {
        std::vector<torch::Tensor> res;
        for (int i = 0; i < input.size(); i++)
        {
            torch::Tensor pinned = torch::empty_like(input[i], torch::TensorOptions().device(torch::kCPU).pinned_memory(pin_memory));
            pinned.copy_(input[i], true);
            res.push_back(pinned);
            input[i] = torch::Tensor();
        }
        input.clear();
        return res;
    });
    auto wrapper = std::make_unique<AsyncToCpuFuture>(std::move(future));
    return wrapper;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<CppThread>(m, "CppThread")
        .def(pybind11::init<>())
        .def("join", &CppThread::join)
        .def("done", &CppThread::done);
    pybind11::class_<AsyncToCpuFuture>(m, "AsyncToCpuFuture")
        .def(pybind11::init<>())
        .def("get", &AsyncToCpuFuture::get)
        .def("done", &AsyncToCpuFuture::done)
        .def("wait", &AsyncToCpuFuture::wait);

    m.def("calculate_normal_fwd", &calculate_normal_fwd, "calculate_normal_fwd");
    m.def("calculate_normal_bwd", &calculate_normal_bwd, "calculate_normal_bwd");
    m.def("calculate_tangent_fwd", &calculate_tangent_fwd, "calculate_tangent_fwd");
    m.def("calculate_tangent_bwd", &calculate_tangent_bwd, "calculate_tangent_bwd");
    m.def("virtual_shadow_map_feedback", &virtual_shadow_map_feedback, "virtual_shadow_map_feedback");
    m.def("async_add_", &async_add_, "async_add_");
    m.def("async_copy_", &async_copy_, "async_copy_");
    m.def("async_multi_tensor_adam", &async_multi_tensor_adam, "async_multi_tensor_adam");
    m.def("async_to_cpu", &async_to_cpu, "async_to_cpu");
}