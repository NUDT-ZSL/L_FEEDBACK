#include "renderer.hpp"

#include "errors.hpp"
#include "hash.hpp"
#include "json.hpp"
#include "math.hpp"
#include "png.hpp"
#include "scene.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace vlr {
namespace {

constexpr float kReferenceStepMeters = 0.030f;
constexpr int kReferenceMaxSteps = 512;
constexpr std::uint32_t kNoiseDimension = 32;
constexpr const char* kRecoveryRawName = "volume_light_recovery.rgba";
constexpr const char* kRecoveryMetaName = "volume_light_recovery.json";
constexpr const char* kRecoveryReadyName = "volume_light_recovery.ready";

double nowMilliseconds() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               Clock::now().time_since_epoch()).count();
}

std::string apiVersionString(std::uint32_t version) {
    std::ostringstream out;
    out << VK_VERSION_MAJOR(version) << "." << VK_VERSION_MINOR(version) << "."
        << VK_VERSION_PATCH(version);
    return out.str();
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void*) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "validation: " << callbackData->pMessage << "\n";
    }
    return VK_FALSE;
}

struct AllocationHeader {
    void* base = nullptr;
    std::size_t size = 0;
};

struct AllocationState {
    std::uint64_t calls = 0;
    std::int64_t live = 0;
    std::int64_t failAtCall = -1;
};

void* alignedVulkanAllocate(void* userData, std::size_t size, std::size_t alignment,
                            VkSystemAllocationScope) {
    auto* state = static_cast<AllocationState*>(userData);
    ++state->calls;
    if (state->failAtCall >= 0 &&
        state->calls == static_cast<std::uint64_t>(state->failAtCall)) {
        return nullptr;
    }
    alignment = std::max<std::size_t>(alignment, alignof(AllocationHeader));
    const std::size_t rawSize = size + alignment + sizeof(AllocationHeader);
    void* raw = std::malloc(rawSize);
    if (raw == nullptr) {
        return nullptr;
    }
    const auto base = reinterpret_cast<std::uintptr_t>(raw);
    const auto start = base + sizeof(AllocationHeader);
    const auto aligned = (start + alignment - 1u) & ~(alignment - 1u);
    auto* header = reinterpret_cast<AllocationHeader*>(aligned - sizeof(AllocationHeader));
    header->base = raw;
    header->size = size;
    ++state->live;
    return reinterpret_cast<void*>(aligned);
}

void* alignedVulkanReallocate(void* userData, void* original, std::size_t size,
                              std::size_t alignment, VkSystemAllocationScope scope) {
    if (original == nullptr) {
        return alignedVulkanAllocate(userData, size, alignment, scope);
    }
    if (size == 0) {
        return nullptr;
    }
    auto* state = static_cast<AllocationState*>(userData);
    const auto* oldHeader = reinterpret_cast<const AllocationHeader*>(
        reinterpret_cast<std::uintptr_t>(original) - sizeof(AllocationHeader));
    const std::size_t oldSize = oldHeader->size;
    void* replacement = alignedVulkanAllocate(userData, size, alignment, scope);
    if (replacement == nullptr) {
        return nullptr;
    }
    std::memcpy(replacement, original, std::min(oldSize, size));
    std::free(oldHeader->base);
    --state->live;
    return replacement;
}

void alignedVulkanFree(void* userData, void* memory) {
    if (memory == nullptr) {
        return;
    }
    auto* state = static_cast<AllocationState*>(userData);
    auto* header = reinterpret_cast<AllocationHeader*>(
        reinterpret_cast<std::uintptr_t>(memory) - sizeof(AllocationHeader));
    std::free(header->base);
    --state->live;
}

struct Vertex {
    Vec3 position;
    Vec3 normal;
    Vec3 albedo;
};

struct ImageResource {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::size_t memorySize = 0;
    VkImageView view = VK_NULL_HANDLE;
};

struct BufferResource {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::size_t memorySize = 0;
    void* mapped = nullptr;
};

struct FrameResources {
    BufferResource uniform;
    BufferResource sampleCounter;
    BufferResource readback;
    ImageResource gColor;
    ImageResource gDepth;
    ImageResource volume;
    ImageResource output;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkFramebuffer gbufferFramebuffer = VK_NULL_HANDLE;
    VkFramebuffer compositeFramebuffer = VK_NULL_HANDLE;
    std::vector<std::uint8_t> pixels;
    std::uint32_t sampleCount = 0;
    double frameMilliseconds = 0.0;
};

struct alignas(16) SceneUbo {
    Mat4 viewProj;
    Mat4 invViewProj;
    Mat4 dirLightViewProj;
    Mat4 pointFaceViewProj[6];
    Vec4 cameraPos;
    Vec4 cameraParams;
    Vec4 dirLightDirection;
    Vec4 dirLightColor;
    Vec4 pointLightPosition;
    Vec4 pointLightColor;
    Vec4 mediumCoefficients;
    Vec4 volumeBoundsMin;
    Vec4 volumeBoundsMax;
    Vec4 noiseParams;
    Vec4 shadowParams;
    Vec4 pointShadowParams;
    Vec4 toneParams;
    Vec4 backgroundColor;
    Vec4 renderParams;
};

static_assert(sizeof(SceneUbo) == 816, "SceneUbo must match the GLSL std140 layout");

struct ImageDesc {
    std::uint32_t width = 1;
    std::uint32_t height = 1;
    std::uint32_t depth = 1;
    std::uint32_t layers = 1;
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    VkImageUsageFlags usage = 0;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D;
    VkImageCreateFlags flags = 0;
    std::uint32_t baseArrayLayer = 0;
};

} // namespace

struct Renderer::Impl {
    explicit Impl(RendererOptions optionsIn) : options(std::move(optionsIn)) {}

    RendererOptions options;
    AllocationState allocationState;
    VkAllocationCallbacks allocator{};
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    PFN_vkDestroyDebugUtilsMessengerEXT destroyDebugMessenger = nullptr;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queueFamily = 0;
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    VkPhysicalDeviceProperties physicalProperties{};
    bool validationEnabled = false;
    bool deviceLost = false;
    int liveResources = 0;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer shadowCommandBuffer = VK_NULL_HANDLE;
    VkFence shadowFence = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout framePipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout shadowPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

    VkRenderPass directionalShadowRenderPass = VK_NULL_HANDLE;
    VkRenderPass pointShadowRenderPass = VK_NULL_HANDLE;
    VkRenderPass gbufferRenderPass = VK_NULL_HANDLE;
    VkRenderPass compositeRenderPass = VK_NULL_HANDLE;

    ImageResource directionalShadow;
    ImageResource pointShadow;
    std::array<VkImageView, 6> pointShadowViews{};
    VkFramebuffer directionalShadowFramebuffer = VK_NULL_HANDLE;
    std::array<VkFramebuffer, 6> pointShadowFramebuffers{};

    VkSampler noiseSampler = VK_NULL_HANDLE;
    VkSampler depthSampler = VK_NULL_HANDLE;
    VkSampler colorSampler = VK_NULL_HANDLE;

    ImageResource noiseImage;
    BufferResource vertexBuffer;
    BufferResource indexBuffer;
    std::uint32_t indexCount = 0;

    VkPipeline directionalShadowPipeline = VK_NULL_HANDLE;
    VkPipeline pointShadowPipeline = VK_NULL_HANDLE;
    VkPipeline gbufferPipeline = VK_NULL_HANDLE;
    VkPipeline compositePipeline = VK_NULL_HANDLE;
    VkPipeline volumePipeline = VK_NULL_HANDLE;
    std::vector<VkShaderModule> transientShaderModules;

    std::vector<FrameResources> frames;
    Scene scene;
    NoiseVolume noise;
    std::filesystem::path outputPngPath;
    std::filesystem::path statsPath;
    std::filesystem::path referencePngPath;
    std::filesystem::path shaderDirectory;
    std::filesystem::path recoveryDirectory;
    SceneUbo sceneUbo{};
    double shadowMilliseconds = 0.0;
    bool initialized = false;
    bool cleanupDone = false;
    std::vector<std::uint8_t> lastRecoveryPixels;
    std::string lastRecoveryPixelHash;
    std::uint32_t lastRecoveryFrame = 0;
    std::array<std::uint32_t, 2> mainSampleCounts{};
    std::array<double, 2> mainFrameMilliseconds{};
    std::array<std::string, 2> mainPixelHashes{};
    std::uint32_t referenceSampleCount = 0;
    double referenceFrameMilliseconds = 0.0;

    ~Impl() { cleanup(); }

    void check(VkResult result, const std::string& operation) {
        if (result == VK_SUCCESS) {
            return;
        }
        if (result == VK_ERROR_DEVICE_LOST) {
            deviceLost = true;
            fail(ErrorCode::DeviceLost,
                 operation + " returned VK_ERROR_DEVICE_LOST" + recoverySuffix());
        }
        if (result == VK_ERROR_OUT_OF_HOST_MEMORY ||
            result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
            fail(ErrorCode::Allocation,
                 operation + " returned VK_ERROR_OUT_OF_" +
                 (result == VK_ERROR_OUT_OF_HOST_MEMORY ? std::string("HOST_MEMORY")
                                                        : std::string("DEVICE_MEMORY")));
        }
        if (result == VK_ERROR_OUT_OF_POOL_MEMORY ||
            result == VK_ERROR_FRAGMENTED_POOL) {
            fail(ErrorCode::Descriptor,
                 operation + " exhausted the descriptor pool");
        }
        fail(ErrorCode::Vulkan,
             operation + " failed with VkResult " + std::to_string(result));
    }

    std::string recoverySuffix() const {
        if (lastRecoveryPixels.empty()) {
            return "; no complete frame is available for recovery";
        }
        return "; recovery frame=" + std::to_string(lastRecoveryFrame) +
               " path=" + (recoveryDirectory / kRecoveryRawName).string();
    }

    VkBuffer createBufferTracked(const VkBufferCreateInfo& info) {
        VkBuffer buffer = VK_NULL_HANDLE;
        check(vkCreateBuffer(device, &info, &allocator, &buffer), "vkCreateBuffer");
        ++liveResources;
        return buffer;
    }

    VkImage createImageTracked(const VkImageCreateInfo& info) {
        VkImage image = VK_NULL_HANDLE;
        check(vkCreateImage(device, &info, &allocator, &image), "vkCreateImage");
        ++liveResources;
        return image;
    }

    VkImageView createImageViewTracked(const VkImageViewCreateInfo& info) {
        VkImageView view = VK_NULL_HANDLE;
        check(vkCreateImageView(device, &info, &allocator, &view), "vkCreateImageView");
        ++liveResources;
        return view;
    }

    VkSampler createSamplerTracked(const VkSamplerCreateInfo& info) {
        VkSampler sampler = VK_NULL_HANDLE;
        check(vkCreateSampler(device, &info, &allocator, &sampler), "vkCreateSampler");
        ++liveResources;
        return sampler;
    }

    VkRenderPass createRenderPassTracked(const VkRenderPassCreateInfo& info) {
        VkRenderPass pass = VK_NULL_HANDLE;
        check(vkCreateRenderPass(device, &info, &allocator, &pass), "vkCreateRenderPass");
        ++liveResources;
        return pass;
    }

    VkFramebuffer createFramebufferTracked(const VkFramebufferCreateInfo& info) {
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        check(vkCreateFramebuffer(device, &info, &allocator, &framebuffer),
              "vkCreateFramebuffer");
        ++liveResources;
        return framebuffer;
    }

    VkPipelineLayout createPipelineLayoutTracked(const VkPipelineLayoutCreateInfo& info) {
        VkPipelineLayout layout = VK_NULL_HANDLE;
        check(vkCreatePipelineLayout(device, &info, &allocator, &layout),
              "vkCreatePipelineLayout");
        ++liveResources;
        return layout;
    }

    VkDescriptorSetLayout createDescriptorSetLayoutTracked(
        const VkDescriptorSetLayoutCreateInfo& info) {
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        check(vkCreateDescriptorSetLayout(device, &info, &allocator, &layout),
              "vkCreateDescriptorSetLayout");
        ++liveResources;
        return layout;
    }

    VkPipeline createGraphicsPipelineTracked(
        VkPipelineCache cache, const VkGraphicsPipelineCreateInfo& info) {
        VkPipeline pipeline = VK_NULL_HANDLE;
        check(vkCreateGraphicsPipelines(device, cache, 1, &info, &allocator, &pipeline),
              "vkCreateGraphicsPipelines");
        ++liveResources;
        return pipeline;
    }

    VkPipeline createComputePipelineTracked(
        VkPipelineCache cache, const VkComputePipelineCreateInfo& info) {
        VkPipeline pipeline = VK_NULL_HANDLE;
        check(vkCreateComputePipelines(device, cache, 1, &info, &allocator, &pipeline),
              "vkCreateComputePipelines");
        ++liveResources;
        return pipeline;
    }

    VkDescriptorPool createDescriptorPoolTracked(const VkDescriptorPoolCreateInfo& info) {
        VkDescriptorPool pool = VK_NULL_HANDLE;
        check(vkCreateDescriptorPool(device, &info, &allocator, &pool),
              "vkCreateDescriptorPool");
        ++liveResources;
        return pool;
    }

    VkCommandPool createCommandPoolTracked(const VkCommandPoolCreateInfo& info) {
        VkCommandPool pool = VK_NULL_HANDLE;
        check(vkCreateCommandPool(device, &info, &allocator, &pool),
              "vkCreateCommandPool");
        ++liveResources;
        return pool;
    }

    VkFence createFenceTracked(const VkFenceCreateInfo& info) {
        VkFence fence = VK_NULL_HANDLE;
        check(vkCreateFence(device, &info, &allocator, &fence), "vkCreateFence");
        ++liveResources;
        return fence;
    }

    VkShaderModule createShaderModuleTracked(const VkShaderModuleCreateInfo& info) {
        VkShaderModule module = VK_NULL_HANDLE;
        check(vkCreateShaderModule(device, &info, &allocator, &module),
              "vkCreateShaderModule");
        ++liveResources;
        return module;
    }

    VkDeviceMemory allocateMemoryTracked(const VkMemoryAllocateInfo& info) {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        check(vkAllocateMemory(device, &info, &allocator, &memory), "vkAllocateMemory");
        ++liveResources;
        return memory;
    }

    void destroyBuffer(BufferResource& resource) {
        if (resource.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, resource.buffer, &allocator);
            resource.buffer = VK_NULL_HANDLE;
            --liveResources;
        }
        if (resource.memory != VK_NULL_HANDLE) {
            if (resource.mapped != nullptr) {
                vkUnmapMemory(device, resource.memory);
            }
            vkFreeMemory(device, resource.memory, &allocator);
            resource.memory = VK_NULL_HANDLE;
            --liveResources;
        }
        resource.mapped = nullptr;
        resource.memorySize = 0;
    }

    void destroyImage(ImageResource& resource) {
        if (resource.view != VK_NULL_HANDLE) {
            vkDestroyImageView(device, resource.view, &allocator);
            resource.view = VK_NULL_HANDLE;
            --liveResources;
        }
        if (resource.image != VK_NULL_HANDLE) {
            vkDestroyImage(device, resource.image, &allocator);
            resource.image = VK_NULL_HANDLE;
            --liveResources;
        }
        if (resource.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, resource.memory, &allocator);
            resource.memory = VK_NULL_HANDLE;
            --liveResources;
        }
        resource.memorySize = 0;
    }

    void destroyView(VkImageView& view) {
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(device, view, &allocator);
            view = VK_NULL_HANDLE;
            --liveResources;
        }
    }

    void destroyFramebuffer(VkFramebuffer& framebuffer) {
        if (framebuffer != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(device, framebuffer, &allocator);
            framebuffer = VK_NULL_HANDLE;
            --liveResources;
        }
    }

    void destroyRenderPass(VkRenderPass& pass) {
        if (pass != VK_NULL_HANDLE) {
            vkDestroyRenderPass(device, pass, &allocator);
            pass = VK_NULL_HANDLE;
            --liveResources;
        }
    }

    void destroySampler(VkSampler& sampler) {
        if (sampler != VK_NULL_HANDLE) {
            vkDestroySampler(device, sampler, &allocator);
            sampler = VK_NULL_HANDLE;
            --liveResources;
        }
    }

    void destroyPipeline(VkPipeline& pipeline) {
        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, pipeline, &allocator);
            pipeline = VK_NULL_HANDLE;
            --liveResources;
        }
    }

    void destroyPipelineLayout(VkPipelineLayout& layout) {
        if (layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, layout, &allocator);
            layout = VK_NULL_HANDLE;
            --liveResources;
        }
    }

    void destroyDescriptorSetLayout(VkDescriptorSetLayout& layout) {
        if (layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, layout, &allocator);
            layout = VK_NULL_HANDLE;
            --liveResources;
        }
    }

    void destroyDescriptorPool(VkDescriptorPool& pool) {
        if (pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, pool, &allocator);
            pool = VK_NULL_HANDLE;
            --liveResources;
        }
    }

    void destroyCommandPool(VkCommandPool& pool) {
        if (pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, pool, &allocator);
            pool = VK_NULL_HANDLE;
            --liveResources;
        }
    }

    void destroyFence(VkFence& fence) {
        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(device, fence, &allocator);
            fence = VK_NULL_HANDLE;
            --liveResources;
        }
    }

    void destroyShaderModule(VkShaderModule& module) {
        if (module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, module, &allocator);
            module = VK_NULL_HANDLE;
            --liveResources;
        }
    }

    std::uint32_t findMemoryType(std::uint32_t typeBits, VkMemoryPropertyFlags properties) const {
        for (std::uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if ((typeBits & (1u << i)) != 0u &&
                (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        fail(ErrorCode::Allocation, "no compatible Vulkan memory type");
    }

    void createBuffer(BufferResource& resource, std::size_t size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties) {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        resource.buffer = createBufferTracked(info);

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, resource.buffer, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, properties);
        resource.memory = allocateMemoryTracked(allocation);
        resource.memorySize = static_cast<std::size_t>(requirements.size);
        check(vkBindBufferMemory(device, resource.buffer, resource.memory, 0),
              "vkBindBufferMemory");
        if ((properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0u) {
            check(vkMapMemory(device, resource.memory, 0, VK_WHOLE_SIZE, 0, &resource.mapped),
                  "vkMapMemory");
        }
    }

    void createImage(ImageResource& resource, const ImageDesc& desc,
                     VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.flags = desc.flags;
        info.imageType = desc.depth > 1 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
        info.format = desc.format;
        info.extent = {desc.width, desc.height, desc.depth};
        info.mipLevels = 1;
        info.arrayLayers = desc.layers;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = desc.usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        resource.image = createImageTracked(info);

        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device, resource.image, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, properties);
        resource.memory = allocateMemoryTracked(allocation);
        resource.memorySize = static_cast<std::size_t>(requirements.size);
        check(vkBindImageMemory(device, resource.image, resource.memory, 0),
              "vkBindImageMemory");

        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = resource.image;
        viewInfo.viewType = desc.viewType;
        viewInfo.format = desc.format;
        viewInfo.subresourceRange.aspectMask = desc.aspect;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = desc.baseArrayLayer;
        viewInfo.subresourceRange.layerCount = desc.layers;
        resource.view = createImageViewTracked(viewInfo);
    }

    void flushBuffer(BufferResource& resource) {
        if (resource.mapped != nullptr) {
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            range.memory = resource.memory;
            range.offset = 0;
            range.size = VK_WHOLE_SIZE;
            check(vkFlushMappedMemoryRanges(device, 1, &range), "vkFlushMappedMemoryRanges");
        }
    }

    void invalidateBuffer(BufferResource& resource) {
        if (resource.mapped != nullptr) {
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            range.memory = resource.memory;
            range.offset = 0;
            range.size = VK_WHOLE_SIZE;
            check(vkInvalidateMappedMemoryRanges(device, 1, &range),
                  "vkInvalidateMappedMemoryRanges");
        }
    }

    void initialize();
    void run();
    void initializeVulkan();
    void createCommandPool();
    void createSharedResources();
    void createFrameResources();
    void createDescriptorResources();
    void uploadGeometryAndNoise();
    void createPipelines();
    void injectAllocationFailure();
    void renderShadowMaps();
    void renderFrame(std::size_t slot, bool reference);
    void writeUbo(std::size_t slot, bool reference);
    void persistRecovery(std::size_t slot);
    void announceRecovery();
    void clearRecovery();
    void writeStats(const std::vector<std::uint8_t>& pngA,
                    const std::vector<std::uint8_t>& pngB,
                    const std::vector<std::uint8_t>& referencePng,
                    double luminanceError);
    void buildGeometry();
    void cleanup();

    std::vector<std::uint8_t> readSpirv(const std::string& filename) {
        if (options.injection == Injection::Shader) {
            fail(ErrorCode::Shader,
                 "injected shader compilation failure while loading " + filename);
        }
        const std::filesystem::path path = shaderDirectory / filename;
        const std::vector<std::uint8_t> bytes = readFileBytes(path);
        if (bytes.size() % 4 != 0 || bytes.size() < 4 ||
            bytes[0] != 0x03 || bytes[1] != 0x02 || bytes[2] != 0x23 ||
            bytes[3] != 0x07) {
            fail(ErrorCode::Shader, "invalid SPIR-V module: " + path.string());
        }
        return bytes;
    }
};

void Renderer::Impl::initialize() {
    scene = loadScene(options.scenePath);
    noise = loadNoise(scene.noisePath, scene.noiseDimension);

    if (options.outputPng) {
        outputPngPath = std::filesystem::absolute(*options.outputPng);
    } else {
        outputPngPath = resolveRelative(scene.sourceDirectory, scene.output.pngPath);
    }
    if (options.statsJson) {
        statsPath = std::filesystem::absolute(*options.statsJson);
    } else {
        statsPath = resolveRelative(scene.sourceDirectory, scene.output.statsPath);
    }
    if (options.referencePng) {
        referencePngPath = std::filesystem::absolute(*options.referencePng);
    } else if (options.outputPng) {
        referencePngPath = outputPngPath;
        referencePngPath.replace_extension(".reference.png");
    } else {
        referencePngPath =
            resolveRelative(scene.sourceDirectory, scene.output.referencePngPath);
    }
    if (options.shaderDirectory) {
        shaderDirectory = std::filesystem::absolute(*options.shaderDirectory);
    } else {
#ifdef RENDERER_DEFAULT_SHADER_DIR
        shaderDirectory = RENDERER_DEFAULT_SHADER_DIR;
#else
        shaderDirectory = std::filesystem::current_path() / "shaders";
#endif
        if (!std::filesystem::exists(shaderDirectory / "volume.comp.spv")) {
            shaderDirectory = std::filesystem::current_path() / "shaders";
        }
    }
    if (options.recoveryDirectory) {
        recoveryDirectory = std::filesystem::absolute(*options.recoveryDirectory);
    } else {
        recoveryDirectory =
            std::filesystem::temp_directory_path() / "volume-light-renderer-recovery";
    }
    std::error_code ec;
    std::filesystem::create_directories(recoveryDirectory, ec);
    if (ec) {
        fail(ErrorCode::Recovery,
             "cannot create recovery directory: " + recoveryDirectory.string());
    }
    announceRecovery();

    if (options.frameCount < 2) {
        fail(ErrorCode::Usage, "at least two frames must be rendered");
    }
    if (options.volumeStepOverride) {
        scene.volume.stepMeters = *options.volumeStepOverride;
        if (scene.volume.stepMeters < 0.01f || scene.volume.stepMeters > 1.0f) {
            fail(ErrorCode::Schema, "volume step override is outside [0.01, 1.0] m");
        }
    }
    if (options.volumeMaxStepsOverride) {
        scene.volume.maxSteps = *options.volumeMaxStepsOverride;
        if (scene.volume.maxSteps < 8 || scene.volume.maxSteps > 2048) {
            fail(ErrorCode::Schema,
                 "volume maxSteps override is outside [8, 2048]");
        }
    }

    initializeVulkan();
    createCommandPool();
    createSharedResources();
    createFrameResources();
    createDescriptorResources();
    uploadGeometryAndNoise();
    createPipelines();
    injectAllocationFailure();
    initialized = true;

    std::cout << "device: " << physicalProperties.deviceName << " api="
              << apiVersionString(physicalProperties.apiVersion)
              << " validation=" << (validationEnabled ? "on" : "off") << "\n";
    std::cout << "scene: " << scene.sourcePath.string() << " schema="
              << scene.schemaVersion << " sha256=" << scene.sourceSha256 << "\n";
}

void Renderer::Impl::initializeVulkan() {
    allocator.pUserData = &allocationState;
    allocator.pfnAllocation = alignedVulkanAllocate;
    allocator.pfnReallocation = alignedVulkanReallocate;
    allocator.pfnFree = alignedVulkanFree;
    allocator.pfnInternalAllocation = nullptr;
    allocator.pfnInternalFree = nullptr;

    std::uint32_t loaderVersion = VK_API_VERSION_1_0;
    const auto enumerateVersion =
        reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
    if (enumerateVersion != nullptr) {
        check(enumerateVersion(&loaderVersion), "vkEnumerateInstanceVersion");
    }
    if (loaderVersion < VK_API_VERSION_1_3) {
        fail(ErrorCode::Unsupported,
             "Vulkan loader is older than 1.3 (" + apiVersionString(loaderVersion) + ")");
    }

    bool validationLayerAvailable = false;
    if (options.enableValidation) {
        std::uint32_t layerCount = 0;
        check(vkEnumerateInstanceLayerProperties(&layerCount, nullptr),
              "vkEnumerateInstanceLayerProperties");
        std::vector<VkLayerProperties> layers(layerCount);
        if (layerCount > 0) {
            check(vkEnumerateInstanceLayerProperties(&layerCount, layers.data()),
                  "vkEnumerateInstanceLayerProperties");
        }
        validationLayerAvailable =
            std::any_of(layers.begin(), layers.end(), [](const VkLayerProperties& layer) {
                return std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0;
            });
    }

    std::uint32_t extensionCount = 0;
    check(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr),
          "vkEnumerateInstanceExtensionProperties");
    std::vector<VkExtensionProperties> extensions(extensionCount);
    if (extensionCount > 0) {
        check(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
                                                    extensions.data()),
              "vkEnumerateInstanceExtensionProperties");
    }
    const bool debugUtilsAvailable =
        std::any_of(extensions.begin(), extensions.end(),
                    [](const VkExtensionProperties& extension) {
                        return std::strcmp(extension.extensionName,
                                           VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0;
                    });

    std::vector<const char*> enabledLayers;
    std::vector<const char*> enabledExtensions;
    if (validationLayerAvailable) {
        enabledLayers.push_back("VK_LAYER_KHRONOS_validation");
        if (debugUtilsAvailable) {
            enabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
    }

    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "volume-shadow-renderer";
    application.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    application.pEngineName = "vlr";
    application.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    application.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pApplicationInfo = &application;
    instanceInfo.enabledLayerCount = static_cast<std::uint32_t>(enabledLayers.size());
    instanceInfo.ppEnabledLayerNames = enabledLayers.data();
    instanceInfo.enabledExtensionCount =
        static_cast<std::uint32_t>(enabledExtensions.size());
    instanceInfo.ppEnabledExtensionNames = enabledExtensions.data();
    check(vkCreateInstance(&instanceInfo, &allocator, &instance), "vkCreateInstance");
    validationEnabled = validationLayerAvailable;

    if (validationEnabled && debugUtilsAvailable) {
        VkDebugUtilsMessengerCreateInfoEXT messengerInfo{
            VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        messengerInfo.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        messengerInfo.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        messengerInfo.pfnUserCallback = debugCallback;
        const auto createMessenger =
            reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        destroyDebugMessenger =
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (createMessenger != nullptr) {
            check(createMessenger(instance, &messengerInfo, &allocator, &debugMessenger),
                  "vkCreateDebugUtilsMessengerEXT");
        }
    }

    std::uint32_t physicalDeviceCount = 0;
    check(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr),
          "vkEnumeratePhysicalDevices");
    if (physicalDeviceCount == 0) {
        fail(ErrorCode::Unsupported, "no Vulkan physical device found");
    }
    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    check(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data()),
          "vkEnumeratePhysicalDevices");

    for (const VkPhysicalDevice candidate : physicalDevices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        if (properties.apiVersion < VK_API_VERSION_1_3) {
            continue;
        }
        std::uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, families.data());
        for (std::uint32_t family = 0; family < familyCount; ++family) {
            if ((families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0u &&
                (families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0u) {
                physicalDevice = candidate;
                physicalProperties = properties;
                queueFamily = family;
                break;
            }
        }
        if (physicalDevice != VK_NULL_HANDLE) {
            break;
        }
    }
    if (physicalDevice == VK_NULL_HANDLE) {
        fail(ErrorCode::Unsupported,
             "no Vulkan 1.3 device with a combined graphics+compute queue");
    }

    const std::array<VkFormat, 6> requiredFormats = {
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R32_SFLOAT,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_R8G8B8A8_UNORM
    };
    const std::array<VkFormatFeatureFlags, 6> requiredFeatures = {
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT,
        VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT,
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT,
        VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_TRANSFER_SRC_BIT,
        VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_TRANSFER_SRC_BIT,
        VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_TRANSFER_SRC_BIT
    };
    for (std::size_t i = 0; i < requiredFormats.size(); ++i) {
        VkFormatProperties formatProperties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, requiredFormats[i],
                                            &formatProperties);
        if ((formatProperties.optimalTilingFeatures & requiredFeatures[i]) !=
            requiredFeatures[i]) {
            fail(ErrorCode::Unsupported,
                 "required Vulkan format feature is unavailable for format " +
                 std::to_string(requiredFormats[i]));
        }
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkPhysicalDeviceVulkan13Features features13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    features13.dynamicRendering = VK_FALSE;
    VkPhysicalDeviceFeatures2 features2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.pNext = &features13;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

    VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceInfo.pNext = &features2;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount = 0;
    deviceInfo.pEnabledFeatures = nullptr;
    check(vkCreateDevice(physicalDevice, &deviceInfo, &allocator, &device),
          "vkCreateDevice");
    vkGetDeviceQueue(device, queueFamily, 0, &queue);
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
}

void Renderer::Impl::createCommandPool() {
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamily;
    commandPool = createCommandPoolTracked(poolInfo);
}

void Renderer::Impl::createSharedResources() {
    const ImageDesc directionalDesc{
        static_cast<std::uint32_t>(scene.shadows.resolution),
        static_cast<std::uint32_t>(scene.shadows.resolution),
        1, 1,
        VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_VIEW_TYPE_2D,
        0, 0
    };
    createImage(directionalShadow, directionalDesc);

    const ImageDesc pointDesc{
        static_cast<std::uint32_t>(scene.shadows.resolution),
        static_cast<std::uint32_t>(scene.shadows.resolution),
        1, 6,
        VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
        0
    };
    createImage(pointShadow, pointDesc);
    for (std::uint32_t face = 0; face < 6; ++face) {
        VkImageViewCreateInfo faceInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        faceInfo.image = pointShadow.image;
        faceInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        faceInfo.format = VK_FORMAT_D32_SFLOAT;
        faceInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        faceInfo.subresourceRange.baseMipLevel = 0;
        faceInfo.subresourceRange.levelCount = 1;
        faceInfo.subresourceRange.baseArrayLayer = face;
        faceInfo.subresourceRange.layerCount = 1;
        pointShadowViews[face] = createImageViewTracked(faceInfo);
    }

    VkSamplerCreateInfo noiseSamplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    noiseSamplerInfo.magFilter = VK_FILTER_LINEAR;
    noiseSamplerInfo.minFilter = VK_FILTER_LINEAR;
    noiseSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    noiseSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    noiseSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    noiseSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    noiseSamplerInfo.maxLod = 0.0f;
    noiseSampler = createSamplerTracked(noiseSamplerInfo);

    VkSamplerCreateInfo depthSamplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    depthSamplerInfo.magFilter = VK_FILTER_NEAREST;
    depthSamplerInfo.minFilter = VK_FILTER_NEAREST;
    depthSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    depthSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    depthSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    depthSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    depthSamplerInfo.maxLod = 0.0f;
    depthSampler = createSamplerTracked(depthSamplerInfo);

    VkSamplerCreateInfo colorSamplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    colorSamplerInfo.magFilter = VK_FILTER_NEAREST;
    colorSamplerInfo.minFilter = VK_FILTER_NEAREST;
    colorSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    colorSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    colorSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    colorSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    colorSamplerInfo.maxLod = 0.0f;
    colorSampler = createSamplerTracked(colorSamplerInfo);

    {
        VkAttachmentDescription attachment{};
        attachment.format = VK_FORMAT_D32_SFLOAT;
        attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        VkAttachmentReference depthReference{0,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.pDepthStencilAttachment = &depthReference;
        std::array<VkSubpassDependency, 2> dependencies{};
        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependencies[0].dstStageMask =
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependencies[1].srcSubpass = 0;
        dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstStageMask =
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        VkRenderPassCreateInfo passInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        passInfo.attachmentCount = 1;
        passInfo.pAttachments = &attachment;
        passInfo.subpassCount = 1;
        passInfo.pSubpasses = &subpass;
        passInfo.dependencyCount = static_cast<std::uint32_t>(dependencies.size());
        passInfo.pDependencies = dependencies.data();
        directionalShadowRenderPass = createRenderPassTracked(passInfo);
    }

    {
        VkAttachmentDescription attachment{};
        attachment.format = VK_FORMAT_D32_SFLOAT;
        attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference depthReference{0,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.pDepthStencilAttachment = &depthReference;
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependency.dstStageMask =
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo passInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        passInfo.attachmentCount = 1;
        passInfo.pAttachments = &attachment;
        passInfo.subpassCount = 1;
        passInfo.pSubpasses = &subpass;
        passInfo.dependencyCount = 1;
        passInfo.pDependencies = &dependency;
        pointShadowRenderPass = createRenderPassTracked(passInfo);
    }

    {
        std::array<VkAttachmentDescription, 2> attachments{};
        attachments[0].format = VK_FORMAT_R16G16B16A16_SFLOAT;
        attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        attachments[1].format = VK_FORMAT_D32_SFLOAT;
        attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkAttachmentReference colorReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depthReference{1,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorReference;
        subpass.pDepthStencilAttachment = &depthReference;
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependency.dstStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask =
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo passInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        passInfo.attachmentCount = static_cast<std::uint32_t>(attachments.size());
        passInfo.pAttachments = attachments.data();
        passInfo.subpassCount = 1;
        passInfo.pSubpasses = &subpass;
        passInfo.dependencyCount = 1;
        passInfo.pDependencies = &dependency;
        gbufferRenderPass = createRenderPassTracked(passInfo);
    }

    {
        VkAttachmentDescription attachment{};
        attachment.format = VK_FORMAT_R8G8B8A8_UNORM;
        attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        VkAttachmentReference colorReference{0,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorReference;
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo passInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        passInfo.attachmentCount = 1;
        passInfo.pAttachments = &attachment;
        passInfo.subpassCount = 1;
        passInfo.pSubpasses = &subpass;
        passInfo.dependencyCount = 1;
        passInfo.pDependencies = &dependency;
        compositeRenderPass = createRenderPassTracked(passInfo);
    }

    {
        VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        info.renderPass = directionalShadowRenderPass;
        info.attachmentCount = 1;
        info.pAttachments = &directionalShadow.view;
        info.width = static_cast<std::uint32_t>(scene.shadows.resolution);
        info.height = static_cast<std::uint32_t>(scene.shadows.resolution);
        info.layers = 1;
        directionalShadowFramebuffer = createFramebufferTracked(info);
    }
    for (std::uint32_t face = 0; face < 6; ++face) {
        VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        info.renderPass = pointShadowRenderPass;
        info.attachmentCount = 1;
        info.pAttachments = &pointShadowViews[face];
        info.width = static_cast<std::uint32_t>(scene.shadows.resolution);
        info.height = static_cast<std::uint32_t>(scene.shadows.resolution);
        info.layers = 1;
        pointShadowFramebuffers[face] = createFramebufferTracked(info);
    }

    {
        std::array<VkDescriptorSetLayoutBinding, 8> bindings{};
        for (std::size_t i = 0; i < bindings.size(); ++i) {
            bindings[i].binding = static_cast<std::uint32_t>(i);
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags =
                VK_SHADER_STAGE_VERTEX_BIT |
                VK_SHADER_STAGE_FRAGMENT_BIT |
                VK_SHADER_STAGE_COMPUTE_BIT;
        }
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        VkDescriptorSetLayoutCreateInfo info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.bindingCount = static_cast<std::uint32_t>(bindings.size());
        info.pBindings = bindings.data();
        descriptorSetLayout = createDescriptorSetLayoutTracked(info);
    }
}

void Renderer::Impl::createFrameResources() {
    const VkExtent2D extent{
        static_cast<std::uint32_t>(scene.output.width),
        static_cast<std::uint32_t>(scene.output.height)
    };
    frames.resize(static_cast<std::size_t>(options.frameCount));
    for (std::size_t slot = 0; slot < frames.size(); ++slot) {
        FrameResources& frame = frames[slot];
        createBuffer(frame.uniform, sizeof(SceneUbo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        createBuffer(frame.sampleCounter, sizeof(std::uint32_t),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        createBuffer(frame.readback,
                     static_cast<std::size_t>(scene.output.width) * scene.output.height * 4u,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        const ImageDesc colorDesc{
            extent.width, extent.height, 1, 1,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_VIEW_TYPE_2D, 0, 0
        };
        createImage(frame.gColor, colorDesc);
        const ImageDesc depthDesc{
            extent.width, extent.height, 1, 1,
            VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT,
            VK_IMAGE_VIEW_TYPE_2D, 0, 0
        };
        createImage(frame.gDepth, depthDesc);
        const ImageDesc volumeDesc{
            extent.width, extent.height, 1, 1,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_VIEW_TYPE_2D, 0, 0
        };
        createImage(frame.volume, volumeDesc);
        const ImageDesc outputDesc{
            extent.width, extent.height, 1, 1,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_VIEW_TYPE_2D, 0, 0
        };
        createImage(frame.output, outputDesc);

        VkFramebufferCreateInfo gbufferInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        gbufferInfo.renderPass = gbufferRenderPass;
        const std::array<VkImageView, 2> gbufferAttachments{
            frame.gColor.view, frame.gDepth.view};
        gbufferInfo.attachmentCount =
            static_cast<std::uint32_t>(gbufferAttachments.size());
        gbufferInfo.pAttachments = gbufferAttachments.data();
        gbufferInfo.width = extent.width;
        gbufferInfo.height = extent.height;
        gbufferInfo.layers = 1;
        frame.gbufferFramebuffer = createFramebufferTracked(gbufferInfo);

        VkFramebufferCreateInfo compositeInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        compositeInfo.renderPass = compositeRenderPass;
        compositeInfo.attachmentCount = 1;
        compositeInfo.pAttachments = &frame.output.view;
        compositeInfo.width = extent.width;
        compositeInfo.height = extent.height;
        compositeInfo.layers = 1;
        frame.compositeFramebuffer = createFramebufferTracked(compositeInfo);

        VkCommandBufferAllocateInfo commandInfo{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        commandInfo.commandPool = commandPool;
        commandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandInfo.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(device, &commandInfo, &frame.commandBuffer),
              "vkAllocateCommandBuffers");

        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        frame.fence = createFenceTracked(fenceInfo);
        frame.pixels.resize(
            static_cast<std::size_t>(scene.output.width) * scene.output.height * 4u);
    }

    VkCommandBufferAllocateInfo shadowCommandInfo{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    shadowCommandInfo.commandPool = commandPool;
    shadowCommandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    shadowCommandInfo.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(device, &shadowCommandInfo, &shadowCommandBuffer),
          "vkAllocateCommandBuffers");
    VkFenceCreateInfo shadowFenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    shadowFence = createFenceTracked(shadowFenceInfo);

    const ImageDesc noiseDesc{
        kNoiseDimension, kNoiseDimension, kNoiseDimension, 1,
        VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_VIEW_TYPE_3D, 0, 0
    };
    createImage(noiseImage, noiseDesc);
}

void Renderer::Impl::createDescriptorResources() {
    const bool injectDescriptorFailure = options.injection == Injection::Descriptor;
    const std::uint32_t setCount = injectDescriptorFailure ? 1u
                                                           : static_cast<std::uint32_t>(frames.size());
    const std::uint32_t perSetMultiplier = injectDescriptorFailure ? 1u : setCount;
    std::array<VkDescriptorPoolSize, 4> sizes{};
    sizes[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, perSetMultiplier};
    sizes[1] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, perSetMultiplier * 5u};
    sizes[2] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, perSetMultiplier};
    sizes[3] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, perSetMultiplier};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = setCount;
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
    poolInfo.pPoolSizes = sizes.data();
    descriptorPool = createDescriptorPoolTracked(poolInfo);

    for (std::size_t slot = 0; slot < frames.size(); ++slot) {
        if (injectDescriptorFailure && slot >= 1) {
            fail(ErrorCode::Descriptor,
                 "injected descriptor set exhaustion after allocating " +
                     std::to_string(slot) + " frame set(s)");
        }
        VkDescriptorSetAllocateInfo allocation{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool = descriptorPool;
        allocation.descriptorSetCount = 1;
        allocation.pSetLayouts = &descriptorSetLayout;
        const VkResult result =
            vkAllocateDescriptorSets(device, &allocation, &frames[slot].descriptorSet);
        check(result, "vkAllocateDescriptorSets");
    }

    for (std::size_t slot = 0; slot < frames.size(); ++slot) {
        FrameResources& frame = frames[slot];
        VkDescriptorBufferInfo uniform{};
        uniform.buffer = frame.uniform.buffer;
        uniform.offset = 0;
        uniform.range = sizeof(SceneUbo);
        VkDescriptorBufferInfo counter{};
        counter.buffer = frame.sampleCounter.buffer;
        counter.offset = 0;
        counter.range = sizeof(std::uint32_t);
        VkDescriptorImageInfo noiseInfo{};
        noiseInfo.sampler = noiseSampler;
        noiseInfo.imageView = noiseImage.view;
        noiseInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo directionalInfo{};
        directionalInfo.sampler = depthSampler;
        directionalInfo.imageView = directionalShadow.view;
        directionalInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo pointInfo{};
        pointInfo.sampler = depthSampler;
        pointInfo.imageView = pointShadow.view;
        pointInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo depthInfo{};
        depthInfo.sampler = depthSampler;
        depthInfo.imageView = frame.gDepth.view;
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo colorInfo{};
        colorInfo.sampler = colorSampler;
        colorInfo.imageView = frame.gColor.view;
        colorInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo volumeInfo{};
        volumeInfo.imageView = frame.volume.view;
        volumeInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 8> writes{};
        for (std::size_t i = 0; i < writes.size(); ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = frame.descriptorSet;
            writes[i].dstBinding = static_cast<std::uint32_t>(i);
            writes[i].descriptorCount = 1;
        }
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo = &uniform;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo = &noiseInfo;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].pImageInfo = &directionalInfo;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].pImageInfo = &pointInfo;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[4].pImageInfo = &depthInfo;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[5].pImageInfo = &colorInfo;
        writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[6].pImageInfo = &volumeInfo;
        writes[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[7].pBufferInfo = &counter;
        vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }
}

void Renderer::Impl::buildGeometry() {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    const float half = scene.ground.sizeMeters * 0.5f;
    const float y = scene.ground.height;
    const std::uint32_t groundBase = static_cast<std::uint32_t>(vertices.size());
    vertices.push_back({{-half, y, -half}, {0.0f, 1.0f, 0.0f}, scene.ground.albedo});
    vertices.push_back({{half, y, -half}, {0.0f, 1.0f, 0.0f}, scene.ground.albedo});
    vertices.push_back({{half, y, half}, {0.0f, 1.0f, 0.0f}, scene.ground.albedo});
    vertices.push_back({{-half, y, half}, {0.0f, 1.0f, 0.0f}, scene.ground.albedo});
    indices.insert(indices.end(), {
        groundBase, groundBase + 1, groundBase + 2,
        groundBase, groundBase + 2, groundBase + 3
    });

    constexpr int longitudeSegments = 32;
    constexpr int latitudeSegments = 16;
    for (const Sphere& sphere : scene.spheres) {
        const std::uint32_t base = static_cast<std::uint32_t>(vertices.size());
        for (int latitude = 0; latitude <= latitudeSegments; ++latitude) {
            const float v = static_cast<float>(latitude) / latitudeSegments;
            const float phi = -static_cast<float>(kPi) * 0.5f +
                              static_cast<float>(kPi) * v;
            const float cosPhi = std::cos(phi);
            const float sinPhi = std::sin(phi);
            for (int longitude = 0; longitude <= longitudeSegments; ++longitude) {
                const float u = static_cast<float>(longitude) / longitudeSegments;
                const float theta = static_cast<float>(2.0 * kPi) * u;
                const Vec3 normal{
                    cosPhi * std::cos(theta),
                    sinPhi,
                    cosPhi * std::sin(theta)
                };
                vertices.push_back({
                    sphere.center + normal * sphere.radius,
                    normal,
                    sphere.albedo
                });
            }
        }
        const std::uint32_t row = longitudeSegments + 1;
        for (int latitude = 0; latitude < latitudeSegments; ++latitude) {
            for (int longitude = 0; longitude < longitudeSegments; ++longitude) {
                const std::uint32_t current =
                    base + static_cast<std::uint32_t>(latitude * row + longitude);
                const std::uint32_t next = current + row;
                indices.insert(indices.end(), {
                    current, next, current + 1,
                    current + 1, next, next + 1
                });
            }
        }
    }

    createBuffer(vertexBuffer, vertices.size() * sizeof(Vertex),
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    createBuffer(indexBuffer, indices.size() * sizeof(std::uint32_t),
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    indexCount = static_cast<std::uint32_t>(indices.size());

    auto upload = [this](const void* data, std::size_t size, BufferResource& destination) {
        BufferResource staging;
        createBuffer(staging, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        std::memcpy(staging.mapped, data, size);
        flushBuffer(staging);

        VkCommandBufferAllocateInfo allocation{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = commandPool;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        VkCommandBuffer command = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(device, &allocation, &command),
              "vkAllocateCommandBuffers");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");
        VkBufferCopy copy{};
        copy.size = size;
        vkCmdCopyBuffer(command, staging.buffer, destination.buffer, 1, &copy);
        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask =
            VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = destination.buffer;
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0,
                             0, nullptr, 1, &barrier, 0, nullptr);
        check(vkEndCommandBuffer(command), "vkEndCommandBuffer");
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence = createFenceTracked(fenceInfo);
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
        check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
              "vkWaitForFences");
        destroyFence(fence);
        vkFreeCommandBuffers(device, commandPool, 1, &command);
        destroyBuffer(staging);
    };

    upload(vertices.data(), vertices.size() * sizeof(Vertex), vertexBuffer);
    upload(indices.data(), indices.size() * sizeof(std::uint32_t), indexBuffer);
}

void Renderer::Impl::uploadGeometryAndNoise() {
    buildGeometry();

    const std::size_t noiseBytes = noise.values.size() * sizeof(float);
    BufferResource staging;
    createBuffer(staging, noiseBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memcpy(staging.mapped, noise.values.data(), noiseBytes);
    flushBuffer(staging);

    VkCommandBufferAllocateInfo allocation{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = commandPool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    check(vkAllocateCommandBuffers(device, &allocation, &command),
          "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");
    VkImageMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.image = noiseImage.image;
    toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransfer.subresourceRange.levelCount = 1;
    toTransfer.subresourceRange.layerCount = 1;
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &toTransfer);
    VkBufferImageCopy copy{};
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent = {kNoiseDimension, kNoiseDimension, kNoiseDimension};
    vkCmdCopyBufferToImage(command, staging.buffer, noiseImage.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    VkImageMemoryBarrier toSample{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toSample.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toSample.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toSample.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toSample.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toSample.image = noiseImage.image;
    toSample.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toSample.subresourceRange.levelCount = 1;
    toSample.subresourceRange.layerCount = 1;
    toSample.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toSample.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toSample);
    check(vkEndCommandBuffer(command), "vkEndCommandBuffer");
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = createFenceTracked(fenceInfo);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
    check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
          "vkWaitForFences");
    destroyFence(fence);
    vkFreeCommandBuffers(device, commandPool, 1, &command);
    destroyBuffer(staging);
}

void Renderer::Impl::createPipelines() {
    VkPipelineLayoutCreateInfo shadowLayoutInfo{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(Mat4);
    shadowLayoutInfo.pushConstantRangeCount = 1;
    shadowLayoutInfo.pPushConstantRanges = &pushRange;
    shadowPipelineLayout = createPipelineLayoutTracked(shadowLayoutInfo);

    VkPipelineLayoutCreateInfo frameLayoutInfo{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    frameLayoutInfo.setLayoutCount = 1;
    frameLayoutInfo.pSetLayouts = &descriptorSetLayout;
    framePipelineLayout = createPipelineLayoutTracked(frameLayoutInfo);

    auto createShader = [this](const char* name, VkShaderStageFlagBits stage) {
        const std::vector<std::uint8_t> bytes = readSpirv(name);
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        info.codeSize = bytes.size();
        info.pCode = reinterpret_cast<const std::uint32_t*>(bytes.data());
        VkShaderModule module = createShaderModuleTracked(info);
        transientShaderModules.push_back(module);
        VkPipelineShaderStageCreateInfo stageInfo{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stageInfo.stage = stage;
        stageInfo.module = module;
        stageInfo.pName = "main";
        return stageInfo;
    };

    auto createShadowPipeline = [&](VkRenderPass renderPass, float bias) {
        VkPipelineShaderStageCreateInfo vertexStage =
            createShader("shadow.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(Vertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        std::array<VkVertexInputAttributeDescription, 1> attributes{};
        attributes[0].location = 0;
        attributes[0].binding = 0;
        attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributes[0].offset = offsetof(Vertex, position);
        VkPipelineVertexInputStateCreateInfo vertexInput{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &binding;
        vertexInput.vertexAttributeDescriptionCount =
            static_cast<std::uint32_t>(attributes.size());
        vertexInput.pVertexAttributeDescriptions = attributes.data();
        VkPipelineInputAssemblyStateCreateInfo assembly{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo raster{
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.0f;
        raster.depthBiasEnable = VK_TRUE;
        raster.depthBiasConstantFactor = bias;
        raster.depthBiasSlopeFactor = 0.5f;
        raster.depthBiasClamp = 0.0f;
        VkPipelineMultisampleStateCreateInfo multisample{
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo depth{
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth.depthTestEnable = VK_TRUE;
        depth.depthWriteEnable = VK_TRUE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS;
        VkPipelineColorBlendStateCreateInfo blend{
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        std::array<VkDynamicState, 2> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
        dynamic.pDynamicStates = dynamicStates.data();
        VkGraphicsPipelineCreateInfo pipelineInfo{
            VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipelineInfo.stageCount = 1;
        pipelineInfo.pStages = &vertexStage;
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &assembly;
        pipelineInfo.pViewportState = &viewport;
        pipelineInfo.pRasterizationState = &raster;
        pipelineInfo.pMultisampleState = &multisample;
        pipelineInfo.pDepthStencilState = &depth;
        pipelineInfo.pColorBlendState = &blend;
        pipelineInfo.pDynamicState = &dynamic;
        pipelineInfo.layout = shadowPipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        return createGraphicsPipelineTracked(VK_NULL_HANDLE, pipelineInfo);
    };

    directionalShadowPipeline =
        createShadowPipeline(directionalShadowRenderPass, scene.shadows.directionalBias);
    pointShadowPipeline =
        createShadowPipeline(pointShadowRenderPass, scene.shadows.pointBias);

    {
        VkPipelineShaderStageCreateInfo stages[2] = {
            createShader("gbuffer.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
            createShader("gbuffer.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)
        };
        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(Vertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        std::array<VkVertexInputAttributeDescription, 3> attributes{};
        attributes[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)};
        attributes[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};
        attributes[2] = {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, albedo)};
        VkPipelineVertexInputStateCreateInfo vertexInput{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &binding;
        vertexInput.vertexAttributeDescriptionCount =
            static_cast<std::uint32_t>(attributes.size());
        vertexInput.pVertexAttributeDescriptions = attributes.data();
        VkPipelineInputAssemblyStateCreateInfo assembly{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo raster{
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisample{
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo depth{
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth.depthTestEnable = VK_TRUE;
        depth.depthWriteEnable = VK_TRUE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS;
        VkPipelineColorBlendAttachmentState colorAttachment{};
        colorAttachment.blendEnable = VK_FALSE;
        colorAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blend{
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 1;
        blend.pAttachments = &colorAttachment;
        std::array<VkDynamicState, 2> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
        dynamic.pDynamicStates = dynamicStates.data();
        VkGraphicsPipelineCreateInfo pipelineInfo{
            VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = stages;
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &assembly;
        pipelineInfo.pViewportState = &viewport;
        pipelineInfo.pRasterizationState = &raster;
        pipelineInfo.pMultisampleState = &multisample;
        pipelineInfo.pDepthStencilState = &depth;
        pipelineInfo.pColorBlendState = &blend;
        pipelineInfo.pDynamicState = &dynamic;
        pipelineInfo.layout = framePipelineLayout;
        pipelineInfo.renderPass = gbufferRenderPass;
        pipelineInfo.subpass = 0;
        gbufferPipeline = createGraphicsPipelineTracked(VK_NULL_HANDLE, pipelineInfo);
    }

    {
        const std::array<VkPipelineShaderStageCreateInfo, 2> stages = {
            createShader("fullscreen.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
            createShader("composite.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)
        };
        VkPipelineVertexInputStateCreateInfo vertexInput{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo assembly{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo raster{
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisample{
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo depth{
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkPipelineColorBlendAttachmentState colorAttachment{};
        colorAttachment.blendEnable = VK_FALSE;
        colorAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blend{
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 1;
        blend.pAttachments = &colorAttachment;
        std::array<VkDynamicState, 2> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
        dynamic.pDynamicStates = dynamicStates.data();
        VkGraphicsPipelineCreateInfo pipelineInfo{
            VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipelineInfo.stageCount = static_cast<std::uint32_t>(stages.size());
        pipelineInfo.pStages = stages.data();
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &assembly;
        pipelineInfo.pViewportState = &viewport;
        pipelineInfo.pRasterizationState = &raster;
        pipelineInfo.pMultisampleState = &multisample;
        pipelineInfo.pDepthStencilState = &depth;
        pipelineInfo.pColorBlendState = &blend;
        pipelineInfo.pDynamicState = &dynamic;
        pipelineInfo.layout = framePipelineLayout;
        pipelineInfo.renderPass = compositeRenderPass;
        pipelineInfo.subpass = 0;
        compositePipeline =
            createGraphicsPipelineTracked(VK_NULL_HANDLE, pipelineInfo);
    }

    {
        const std::vector<std::uint8_t> bytes = readSpirv("volume.comp.spv");
        VkShaderModuleCreateInfo moduleInfo{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = bytes.size();
        moduleInfo.pCode = reinterpret_cast<const std::uint32_t*>(bytes.data());
        VkShaderModule module = createShaderModuleTracked(moduleInfo);
        transientShaderModules.push_back(module);
        VkPipelineShaderStageCreateInfo stage{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = module;
        stage.pName = "main";
        VkComputePipelineCreateInfo pipelineInfo{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = stage;
        pipelineInfo.layout = framePipelineLayout;
        volumePipeline =
            createComputePipelineTracked(VK_NULL_HANDLE, pipelineInfo);
    }

    for (VkShaderModule& module : transientShaderModules) {
        destroyShaderModule(module);
    }
    transientShaderModules.clear();
}

void Renderer::Impl::injectAllocationFailure() {
    if (options.injection != Injection::Allocation) {
        return;
    }
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = 4096;
    allocation.memoryTypeIndex = findMemoryType(
        std::numeric_limits<std::uint32_t>::max(),
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    allocationState.failAtCall = static_cast<std::int64_t>(allocationState.calls + 1);
    VkDeviceMemory memory = VK_NULL_HANDLE;
    const VkResult result = vkAllocateMemory(device, &allocation, &allocator, &memory);
    allocationState.failAtCall = -1;
    if (result == VK_SUCCESS) {
        vkFreeMemory(device, memory, &allocator);
        fail(ErrorCode::Allocation,
             "injected Vulkan memory allocation failure was not honored");
    }
    if (result == VK_ERROR_OUT_OF_HOST_MEMORY ||
        result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
        fail(ErrorCode::Allocation,
             "injected Vulkan memory allocation failure (VkResult " +
             std::to_string(result) + ")");
    }
    check(result, "injected vkAllocateMemory");
}

void Renderer::Impl::renderShadowMaps() {
    check(vkResetCommandBuffer(shadowCommandBuffer, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(shadowCommandBuffer, &begin), "vkBeginCommandBuffer");

    VkViewport viewport{0.0f, 0.0f,
        static_cast<float>(scene.shadows.resolution),
        static_cast<float>(scene.shadows.resolution), 0.0f, 1.0f};
    VkRect2D scissor{{0, 0},
        {static_cast<std::uint32_t>(scene.shadows.resolution),
         static_cast<std::uint32_t>(scene.shadows.resolution)}};
    VkClearValue clearDepth{};
    clearDepth.depthStencil = {1.0f, 0};

    {
        VkRenderPassBeginInfo passInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        passInfo.renderPass = directionalShadowRenderPass;
        passInfo.framebuffer = directionalShadowFramebuffer;
        passInfo.renderArea = scissor;
        passInfo.clearValueCount = 1;
        passInfo.pClearValues = &clearDepth;
        vkCmdBeginRenderPass(shadowCommandBuffer, &passInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(shadowCommandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(shadowCommandBuffer, 0, 1, &scissor);
        vkCmdBindPipeline(shadowCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          directionalShadowPipeline);
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(shadowCommandBuffer, 0, 1, &vertexBuffer.buffer, &offset);
        vkCmdBindIndexBuffer(shadowCommandBuffer, indexBuffer.buffer, 0,
                             VK_INDEX_TYPE_UINT32);
        vkCmdPushConstants(shadowCommandBuffer, shadowPipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Mat4),
                           &sceneUbo.dirLightViewProj);
        vkCmdDrawIndexed(shadowCommandBuffer, indexCount, 1, 0, 0, 0);
        vkCmdEndRenderPass(shadowCommandBuffer);
    }

    VkImageMemoryBarrier pointToAttachment{
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    pointToAttachment.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    pointToAttachment.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    pointToAttachment.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pointToAttachment.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pointToAttachment.image = pointShadow.image;
    pointToAttachment.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    pointToAttachment.subresourceRange.levelCount = 1;
    pointToAttachment.subresourceRange.layerCount = 6;
    pointToAttachment.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(shadowCommandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                             VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &pointToAttachment);

    for (std::uint32_t face = 0; face < 6; ++face) {
        VkRenderPassBeginInfo passInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        passInfo.renderPass = pointShadowRenderPass;
        passInfo.framebuffer = pointShadowFramebuffers[face];
        passInfo.renderArea = scissor;
        passInfo.clearValueCount = 1;
        passInfo.pClearValues = &clearDepth;
        vkCmdBeginRenderPass(shadowCommandBuffer, &passInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(shadowCommandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(shadowCommandBuffer, 0, 1, &scissor);
        vkCmdBindPipeline(shadowCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pointShadowPipeline);
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(shadowCommandBuffer, 0, 1, &vertexBuffer.buffer, &offset);
        vkCmdBindIndexBuffer(shadowCommandBuffer, indexBuffer.buffer, 0,
                             VK_INDEX_TYPE_UINT32);
        vkCmdPushConstants(shadowCommandBuffer, shadowPipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Mat4),
                           &sceneUbo.pointFaceViewProj[face]);
        vkCmdDrawIndexed(shadowCommandBuffer, indexCount, 1, 0, 0, 0);
        vkCmdEndRenderPass(shadowCommandBuffer);
    }

    VkImageMemoryBarrier toRead{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toRead.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    toRead.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    toRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toRead.image = pointShadow.image;
    toRead.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    toRead.subresourceRange.levelCount = 1;
    toRead.subresourceRange.layerCount = 6;
    toRead.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(shadowCommandBuffer,
                         VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toRead);

    VkImageMemoryBarrier directionalToRead{
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    directionalToRead.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    directionalToRead.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    directionalToRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    directionalToRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    directionalToRead.image = directionalShadow.image;
    directionalToRead.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    directionalToRead.subresourceRange.levelCount = 1;
    directionalToRead.subresourceRange.layerCount = 1;
    directionalToRead.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    directionalToRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(shadowCommandBuffer,
                         VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &directionalToRead);

    check(vkEndCommandBuffer(shadowCommandBuffer), "vkEndCommandBuffer");
    check(vkResetFences(device, 1, &shadowFence), "vkResetFences");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &shadowCommandBuffer;
    const double start = nowMilliseconds();
    check(vkQueueSubmit(queue, 1, &submit, shadowFence), "vkQueueSubmit(shadows)");
    check(vkWaitForFences(device, 1, &shadowFence, VK_TRUE, UINT64_MAX),
          "vkWaitForFences(shadows)");
    shadowMilliseconds = nowMilliseconds() - start;
}

void Renderer::Impl::writeUbo(std::size_t slot, bool reference) {
    SceneUbo ubo{};
    const Vec3 eye = scene.camera.position;
    const Vec3 target = scene.camera.target;
    const Vec3 up = normalize(scene.camera.up);
    const Vec3 forward = normalize(target - eye);
    const Vec3 right = normalize(cross(forward, up));
    const Vec3 cameraUp = cross(right, forward);

    const float aspect = static_cast<float>(scene.output.width) /
                         static_cast<float>(scene.output.height);
    const Mat4 view = lookAt(eye, target, cameraUp);
    const Mat4 projection = perspectiveVulkan(
        scene.camera.fovYDegrees, aspect, scene.camera.nearMeters,
        scene.camera.farMeters, true);
    ubo.viewProj = projection * view;
    ubo.invViewProj = inverse(ubo.viewProj);
    ubo.cameraPos = {eye.x, eye.y, eye.z, 1.0f};
    ubo.cameraParams = {scene.camera.nearMeters, scene.camera.farMeters, aspect, 0.0f};

    const Vec3 sceneCenter{0.0f, 2.0f, 0.0f};
    const Vec3 lightDirection = normalize(scene.directionalLight.direction);
    const Vec3 lightEye = sceneCenter - lightDirection * 20.0f;
    const Vec3 lightUp =
        std::abs(lightDirection.y) > 0.95f ? Vec3{0.0f, 0.0f, 1.0f}
                                          : Vec3{0.0f, 1.0f, 0.0f};
    const Mat4 lightView = lookAt(lightEye, sceneCenter, lightUp);
    const Mat4 lightProjection = orthoVulkan(-9.0f, 9.0f, -9.0f, 9.0f, 0.1f, 50.0f);
    ubo.dirLightViewProj = lightProjection * lightView;
    ubo.dirLightDirection = {
        lightDirection.x, lightDirection.y, lightDirection.z,
        scene.directionalLight.intensity
    };
    ubo.dirLightColor = {
        scene.directionalLight.color.x, scene.directionalLight.color.y,
        scene.directionalLight.color.z, 0.0f
    };
    ubo.pointLightPosition = {
        scene.pointLight.position.x, scene.pointLight.position.y,
        scene.pointLight.position.z, scene.pointLight.rangeMeters
    };
    ubo.pointLightColor = {
        scene.pointLight.color.x, scene.pointLight.color.y,
        scene.pointLight.color.z, scene.pointLight.intensity
    };
    ubo.mediumCoefficients = {
        scene.volume.absorption, scene.volume.scattering,
        scene.volume.density, scene.volume.anisotropy
    };
    ubo.volumeBoundsMin = {
        scene.volume.boundsMin.x, scene.volume.boundsMin.y,
        scene.volume.boundsMin.z, 0.0f
    };
    ubo.volumeBoundsMax = {
        scene.volume.boundsMax.x, scene.volume.boundsMax.y,
        scene.volume.boundsMax.z, 0.0f
    };
    const float stepMeters = reference ? kReferenceStepMeters : scene.volume.stepMeters;
    const float maxSteps = reference ? static_cast<float>(kReferenceMaxSteps)
                                     : static_cast<float>(scene.volume.maxSteps);
    ubo.noiseParams = {
        scene.volume.noiseFrequencyPerMeter,
        static_cast<float>(scene.noiseSeed),
        static_cast<float>(scene.noiseDimension),
        stepMeters
    };
    ubo.shadowParams = {
        static_cast<float>(scene.shadows.resolution),
        static_cast<float>(scene.shadows.resolution),
        scene.shadows.directionalBias,
        scene.shadows.pointBias
    };
    ubo.pointShadowParams = {
        std::max(0.01f, scene.pointLight.rangeMeters * 0.001f),
        scene.pointLight.rangeMeters,
        static_cast<float>(scene.shadows.pcfKernel),
        0.0f
    };
    ubo.toneParams = {scene.output.exposure, scene.output.gain, 0.0f, 0.0f};
    ubo.backgroundColor = {
        scene.output.background.x, scene.output.background.y,
        scene.output.background.z, 1.0f
    };
    ubo.renderParams = {maxSteps, reference ? 1.0f : 0.0f, 0.0f, 0.0f};

    const Vec3 faceForward[6] = {
        {1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}
    };
    const Vec3 faceRight[6] = {
        {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f, 1.0f},
        {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}
    };
    const Vec3 faceUp[6] = {
        {0.0f, -1.0f, 0.0f}, {0.0f, -1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f},
        {0.0f, -1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}
    };
    const float pointNear = std::max(0.01f, scene.pointLight.rangeMeters * 0.001f);
    const Mat4 pointProjection =
        perspectiveVulkan(90.0f, 1.0f, pointNear, scene.pointLight.rangeMeters, false);
    for (int face = 0; face < 6; ++face) {
        const Mat4 faceView = viewFromBasis(
            normalize(faceRight[face]), normalize(faceUp[face]),
            normalize(faceForward[face]), scene.pointLight.position);
        ubo.pointFaceViewProj[face] = pointProjection * faceView;
    }

    FrameResources& frame = frames[slot];
    std::memcpy(frame.uniform.mapped, &ubo, sizeof(ubo));
    flushBuffer(frame.uniform);
    sceneUbo = ubo;
}

void Renderer::Impl::renderFrame(std::size_t slot, bool reference) {
    FrameResources& frame = frames[slot];
    writeUbo(slot, reference);
    check(vkResetCommandBuffer(frame.commandBuffer, 0), "vkResetCommandBuffer");
    check(vkResetFences(device, 1, &frame.fence), "vkResetFences");
    VkCommandBuffer command = frame.commandBuffer;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");

    vkCmdFillBuffer(command, frame.sampleCounter.buffer, 0, sizeof(std::uint32_t), 0);
    VkBufferMemoryBarrier counterBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    counterBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    counterBarrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    counterBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    counterBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    counterBarrier.buffer = frame.sampleCounter.buffer;
    counterBarrier.offset = 0;
    counterBarrier.size = sizeof(std::uint32_t);
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         0, nullptr, 1, &counterBarrier, 0, nullptr);

    const std::uint32_t width = static_cast<std::uint32_t>(scene.output.width);
    const std::uint32_t height = static_cast<std::uint32_t>(scene.output.height);
    VkViewport viewport{0.0f, 0.0f, static_cast<float>(width),
                        static_cast<float>(height), 0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, {width, height}};

    {
        VkRenderPassBeginInfo passInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        passInfo.renderPass = gbufferRenderPass;
        passInfo.framebuffer = frame.gbufferFramebuffer;
        passInfo.renderArea = scissor;
        std::array<VkClearValue, 2> clears{};
        clears[0].color.float32[0] = scene.output.background.x;
        clears[0].color.float32[1] = scene.output.background.y;
        clears[0].color.float32[2] = scene.output.background.z;
        clears[0].color.float32[3] = 1.0f;
        clears[1].depthStencil = {1.0f, 0};
        passInfo.clearValueCount = static_cast<std::uint32_t>(clears.size());
        passInfo.pClearValues = clears.data();
        vkCmdBeginRenderPass(command, &passInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(command, 0, 1, &viewport);
        vkCmdSetScissor(command, 0, 1, &scissor);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, gbufferPipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                framePipelineLayout, 0, 1,
                                &frame.descriptorSet, 0, nullptr);
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(command, 0, 1, &vertexBuffer.buffer, &offset);
        vkCmdBindIndexBuffer(command, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(command, indexCount, 1, 0, 0, 0);
        vkCmdEndRenderPass(command);
    }

    {
        std::array<VkImageMemoryBarrier, 2> barriers{};
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image = frame.gColor.image;
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.layerCount = 1;
        barriers[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[1].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        barriers[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[1].image = frame.gDepth.image;
        barriers[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        barriers[1].subresourceRange.levelCount = 1;
        barriers[1].subresourceRange.layerCount = 1;
        barriers[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr,
                             static_cast<std::uint32_t>(barriers.size()),
                             barriers.data());
    }

    {
        VkImageMemoryBarrier toGeneral{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toGeneral.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.image = frame.volume.image;
        toGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toGeneral.subresourceRange.levelCount = 1;
        toGeneral.subresourceRange.layerCount = 1;
        toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &toGeneral);

        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, volumePipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                framePipelineLayout, 0, 1,
                                &frame.descriptorSet, 0, nullptr);
        vkCmdDispatch(command, (width + 7u) / 8u, (height + 7u) / 8u, 1);

        VkImageMemoryBarrier toReadback{
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toReadback.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        toReadback.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        toReadback.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toReadback.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toReadback.image = frame.volume.image;
        toReadback.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toReadback.subresourceRange.levelCount = 1;
        toReadback.subresourceRange.layerCount = 1;
        toReadback.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        toReadback.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &toReadback);
    }

    {
        VkRenderPassBeginInfo passInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        passInfo.renderPass = compositeRenderPass;
        passInfo.framebuffer = frame.compositeFramebuffer;
        passInfo.renderArea = scissor;
        VkClearValue clear{};
        clear.color.float32[0] = 0.0f;
        clear.color.float32[1] = 0.0f;
        clear.color.float32[2] = 0.0f;
        clear.color.float32[3] = 1.0f;
        passInfo.clearValueCount = 1;
        passInfo.pClearValues = &clear;
        vkCmdBeginRenderPass(command, &passInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(command, 0, 1, &viewport);
        vkCmdSetScissor(command, 0, 1, &scissor);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                framePipelineLayout, 0, 1,
                                &frame.descriptorSet, 0, nullptr);
        vkCmdDraw(command, 3, 1, 0, 0);
        vkCmdEndRenderPass(command);
    }

    {
        VkImageMemoryBarrier copyBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        copyBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        copyBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        copyBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copyBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copyBarrier.image = frame.output.image;
        copyBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyBarrier.subresourceRange.levelCount = 1;
        copyBarrier.subresourceRange.layerCount = 1;
        copyBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        copyBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &copyBarrier);
        VkBufferImageCopy copy{};
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.layerCount = 1;
        copy.imageExtent = {width, height, 1};
        vkCmdCopyImageToBuffer(command, frame.output.image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               frame.readback.buffer, 1, &copy);
    }

    check(vkEndCommandBuffer(command), "vkEndCommandBuffer");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    const double start = nowMilliseconds();
    check(vkQueueSubmit(queue, 1, &submit, frame.fence), "vkQueueSubmit(frame)");
    check(vkWaitForFences(device, 1, &frame.fence, VK_TRUE, UINT64_MAX),
          "vkWaitForFences(frame)");
    frame.frameMilliseconds = nowMilliseconds() - start;
    invalidateBuffer(frame.readback);
    std::memcpy(frame.pixels.data(), frame.readback.mapped, frame.pixels.size());
    invalidateBuffer(frame.sampleCounter);
    std::memcpy(&frame.sampleCount, frame.sampleCounter.mapped,
                sizeof(frame.sampleCount));
}

void Renderer::Impl::persistRecovery(std::size_t slot) {
    const FrameResources& frame = frames[slot];
    lastRecoveryPixels = frame.pixels;
    lastRecoveryPixelHash = sha256Hex(frame.pixels);
    lastRecoveryFrame = static_cast<std::uint32_t>(slot);
    const std::filesystem::path rawPath = recoveryDirectory / kRecoveryRawName;
    const std::filesystem::path metaPath = recoveryDirectory / kRecoveryMetaName;
    const std::filesystem::path readyPath = recoveryDirectory / kRecoveryReadyName;
    writeBytesAtomic(rawPath, frame.pixels);

    std::ostringstream metadata;
    metadata << "{\n"
             << "  \"schemaVersion\": 1,\n"
             << "  \"complete\": true,\n"
             << "  \"scene\": \"" << json::escape(scene.sourcePath.string()) << "\",\n"
             << "  \"sceneSha256\": \"" << scene.sourceSha256 << "\",\n"
             << "  \"frameIndex\": " << slot << ",\n"
             << "  \"width\": " << scene.output.width << ",\n"
             << "  \"height\": " << scene.output.height << ",\n"
             << "  \"pixelSha256\": \"" << lastRecoveryPixelHash << "\",\n"
             << "  \"rawFile\": \"" << kRecoveryRawName << "\"\n"
             << "}\n";
    writeTextAtomic(metaPath, metadata.str());
    writeTextAtomic(readyPath,
                    "complete\nsceneSha256=" + scene.sourceSha256 +
                        "\nrawFile=" + std::string(kRecoveryRawName) + "\n");
}

void Renderer::Impl::announceRecovery() {
    const std::filesystem::path readyPath = recoveryDirectory / kRecoveryReadyName;
    const std::filesystem::path rawPath = recoveryDirectory / kRecoveryRawName;
    const std::filesystem::path metaPath = recoveryDirectory / kRecoveryMetaName;
    if (!std::filesystem::exists(readyPath) || !std::filesystem::exists(rawPath)) {
        return;
    }
    std::ifstream metadata(metaPath);
    std::string text((std::istreambuf_iterator<char>(metadata)),
                     std::istreambuf_iterator<char>());
    try {
        const json::Value document = json::parse(text);
        const json::Value* complete = document.find("complete");
        const json::Value* width = document.find("width");
        const json::Value* height = document.find("height");
        const json::Value* pixelHash = document.find("pixelSha256");
        if (complete == nullptr || !complete->isBool() || !complete->asBool() ||
            width == nullptr || !width->isNumber() ||
            height == nullptr || !height->isNumber() ||
            pixelHash == nullptr || !pixelHash->isString()) {
            std::cerr << "recovery: ready marker exists but metadata is incomplete; "
                         "the frame was not accepted\n";
            return;
        }
        const auto expectedSize =
            static_cast<std::uintmax_t>(width->asNumber()) *
            static_cast<std::uintmax_t>(height->asNumber()) * 4u;
        if (std::filesystem::file_size(rawPath) != expectedSize) {
            std::cerr << "recovery: ready marker exists but raw frame size is "
                         "invalid; the frame was not accepted\n";
            return;
        }
        const std::vector<std::uint8_t> raw = readFileBytes(rawPath);
        if (sha256Hex(raw) != pixelHash->asString()) {
            std::cerr << "recovery: ready marker exists but pixel hash is "
                         "invalid; the frame was not accepted\n";
            return;
        }
    } catch (const std::exception&) {
        std::cerr << "recovery: ready marker exists but recovery metadata cannot "
                     "be read; the frame was not accepted\n";
        return;
    }
    std::cerr << "recovery: complete frame already exists at "
              << rawPath.string();
    if (!text.empty()) {
        std::cerr << "\nrecovery-metadata: " << text;
        if (text.back() != '\n') {
            std::cerr << '\n';
        }
    } else {
        std::cerr << '\n';
    }
}

void Renderer::Impl::clearRecovery() {
    std::error_code ec;
    std::filesystem::remove(recoveryDirectory / kRecoveryReadyName, ec);
    ec.clear();
    std::filesystem::remove(recoveryDirectory / kRecoveryMetaName, ec);
    ec.clear();
    std::filesystem::remove(recoveryDirectory / kRecoveryRawName, ec);
}

double luminanceFromPixel(const std::uint8_t* pixel) {
    auto linear = [](std::uint8_t value) {
        const double c = static_cast<double>(value) / 255.0;
        return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * linear(pixel[0]) + 0.7152 * linear(pixel[1]) +
           0.0722 * linear(pixel[2]);
}

void Renderer::Impl::writeStats(const std::vector<std::uint8_t>& pngA,
                                const std::vector<std::uint8_t>& pngB,
                                const std::vector<std::uint8_t>& referencePng,
                                double luminanceError) {
    const std::string pngSha = sha256Hex(pngB);
    const std::string frameASha = sha256Hex(pngA);
    const std::string referenceSha =
        referencePng.empty() ? std::string() : sha256Hex(referencePng);
    const double pixels = static_cast<double>(scene.output.width) * scene.output.height;
    std::ostringstream stats;
    stats << std::fixed << std::setprecision(6);
    stats << "{\n"
          << "  \"schemaVersion\": 1,\n"
          << "  \"status\": \"ok\",\n"
          << "  \"scene\": {\n"
          << "    \"path\": \"" << json::escape(scene.sourcePath.string()) << "\",\n"
          << "    \"name\": \"" << json::escape(scene.name) << "\",\n"
          << "    \"schemaVersion\": " << scene.schemaVersion << ",\n"
          << "    \"sha256\": \"" << scene.sourceSha256 << "\"\n"
          << "  },\n"
          << "  \"device\": {\n"
          << "    \"name\": \"" << json::escape(physicalProperties.deviceName) << "\",\n"
          << "    \"apiVersion\": \"" << apiVersionString(physicalProperties.apiVersion)
          << "\",\n"
          << "    \"driverVersion\": " << physicalProperties.driverVersion << ",\n"
          << "    \"validationLayers\": "
          << (validationEnabled ? "true" : "false") << "\n"
          << "  },\n"
          << "  \"output\": {\n"
          << "    \"png\": \"" << json::escape(outputPngPath.string()) << "\",\n"
          << "    \"width\": " << scene.output.width << ",\n"
          << "    \"height\": " << scene.output.height << ",\n"
          << "    \"pngSha256\": \"" << pngSha << "\",\n"
          << "    \"frameASha256\": \"" << frameASha << "\",\n"
          << "    \"frameBSha256\": \"" << pngSha << "\",\n"
          << "    \"framesIdentical\": "
          << (pngA == pngB ? "true" : "false") << "\n"
          << "  },\n"
          << "  \"reference\": {\n";
    if (options.noReference) {
        stats << "    \"enabled\": false\n";
    } else {
        stats << "    \"enabled\": true,\n"
              << "    \"png\": \"" << json::escape(referencePngPath.string()) << "\",\n"
              << "    \"pngSha256\": \"" << referenceSha << "\",\n"
              << "    \"source\": \"same renderer; fixed seed from scene noise; "
                 "reference stepMeters=0.030; reference maxSteps=512; same camera, "
                 "lights, medium, PCF and exposure; Rec.709 linear luminance\",\n"
              << "    \"noiseSeed\": " << scene.noiseSeed << ",\n"
              << "    \"stepMeters\": " << kReferenceStepMeters << ",\n"
              << "    \"maxSteps\": " << kReferenceMaxSteps << ",\n"
              << "    \"meanAbsoluteLuminanceError\": " << luminanceError << ",\n"
              << "    \"threshold\": 0.020000,\n"
              << "    \"passed\": " << (luminanceError <= 0.02 ? "true" : "false")
              << "\n";
    }
    stats << "  },\n"
          << "  \"statistics\": {\n"
          << "    \"framesRendered\": "
          << (frames.size() + (options.noReference ? 0u : 1u)) << ",\n"
          << "    \"volumeStepMeters\": " << scene.volume.stepMeters << ",\n"
          << "    \"volumeMaxSteps\": " << scene.volume.maxSteps << ",\n"
          << "    \"volumeSamplesFrameA\": " << mainSampleCounts[0] << ",\n"
          << "    \"volumeSamplesFrameB\": " << mainSampleCounts[1] << ",\n"
          << "    \"volumeSamplesPerPixelFrameA\": "
          << (pixels > 0.0 ? mainSampleCounts[0] / pixels : 0.0) << ",\n"
          << "    \"volumeSamplesReference\": " << referenceSampleCount << ",\n"
          << "    \"shadowMapMilliseconds\": " << shadowMilliseconds << ",\n"
          << "    \"frameAMilliseconds\": " << mainFrameMilliseconds[0] << ",\n"
          << "    \"frameBMilliseconds\": " << mainFrameMilliseconds[1] << ",\n"
          << "    \"referenceMilliseconds\": " << referenceFrameMilliseconds << "\n"
          << "  },\n"
          << "  \"determinism\": {\n"
          << "    \"sameInputSamePng\": " << (pngA == pngB ? "true" : "false")
          << ",\n"
          << "    \"framePixelSha256A\": \"" << mainPixelHashes[0] << "\",\n"
          << "    \"framePixelSha256B\": \"" << mainPixelHashes[1] << "\"\n"
          << "  }\n"
          << "}\n";
    writeTextAtomic(statsPath, stats.str());
}

void Renderer::Impl::run() {
    if (!initialized) {
        initialize();
    }
    writeUbo(0, false);
    renderShadowMaps();

    std::vector<std::vector<std::uint8_t>> pngFrames;
    pngFrames.reserve(frames.size());
    for (std::size_t slot = 0; slot < frames.size(); ++slot) {
        renderFrame(slot, false);
        persistRecovery(slot);
        pngFrames.push_back(encodePngRgba8(
            scene.output.width, scene.output.height, frames[slot].pixels));
    }
    for (std::size_t slot = 2; slot < frames.size(); ++slot) {
        if (frames[slot].pixels != frames[1].pixels) {
            fail(ErrorCode::NonDeterministic,
                 "frame " + std::to_string(slot) +
                     " differs from frame 1 for identical input and descriptors");
        }
    }
    if (pngFrames.size() < 2 || pngFrames[0] != pngFrames[1]) {
        fail(ErrorCode::NonDeterministic,
             "the first two PNG frames are not byte-identical");
    }
    mainSampleCounts[0] = frames[0].sampleCount;
    mainSampleCounts[1] = frames[1].sampleCount;
    mainFrameMilliseconds[0] = frames[0].frameMilliseconds;
    mainFrameMilliseconds[1] = frames[1].frameMilliseconds;
    mainPixelHashes[0] = sha256Hex(frames[0].pixels);
    mainPixelHashes[1] = sha256Hex(frames[1].pixels);

    if (options.injection == Injection::DeviceLost) {
        deviceLost = true;
        fail(ErrorCode::DeviceLost,
             "injected VK_ERROR_DEVICE_LOST after persisting the last complete frame" +
                 recoverySuffix());
    }

    std::vector<std::uint8_t> referencePng;
    double luminanceError = 0.0;
    if (!options.noReference) {
        renderFrame(0, true);
        referenceSampleCount = frames[0].sampleCount;
        referenceFrameMilliseconds = frames[0].frameMilliseconds;
        referencePng = encodePngRgba8(
            scene.output.width, scene.output.height, frames[0].pixels);
        double total = 0.0;
        const std::size_t pixelCount =
            static_cast<std::size_t>(scene.output.width) * scene.output.height;
        for (std::size_t i = 0; i < pixelCount; ++i) {
            const std::uint8_t* mainPixel = frames[1].pixels.data() + i * 4;
            const std::uint8_t* referencePixel = frames[0].pixels.data() + i * 4;
            total += std::abs(luminanceFromPixel(mainPixel) -
                              luminanceFromPixel(referencePixel));
        }
        luminanceError = total / static_cast<double>(pixelCount);
        writeBytesAtomic(referencePngPath, referencePng);
    }

    writeBytesAtomic(outputPngPath, pngFrames[1]);
    writeStats(pngFrames[0], pngFrames[1], referencePng, luminanceError);

    std::cout << "output: " << outputPngPath.string() << "\n"
              << "stats: " << statsPath.string() << "\n"
              << "png_sha256=" << sha256Hex(pngFrames[1]) << "\n"
              << "frames_identical=true\n"
              << "volume_samples_frame_a=" << mainSampleCounts[0] << "\n"
              << "volume_samples_frame_b=" << mainSampleCounts[1] << "\n"
              << "shadow_map_ms=" << std::fixed << std::setprecision(3)
              << shadowMilliseconds << "\n";
    if (!options.noReference) {
        std::cout << "reference_png: " << referencePngPath.string() << "\n"
                  << "mean_abs_luminance_error=" << std::setprecision(6)
                  << luminanceError << " threshold=0.020000\n";
    }
    if (options.clearRecovery) {
        clearRecovery();
    }
    if (!options.noReference && luminanceError > 0.02) {
        fail(ErrorCode::Reference,
             "mean absolute luminance error " + std::to_string(luminanceError) +
                 " exceeds 0.02");
    }
}

void Renderer::Impl::cleanup() {
    if (cleanupDone) {
        return;
    }
    cleanupDone = true;
    if (device != VK_NULL_HANDLE) {
        if (vkDeviceWaitIdle(device) == VK_ERROR_DEVICE_LOST) {
            deviceLost = true;
        }
    }
    for (VkShaderModule& module : transientShaderModules) {
        destroyShaderModule(module);
    }
    transientShaderModules.clear();

    for (FrameResources& frame : frames) {
        destroyFramebuffer(frame.gbufferFramebuffer);
        destroyFramebuffer(frame.compositeFramebuffer);
        destroyFence(frame.fence);
        destroyBuffer(frame.uniform);
        destroyBuffer(frame.sampleCounter);
        destroyBuffer(frame.readback);
        destroyImage(frame.gColor);
        destroyImage(frame.gDepth);
        destroyImage(frame.volume);
        destroyImage(frame.output);
    }
    frames.clear();
    if (descriptorPool != VK_NULL_HANDLE) {
        destroyDescriptorPool(descriptorPool);
    }
    for (VkFramebuffer& framebuffer : pointShadowFramebuffers) {
        destroyFramebuffer(framebuffer);
    }
    destroyFramebuffer(directionalShadowFramebuffer);
    for (VkImageView& view : pointShadowViews) {
        destroyView(view);
    }
    destroyImage(directionalShadow);
    destroyImage(pointShadow);
    destroyImage(noiseImage);
    destroyBuffer(vertexBuffer);
    destroyBuffer(indexBuffer);
    destroyPipeline(directionalShadowPipeline);
    destroyPipeline(pointShadowPipeline);
    destroyPipeline(gbufferPipeline);
    destroyPipeline(compositePipeline);
    destroyPipeline(volumePipeline);
    destroyPipelineLayout(framePipelineLayout);
    destroyPipelineLayout(shadowPipelineLayout);
    destroyDescriptorSetLayout(descriptorSetLayout);
    destroyRenderPass(directionalShadowRenderPass);
    destroyRenderPass(pointShadowRenderPass);
    destroyRenderPass(gbufferRenderPass);
    destroyRenderPass(compositeRenderPass);
    destroySampler(noiseSampler);
    destroySampler(depthSampler);
    destroySampler(colorSampler);
    destroyFence(shadowFence);
    destroyCommandPool(commandPool);
    if (device != VK_NULL_HANDLE) {
        vkDestroyDevice(device, &allocator);
        device = VK_NULL_HANDLE;
    }
    if (debugMessenger != VK_NULL_HANDLE && destroyDebugMessenger != nullptr) {
        destroyDebugMessenger(instance, debugMessenger, &allocator);
        debugMessenger = VK_NULL_HANDLE;
    }
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, &allocator);
        instance = VK_NULL_HANDLE;
    }
}

Renderer::Renderer(RendererOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

Renderer::~Renderer() = default;

void Renderer::run() {
    impl_->run();
}

void Renderer::cleanup() {
    impl_->cleanup();
}

int Renderer::liveResourceCount() const noexcept {
    return impl_->liveResources;
}

long long Renderer::liveAllocatorCount() const noexcept {
    return static_cast<long long>(impl_->allocationState.live);
}

bool Renderer::deviceWasLost() const noexcept {
    return impl_->deviceLost;
}

} // namespace vlr
