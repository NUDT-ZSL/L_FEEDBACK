#include "core/command_recorder.hpp"

#include "core/error.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

namespace uid166 {

namespace {

using Clock = std::chrono::steady_clock;

uint64_t microsBetween(Clock::time_point a, Clock::time_point b) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(b - a).count());
}

struct WorkerContext {
    const std::vector<uint32_t>* visible = nullptr;
    const std::vector<SceneObject>* objects = nullptr;
    std::vector<RenderCommand>* output = nullptr;
    std::vector<uint8_t>* written = nullptr;
    std::atomic<size_t>* nextIndex = nullptr;
    std::atomic<bool>* budgetHit = nullptr;
    Clock::time_point deadline{};
    bool hasDeadline = false;
    uint32_t workerId = 0;
    WorkerLoad* load = nullptr;
};

void runWorker(WorkerContext& context) {
    const Clock::time_point start = Clock::now();
    constexpr size_t chunkSize = 16;
    while (true) {
        if (context.hasDeadline && Clock::now() > context.deadline) {
            context.budgetHit->store(true, std::memory_order_relaxed);
            break;
        }
        const size_t begin = context.nextIndex->fetch_add(chunkSize, std::memory_order_relaxed);
        if (begin >= context.visible->size()) break;
        const size_t end = std::min(begin + chunkSize, context.visible->size());
        for (size_t i = begin; i < end; ++i) {
            const uint32_t objectId = (*context.visible)[i];
            RenderCommand& command = (*context.output)[i];
            command.objectId = objectId;
            command.material = objectId < context.objects->size()
                ? (*context.objects)[objectId].material : 0u;
            command.world = objectId < context.objects->size()
                ? objectWorldMatrix((*context.objects)[objectId]) : Mat4::identity();
            (*context.written)[i] = 1;
        }
        context.load->commands += static_cast<uint32_t>(end - begin);
    }
    context.load->elapsedMicros = microsBetween(start, Clock::now());
}

#ifdef _WIN32
void CALLBACK threadPoolCallback(PTP_CALLBACK_INSTANCE, PVOID parameter, PTP_WORK) {
    runWorker(*static_cast<WorkerContext*>(parameter));
}
#endif

void buildBatches(RecordingResult& result) {
    std::array<std::vector<RenderCommand>, 65> byMaterial;
    for (const RenderCommand& command : result.commands) {
        const size_t material = std::min<size_t>(command.material, byMaterial.size() - 1);
        byMaterial[material].push_back(command);
    }
    for (size_t material = 0; material < byMaterial.size(); ++material) {
        if (byMaterial[material].empty()) continue;
        RenderBatch batch;
        batch.material = static_cast<uint32_t>(material);
        batch.commands = std::move(byMaterial[material]);
        result.batches.push_back(std::move(batch));
    }
    result.mergedDrawCalls = static_cast<uint32_t>(result.batches.size());
}

} // namespace

RecordingResult CommandRecorder::record(const std::vector<uint32_t>& visibleObjectIds,
                                        const std::vector<SceneObject>& objects,
                                        const RecordingConfig& config) const {
    RecordingResult result;
    result.commands.resize(visibleObjectIds.size());
    std::vector<uint8_t> written(visibleObjectIds.size(), 0);
    if (visibleObjectIds.empty()) {
        result.elapsedMicros = 0;
        return result;
    }

    const uint32_t hardware = std::max(1u, std::thread::hardware_concurrency());
    const uint32_t requested = config.workerCount == 0 ? hardware : config.workerCount;
    const uint32_t workerCount = std::max(1u, std::min({
        requested,
        static_cast<uint32_t>((visibleObjectIds.size() + 63) / 64),
        hardware,
        16u,
    }));
    result.workers.resize(workerCount);
    std::vector<WorkerContext> contexts(workerCount);
    std::atomic<size_t> nextIndex{0};
    std::atomic<bool> budgetHit{false};
    const Clock::time_point start = Clock::now();
    const Clock::time_point deadline = start + std::chrono::microseconds(config.budgetMicros);

    for (uint32_t i = 0; i < workerCount; ++i) {
        contexts[i].visible = &visibleObjectIds;
        contexts[i].objects = &objects;
        contexts[i].output = &result.commands;
        contexts[i].written = &written;
        contexts[i].nextIndex = &nextIndex;
        contexts[i].budgetHit = &budgetHit;
        contexts[i].deadline = deadline;
        contexts[i].hasDeadline = config.budgetMicros != 0;
        contexts[i].workerId = i;
        contexts[i].load = &result.workers[i];
        result.workers[i].workerId = i;
    }

#ifdef _WIN32
    std::vector<PTP_WORK> works(workerCount, nullptr);
    bool poolFailure = false;
    for (uint32_t i = 0; i < workerCount; ++i) {
        works[i] = CreateThreadpoolWork(threadPoolCallback, &contexts[i], nullptr);
        if (!works[i]) {
            poolFailure = true;
            break;
        }
        SubmitThreadpoolWork(works[i]);
    }
    for (PTP_WORK work : works) {
        if (work) WaitForThreadpoolWorkCallbacks(work, FALSE);
    }
    for (PTP_WORK work : works) {
        if (work) CloseThreadpoolWork(work);
    }
    if (poolFailure) {
        throw AppError(ErrorCode::threadPool, "CreateThreadpoolWork failed");
    }
#else
    std::vector<std::thread> threads;
    threads.reserve(workerCount);
    try {
        for (WorkerContext& context : contexts) {
            threads.emplace_back([&context] { runWorker(context); });
        }
    } catch (const std::system_error& error) {
        throw AppError(ErrorCode::threadPool,
                       std::string("failed to start recording thread: ") + error.what());
    }
    for (std::thread& thread : threads) thread.join();
#endif
    result.parallelStageMicros = microsBetween(start, Clock::now());
    result.budgetHit = budgetHit.load(std::memory_order_relaxed);

    const Clock::time_point fallbackStart = Clock::now();
    for (size_t i = 0; i < visibleObjectIds.size(); ++i) {
        if (written[i]) continue;
        const uint32_t objectId = visibleObjectIds[i];
        result.commands[i].objectId = objectId;
        result.commands[i].material = objectId < objects.size() ? objects[objectId].material : 0u;
        result.commands[i].world = objectId < objects.size()
            ? objectWorldMatrix(objects[objectId]) : Mat4::identity();
        ++result.fallbackSerialCommands;
    }
    result.elapsedMicros = microsBetween(start, Clock::now());
    (void)fallbackStart;
    result.budgetExceeded = config.budgetMicros != 0 && result.elapsedMicros > config.budgetMicros;
    buildBatches(result);
    return result;
}

} // namespace uid166

