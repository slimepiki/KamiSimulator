#include "../YBSimulator.h"

#define checkCudaErrors(ans)                  \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

namespace YarnBall {
void Sim::rebuildCUDAGraph() {
    // Graph is still good
    if (meta.numItr == lastItr) return;
    checkCudaErrors(cudaGetLastError());

    cudaStreamSynchronize(stream);
    // Build graph with detection
    {
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        recomputeStepLimit();

        for (int i = 0; i < meta.numItr; i++) iterateCosserat();

        endIterate();

        cudaGraph_t graph;
        cudaStreamEndCapture(stream, &graph);

        if (stepGraph) cudaGraphExecDestroy(stepGraph);
        cudaGraphInstantiate(&stepGraph, graph, NULL, NULL, 0);
        cudaGraphDestroy(graph);
    }

    checkCudaErrors(cudaGetLastError());

    lastItr = meta.numItr;
}

float Sim::advance(float h) {
    if (h <= 0) return 0;

    int steps = max(1, (int)ceil(h / maxH));
    meta.lastH = meta.h;
    meta.h = h / steps;

    rebuildCUDAGraph();
    uploadMeta();
    // printf("h:%f, dt: %f, steps: %d\n", h, meta.h, steps);

    for (int s = 0; s < steps; s++, stepCounter++) {
        initIterate();

        // When the number of segments is too large, this function will crash due to the thrust library.
        if (meta.enableHairHairColl && meta.detectionPeriod > 0 && stepCounter % meta.detectionPeriod == 0) detectCollisions();

        // printf("collend:\t %d/%d\n", s, steps - 1);
        cudaGraphLaunch(stepGraph, stream);
    }

    meta.time += h;
    checkErrors();
    return h;
}

void Sim::step(float h) { advance(maxH); }
}  // namespace YarnBall