#include <vector>

#include "solver.h"
#include "SFieldCPU.h"
#include "HairGen.h"
#include "HairLoader.h"

#include "../../Hair.h"
#include "../../Body.h"
#include "../../PNMesh.h"
#include "../../KamiSimulator.h"
#include "../../utilities/LinearUtil.h"

#include "../../../extern/glm/glm/common.hpp"
#include "../../../extern/cuda_helpers/helper_cuda.h"

using Kami::LinearUtil::EigenVecX3f2DArray;

AMSSolver::AMSSolver(const HairInfo& hair, const HeadInfo& head) {
    SelectCudaDevice();
    InitParams();
    SimParams.HairDim = EulerDim;
    SimParams.SDFDim = head.SDFDim;
}

AMSSolver::AMSSolver(weak_ptr<Hair> _hair, weak_ptr<Body> _body) {
    SelectCudaDevice();
    InitParams();
    SimParams.HairDim = EulerDim;
    auto hSDFsize = _body.lock()->GetCurrentHeadOrBoxPNMeshPtr()->GetPNMeshParam().SDFDim;
    SimParams.SDFDim = make_int3(hSDFsize.x, hSDFsize.y, hSDFsize.z);
}

AMSSolver::~AMSSolver() { FreeBuffers(); }

// void AMSSolver::Clear() {}

void AMSSolver::InitBuffers(vector<shared_ptr<SimpleObject>>& objs, shared_ptr<HairGenerator> hairGen) {
    // Unwrap generator
    vector<float>& restLengths = hairGen->GetRestLengths();
    vector<Particle>& particles = hairGen->GetParticles();
    vector<InterParticle>& interParticles = hairGen->InterParticles();
    vector<int>& rootIdx = hairGen->GetRootIdx();
    Meshes = objs;

    // System Dimensions
    int numGridCellsU = (SimParams.HairDim.x + 1) * SimParams.HairDim.y * SimParams.HairDim.z;
    int numGridCellsV = SimParams.HairDim.x * (SimParams.HairDim.y + 1) * SimParams.HairDim.z;
    int numGridCellsW = SimParams.HairDim.x * SimParams.HairDim.y * (SimParams.HairDim.z + 1);
    int numGridCells = SimParams.HairDim.x * SimParams.HairDim.y * SimParams.HairDim.z;
    int numSdfCells = SimParams.SDFDim.x * SimParams.SDFDim.y * SimParams.SDFDim.z;
    int numTriangles = objs[0]->GetNumTriangles();
    int numVertices = objs[0]->GetNumVertices();
    int numFrames = objs[0]->GetNumFrames();
    int numParticles = particles.size();
    int numRoots = rootIdx.size();

    SimParams.MaxStrandLength = hairGen->GetMaxStrandLength();
    StrandLength = hairGen->GetStrandLengths();
    SimParams.NumGridCellsU = numGridCellsU;
    SimParams.NumGridCellsV = numGridCellsV;
    SimParams.NumGridCellsW = numGridCellsW;
    SimParams.NumParticles = numParticles;
    SimParams.NumInter = interParticles.size();
    SimParams.NumTriangles = numTriangles;
    SimParams.NumGridCells = numGridCells;
    SimParams.NumSdfCells = numSdfCells;
    SimParams.NumVertices = numVertices;
    SimParams.NumFrames = numFrames;
    SimParams.NumRoots = numRoots;
    SimParams.MeshSequence = numFrames > 1;
    // Allocate GPU Memory

    // Eulerian hair
    HairWeightU = make_shared<CuBuffer<float>>(numGridCellsU);
    HairWeightV = make_shared<CuBuffer<float>>(numGridCellsV);
    HairWeightW = make_shared<CuBuffer<float>>(numGridCellsW);
    HairVelU = make_shared<CuBuffer<float>>(numGridCellsU);
    HairVelV = make_shared<CuBuffer<float>>(numGridCellsV);
    HairVelW = make_shared<CuBuffer<float>>(numGridCellsW);
    HairPicU = make_shared<CuBuffer<float>>(numGridCellsU);
    HairPicV = make_shared<CuBuffer<float>>(numGridCellsV);
    HairPicW = make_shared<CuBuffer<float>>(numGridCellsW);
    HairPressure = make_shared<CuBuffer<float2>>(numGridCells);
    VoxelType = make_shared<CuBuffer<CellType>>(numGridCells);
    HairDiv = make_shared<CuBuffer<float>>(numGridCells);

    DBuffers.HairPressure = HairPressure->DevPtr();
    DBuffers.HairWeightU = HairWeightU->DevPtr();
    DBuffers.HairWeightV = HairWeightV->DevPtr();
    DBuffers.HairWeightW = HairWeightW->DevPtr();
    DBuffers.VoxelType = VoxelType->DevPtr();
    DBuffers.HairVelU = HairVelU->DevPtr();
    DBuffers.HairVelV = HairVelV->DevPtr();
    DBuffers.HairVelW = HairVelW->DevPtr();
    DBuffers.HairPicU = HairPicU->DevPtr();
    DBuffers.HairPicV = HairPicV->DevPtr();
    DBuffers.HairPicW = HairPicW->DevPtr();
    DBuffers.HairDiv = HairDiv->DevPtr();

    // Lagrangian Hair
    InterParticles = make_shared<CuBuffer<InterParticle>>(interParticles.size(), true);
    Particles = make_shared<CuBuffer<Particle>>(numParticles, true);
    RestLengths = make_shared<CuBuffer<float>>(restLengths.size());
    RootIdx = make_shared<CuBuffer<int>>(numRoots, true);

    DBuffers.RestLenghts = RestLengths->DevPtr();
    DBuffers.Particles = Particles->DevPtr();
    DBuffers.InterParticles = InterParticles->DevPtr();
    DBuffers.RootIdx = RootIdx->DevPtr();

    RestLengths->CopyHostToDevice(restLengths.data());
    Particles->CopyHostToDevice(particles.data());
    InterParticles->CopyHostToDevice(interParticles.data());
    RootIdx->CopyHostToDevice(rootIdx.data());

    Particles->CopyDeviceToHost();
    RootIdx->CopyDeviceToHost();

    // Heptadiagional solver
    StrandA = make_shared<CuBuffer<hcuMat3>>(7 * numParticles);
    StrandL = make_shared<CuBuffer<hcuMat3>>(4 * numParticles);
    StrandU = make_shared<CuBuffer<hcuMat3>>(4 * numParticles);
    StrandV = make_shared<CuBuffer<float3>>(numParticles);
    StrandB = make_shared<CuBuffer<float3>>(numParticles);

    DBuffers.StrandL = StrandL->DevPtr();
    DBuffers.StrandU = StrandU->DevPtr();
    DBuffers.StrandV = StrandV->DevPtr();
    DBuffers.StrandA = StrandA->DevPtr();
    DBuffers.StrandB = StrandB->DevPtr();

    // Head
    HeadVertices = make_shared<CuBuffer<SimpleVertex>>(numVertices, true);
    HeadTriangles = make_shared<CuBuffer<SimpleTriangle>>(numTriangles);
    RigidMotion = make_shared<CuBuffer<hcuMat4>>(3);
    Sdf = make_shared<CuBuffer<float>>(numSdfCells, true);
    NablaSdf = make_shared<CuBuffer<float3>>(numSdfCells, true);
    HeadVelX = make_shared<CuBuffer<float>>(numSdfCells);
    HeadVelY = make_shared<CuBuffer<float>>(numSdfCells);
    HeadVelZ = make_shared<CuBuffer<float>>(numSdfCells);
    HeadVelWeight = make_shared<CuBuffer<float>>(numSdfCells);
    SdfCPU = make_shared<SFieldCPU>(objs[0]);
    RootBary = make_shared<CuBuffer<float3>>(numRoots, true);
    RootTri = make_shared<CuBuffer<int>>(numRoots, true);

    DBuffers.HeadTriangles = HeadTriangles->DevPtr();
    DBuffers.HeadVertices = HeadVertices->DevPtr();
    DBuffers.RigidMotion = RigidMotion->DevPtr();
    DBuffers.Sdf = Sdf->DevPtr();
    DBuffers.NablaSdf = NablaSdf->DevPtr();
    DBuffers.RootBary = RootBary->DevPtr();
    DBuffers.RootTri = RootTri->DevPtr();
    DBuffers.HeadVelX = HeadVelX->DevPtr();
    DBuffers.HeadVelY = HeadVelY->DevPtr();
    DBuffers.HeadVelZ = HeadVelZ->DevPtr();
    DBuffers.HeadVelWeight = HeadVelWeight->DevPtr();

    HeadTriangles->CopyHostToDevice(objs[0]->GetTrianglesRaw().data());
    HeadVertices->CopyHostToDevice(objs[0]->GetVerticesRaw().data());

    vector<hcuMat4> NoTrans = {hcuMat4::Identity(), hcuMat4::Identity(), hcuMat4::Identity()};
    RigidMotion->CopyHostToDevice(NoTrans.data());

    // Animation (mesh sequence)
    if (SimParams.MeshSequence) {
        AnimVertices = make_shared<CuBuffer<MiniVertex>>(numFrames * numVertices);
        DBuffers.AnimVertices = AnimVertices->DevPtr();
        AnimVertices->CopyHostToDevice(objs[0]->GetAnimVerticesRaw().data());
    }

    // Compute block sizes for CUDA
    BlockThreadHair.x = DivUp(numParticles, BlockThreadHair.y);
    BlockThreadMesh.x = DivUp(numTriangles, BlockThreadMesh.y);
    BlockThreadVert.x = DivUp(numVertices, BlockThreadVert.y);
    BlockThreadGridU.x = DivUp(numGridCellsU, BlockThreadGridU.y);
    BlockThreadGridV.x = DivUp(numGridCellsV, BlockThreadGridV.y);
    BlockThreadGridW.x = DivUp(numGridCellsW, BlockThreadGridW.y);
    BlockThreadGrid.x = DivUp(numGridCells, BlockThreadGrid.y);
    BlockThreadSdf.x = DivUp(numSdfCells, BlockThreadSdf.y);
    BlockThreadRoot.x = DivUp(numRoots, BlockThreadRoot.y);
    BlockThreadInter.x = DivUp(interParticles.size(), BlockThreadInter.y);

    // Params to GPU
    copySimParamsToDevice(&SimParams);
}

void AMSSolver::InitParticles() {
    launchKernelInitParticles(BlockThreadHair, DBuffers);

    // Init head velocity
    launchKernelInitHeadVerticesVel(BlockThreadVert, DBuffers);
    launchKernelInitHeadVel(BlockThreadSdf, DBuffers);
    Particles->CopyDeviceToHost();
}

void AMSSolver::ComputeSDFCPU(weak_ptr<Body> _body) {
    // Build on CPU
    int gridSize = SimParams.SDFDim.x * SimParams.SDFDim.y * SimParams.SDFDim.z;

    if (SdfCPU->SetSDFFromPNMesh(_body.lock()->GetCurrentHeadOrBoxPNMeshPtr())) {
#pragma omp parallel for
        for (int i = 0; i < gridSize; i++) {
            Sdf->HostPtr()[i] = SimParams.Parity * SdfCPU->GetSDF()[i];
            NablaSdf->HostPtr()[i] = SimParams.Parity * SdfCPU->GetNabla()[i];
        }

        // Copy to GPU
        Sdf->CopyHostToDevice();
        NablaSdf->CopyHostToDevice();
    }
}

void AMSSolver::ComputeBary() {
    // We need the SDf, so we do these computations on CPU
    Particles->CopyDeviceToHost();
    RootIdx->CopyDeviceToHost();

    // Unwrap
    vector<shared_ptr<SimpleTriangle>>& triangles = Meshes[0]->GetTriangles();
    vector<shared_ptr<SimpleVertex>>& vertices = Meshes[0]->GetVertices();
    Particle* particlesCPU = Particles->HostPtr();
    float3* baryCPU = RootBary->HostPtr();
    int* idxCPU = RootIdx->HostPtr();

    // Fills in parallel-cpu
#pragma omp parallel for
    for (size_t i = 0; i < SimParams.NumRoots; i++) {
        // Get closest triangle to this root
        PointSDF sample = SdfCPU->DistancePointField(particlesCPU[idxCPU[i]].Position);

        // Get barycentric coordinates
        shared_ptr<SimpleTriangle>& triangle = triangles[sample.TriIdx];
        float3 a = vertices[triangle->V[0]]->Pos;
        float3 b = vertices[triangle->V[1]]->Pos;
        float3 c = vertices[triangle->V[2]]->Pos;
        baryCPU[i] = BaryCoordinates(sample.NearestPoint, a, b, c);
    }

    // Copy everything to GPU
    RootBary->CopyHostToDevice();
    RootTri->CopyHostToDevice();
}

void AMSSolver::UpdateSimulation(weak_ptr<Hair> _hair, weak_ptr<Body> _body) {
    // Update parameters on GPU
    // also updates inverse mass and nested dt as it may be
    // that some parameter was changed
    SimParams.Mu = 1.f / SimParams.HairMass;
    SimParams.DtN = SimParams.Dt / (1.f * SimParams.NestedSteps);
    SimParams.InvDt = 1.f / SimParams.Dt;

    hcuMat4 rot = hcuMat4::RotateZ(SimParams.RigidAngle.z) * hcuMat4::RotateY(SimParams.RigidAngle.y) * hcuMat4::RotateX(SimParams.RigidAngle.x);
    hcuMat4 rigid = hcuMat4::Translate(SimParams.RigidPos) * rot;
    hcuMat4 inv = hcuMat4::Transpose(rigid.Inverse());
    vector<hcuMat4> mats = {rigid, inv, rigid.Inverse()};
    RigidMotion->CopyHostToDevice(mats.data());
    SimParams.HairMin = rigid * SimParams.HairMin0;
    SimParams.HairCenter = rigid * SimParams.HairCenter0;
    SimParams.HairAxis[0] = rot * SimParams.HairAxis0[0];
    SimParams.HairAxis[1] = rot * SimParams.HairAxis0[1];
    SimParams.HairAxis[2] = rot * SimParams.HairAxis0[2];
    copySimParamsToDevice(&SimParams);

    // Custom Animation
    if (SimParams.Animate) UpdateAnimation(_body);

    // Update Solid Geometry
    // UpdateMesh();
    // UpdateRigidSDF();
    UpdateRoots();

    // UpdateSDFManual();
    // MapOpenGL();

    UpdateDynamics();

    // Export
    // ExportData();
    // WritebackData(_hair, _body);

    // Elapsed time
    SimParams.SimTime += SimParams.Dt;

    // Pause for debugging
    // SimParams.PauseSim = true;
}

void AMSSolver::UpdateProcessor(weak_ptr<Hair> _hair, weak_ptr<Body> _body) {
    // Update parameters on GPU
    // also updates inverse mass and nested dt as it may be
    // that some parameter was changed
    SimParams.Mu = 1.f / SimParams.HairMass;
    SimParams.DtN = SimParams.Dt / (1.f * SimParams.NestedSteps);
    copySimParamsToDevice(&SimParams);

    // Move roots to head
    FixRootPositions();

    // Export
    // ExportData();
    // WritebackData(hair, body);
}

void AMSSolver::Reset() {
    // Restart particles
    launchKernelResetParticles(BlockThreadHair, DBuffers);

    // Restar export indices
    ExportCounter = 0;
    ExportIdx = 0;
    SimParams.SimTime = 0.f;
}

void AMSSolver::UseSdfOrNot(bool yn) { SimParams.SolidCollision = yn; }

void AMSSolver::WritebackData(weak_ptr<Hair> _hair, weak_ptr<Body> _body) {
    WritebackKamiHair(_hair);
    // Update data on CPU
    if (SimParams.AnimationType == SEQUENCE && SimParams.Animate == true) {
        Meshes[0]->UpdateVerticesFromSeq(HeadVertices->HostPtr());
    }
    Meshes[0]->WriteBackKamiHead(_body);
}

void AMSSolver::WritebackKamiHair(weak_ptr<Hair> _hair) {
    Particles->CopyDeviceToHost();

    auto& hVerts = _hair.lock()->GetCurrentVerticesRef();
    auto& hVels = _hair.lock()->GetCurrentVelocitiesRef();
    auto& numStrands = _hair.lock()->GetStrandVertCountRef();

#pragma omp parallel for
    for (size_t i = 0; i < SimParams.NumRoots; i++) {
#pragma omp parallel for
        for (size_t j = 0; j < SimParams.MaxStrandLength; j++) {
            float damping = 0.5;
            if (j < numStrands[i]) {
                int rootIdx = RootIdx->HostPtr()[i];
                hVerts->SetEntryToArray(Float2GLM(Particles->HostPtr()[rootIdx + j].Position), i, j);
                hVels->SetEntryToArray(Float2GLM(Particles->HostPtr()[rootIdx + j].Velocity), i, j);
            } else if (SimParams.FixedHairLength) {
                int lastIdx = RootIdx->HostPtr()[i] + StrandLength[i] - 1;
                float3 dir = Particles->HostPtr()[lastIdx].Position - Particles->HostPtr()[lastIdx - 1].Position;
                dir = 0.001 * normalize(dir);
                hVerts->SetEntryToArray(Float2GLM(Particles->HostPtr()[lastIdx + j - 1].Position + dir), i, j);
                hVels->SetEntryToArray(Float2GLM(Particles->HostPtr()[lastIdx + j - 1].Velocity * damping), i, j);
                damping *= 0.5;
            }
        }
    }
}

void AMSSolver::SendData(weak_ptr<Hair> _hair, weak_ptr<Body> _body) {
    SendKamiHair(_hair);
    Meshes[0]->SendKamiHead(_body);
    HeadTriangles->CopyHostToDevice(Meshes[0]->GetTrianglesRaw().data());
    HeadVertices->CopyHostToDevice(Meshes[0]->GetVerticesRaw().data());
}

void AMSSolver::SendKamiHair(weak_ptr<Hair> hair) {
    auto& hVerts = hair.lock()->GetCurrentVerticesRef();
    auto& hVels = hair.lock()->GetCurrentVelocitiesRef();
    auto& numStrands = hair.lock()->GetStrandVertCountRef();

#pragma omp parallel for
    for (size_t i = 0; i < SimParams.NumRoots; i++) {
#pragma omp parallel for
        for (size_t j = 0; j < SimParams.MaxStrandLength; j++) {
            float damping = 0.5;
            if (j < numStrands[i]) {
                int rootIdx = RootIdx->HostPtr()[i];
                Particles->HostPtr()[rootIdx + j].Position = GLM2Float(hVerts->GetEntryVal(i, j));
                Particles->HostPtr()[rootIdx + j].Velocity = GLM2Float(hVels->GetEntryVal(i, j));
            } else if (SimParams.FixedHairLength) {
                int lastIdx = RootIdx->HostPtr()[i] + StrandLength[i] - 1;
                glm::vec3 dir = hVerts->GetEntryVal(i, j) - hVerts->GetEntryVal(i, j - 1);
                dir = 0.001f * glm::normalize(dir);
                Particles->HostPtr()[lastIdx + j - 1].Position = GLM2Float(hVerts->GetEntryVal(i, StrandLength[i] - 1) + dir);
                Particles->HostPtr()[lastIdx + j - 1].Velocity = GLM2Float(hVels->GetEntryVal(i, StrandLength[i] - 1) * 0.f);
                damping *= 0.5;
            }
        }
    }
    Particles->CopyHostToDevice();
}

void AMSSolver::UpdateAnimation(weak_ptr<Body> _body) {
    switch (SimParams.AnimationType) {
        case (SEQUENCE): {
            // transforms time to equivalent frame
            float lambda;
            int a, b;
            AnimTimeToFrame(a, b, lambda);

            // update vertices from sequence
            launchKernelUpdateAnimSeq(BlockThreadVert, a, b, lambda, DBuffers);

            // update velocity head (particle to grid)
            launchKernelUpdateVelocitySdf(BlockThreadSdf, BlockThreadVert, DBuffers);

            // update hair roots
            // launchKernelUpdateRootsSeq(BlockThreadRoot, DBuffers);

            // update corresponding box
            UpdateAnimGridBox(a, b, lambda);

            // update Sdf
            HeadVertices->CopyDeviceToHost();
            SdfCPU->UpdateVertices(HeadVertices->HostPtr());
            ComputeSDFCPU(_body);

            break;
        }
        case (WIND): {
            WindBlowing();
        }
        default:
            break;
    }
}

void AMSSolver::UpdateDynamics() {
    // Nested Integration
    for (size_t i = 0; i < SimParams.NestedSteps; i++) {
        // Prepare linear system
        launchKernelFillMatrices(BlockThreadHair, DBuffers);

        // Solve implicit Velocity
        launchKernelSolveVelocity(BlockThreadRoot, DBuffers);

        // Update positions
        launchKernelPositionUpdate(BlockThreadHair, DBuffers);
    }

    // Head collision
    if (SimParams.SolidCollision) launchKernelSDFHeadCollision(BlockThreadHair, DBuffers);

    // Strain limiting
    launchKernelStrainLimiting(BlockThreadRoot, DBuffers);

    // Hair-hair interaction
    if (SimParams.HairCollision) {
        // Eulerian part (coarse collision detection)

        // Rasterize particles into grid
        launchKernelSegmentToEulerianGrid(BlockThreadHair, BlockThreadGrid, BlockThreadGridU, BlockThreadGridV, BlockThreadGridW, DBuffers);

        // Enforce incompressible condition
        launchKernelProjectVelocity(BlockThreadGrid, BlockThreadGridU, BlockThreadGridV, BlockThreadGridW, SimParams.NumStepsJacobi, DBuffers);

        // Transfer velocity back to particles
        launchKernelEulerianGridToParticle(BlockThreadHair, DBuffers);

        // Lagrangian part (detailed collision detection)
    }

    // Swaps positions (x and x0)
    launchKernelSwapPositions(BlockThreadHair, DBuffers);

    // Update Cut Particles
    launchKernelCutHair(BlockThreadHair, DBuffers);
}

void AMSSolver::UpdateRoots() { launchKernelMoveRoots(BlockThreadHair, DBuffers); }

void AMSSolver::UpdateMesh() { launchKernelUpdateMesh(BlockThreadMesh, DBuffers); }

void AMSSolver::FixRootPositions() {
    // Gets particles on CPU
    Particles->CopyDeviceToHost();
    Particle* particlesCPU = Particles->HostPtr();

// Fills in parallel-cpu
#pragma omp parallel for
    for (size_t i = 0; i < SimParams.NumParticles; i++) {
        // Only moves roots
        if (particlesCPU[i].LocalIdx == 0) {
            PointSDF sample = SdfCPU->DistancePointField(particlesCPU[i].Position);
            float3 dirVector = sample.NearestPoint - particlesCPU[i].Position;
            for (int j = 0; j < particlesCPU[i].StrandLength; j++) {
                particlesCPU[i + j].Position += dirVector;
            }
        }
    }

    // Copy everything to GPU
    Particles->CopyHostToDevice();
}

void AMSSolver::WindBlowing() {
    float t = SimParams.SimTime;
    if (t < 1.f) {
        SimParams.WindSpeed.x = SimParams.WindPeak + 4.f * sinf(cosf(t));
    } else if (t < 3.f) {
        SimParams.WindSpeed.x = -SimParams.WindPeak + 4.f * sinf(cosf(t));
    } else if (t < 4.f) {
        SimParams.WindSpeed.x = SimParams.WindPeak + 4.f * sinf(cosf(t));
    } else if (t < 6.f) {
        SimParams.WindSpeed.x = -SimParams.WindPeak + 4.f * sinf(cosf(t));
    } else if (t < 8.f) {
        SimParams.WindSpeed.x = SimParams.WindPeak + 4.f * sinf(cosf(t));
    } else if (t < 12.f)
        SimParams.WindSpeed.x = -SimParams.WindPeak / 2.f;
    else
        SimParams.WindSpeed.x = 0.f;
}

void AMSSolver::FreeBuffers() {
    // TextSDF->destroy();
}

void AMSSolver::InitParams() {
    SimParams.RigidAngle = make_float3(0.f);
    SimParams.RigidPos = make_float3(0.f);

    // SimParams.CutMin = 4;

    SimParams.CutRadius = 0.4f;

    // Basic parameters
    SimParams.NumVertices = 0;
    SimParams.NumTriangles = 0;
    SimParams.NumParticles = 0;
    SimParams.NumRoots = 0;
    SimParams.NumSprings = 0;

    // Physical parameters
    SimParams.Dt = 0.02f;
    SimParams.Gravity = 26.18f;
    SimParams.AngularK = 50.f;
    SimParams.GravityK = 50.f;  // 0.23;
    SimParams.EdgeK = 50.f;
    SimParams.BendK = 50.f;
    SimParams.TorsionK = 100.f;  // 50.f;
    SimParams.Damping = 1.5;
    SimParams.HairMass = 0.1f;
    SimParams.Friction = 1.f;  // 3.5f;
    SimParams.WindSpeed = make_float3(0);
    SimParams.WindPeak = 3.0f;

    // Eulerian simulation
    SimParams.LengthIncreaseGrid = 0.3f;  // percentage of grid increase w.r.t. loaded obj
    SimParams.ThreshSdf = .015f;
    SimParams.NumGridNeighbors = 1;
    SimParams.NumStepsJacobi = 50;
    SimParams.JacobiWeight = 0.6f;
    SimParams.FlipWeight = 0.8f;  // 0.95f;

    // AMSSolver paremters
    SimParams.StrainError = 0.1;  // 0.2
    // SimParams.StrainSteps = 4;    // 10;
    SimParams.NestedSteps = 1;  // 10;

    // Animation
    SimParams.MeshSequence = false;
    SimParams.SimTime = 0.f;
    SimParams.AnimDuration = 27.f;
    SimParams.SimEndTime = -1.f;
    SimParams.NumFrames = 1;
    SimParams.RootMove = make_float3(0.f);

    // Preprocessing
    // SimParams.TooglePreProcess = true;
    // SimParams.RecomputeSdf = true;

    // Experiments manuscript
    // SimParams.Experiment = NONE;

    // Toogle Options
    SimParams.FixedHairLength = false;
    SimParams.Animate = true;
    SimParams.AnimationType = WIND;
    // SimParams.PauseSim = false;
    //  SimParams.ExportVDB = false;

    // Toogle Debug
    SimParams.SolidCollision = true;
    SimParams.HairCollision = true;
    SimParams.DrawHead = true;
    SimParams.DrawHeadWF = false;
}

void AMSSolver::ModifyParams(ModParams mp) {
    SimParams.WindPeak = mp.windPeak;
    SimParams.ThreshSdf = mp.SDFThreshold;
}

int AMSSolver::SelectCudaDevice() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device = deviceCount - 1;
    checkCudaErrors(cudaSetDevice(device));
    return device;
}

void AMSSolver::UpdateAnimGridBox(const int& a, const int& b, const float& lambda) {
    // Number of cells per dimension
    int nx = SimParams.HairDim.x;
    int ny = SimParams.HairDim.y;
    int nz = SimParams.HairDim.z;

    // Total size of grid embedding obj
    vector<vector<float3>>& boxes = Meshes[0]->GetAnimAABB();
    float3 minLambda = (1 - lambda) * boxes[a][0] + lambda * boxes[b][0];
    float3 maxLambda = (1 - lambda) * boxes[a][1] + lambda * boxes[b][1];

    vector<float3> minMax = {minLambda, maxLambda};
    float3 distances = minMax[1] - minMax[0];

    // Expands grid around obj
    float scale = SimParams.LengthIncreaseGrid;
    float3 dimIncrease = 0.5f * scale * distances;
    vector<float3> newMinMax = {minMax[0] - dimIncrease, minMax[1] + dimIncrease};

    // Stream info into solver
    float3 newDist = (newMinMax[1] - newMinMax[0]);
    float dx = newDist.x / (1.f * nx);
    float dy = newDist.y / (1.f * ny);
    float dz = newDist.z / (1.f * nz);

    // After bone transform (identity at t0)
    SimParams.HairCenter = 0.5 * (newMinMax[1] + newMinMax[0]);
    SimParams.HairAxis[0] = make_float3(1.f, 0.f, 0.f);
    SimParams.HairAxis[1] = make_float3(0.f, 1.f, 0.f);
    SimParams.HairAxis[2] = make_float3(0.f, 0.f, 1.f);
    SimParams.HairMin = newMinMax[0];

    // Static params
    SimParams.MaxWeight = 1.5 * max(dx, max(dy, dz)) * (SimParams.NumGridNeighbors + 1.f);
    SimParams.HairInvSqDs = make_float3(1.f / (dx * dx), 1.f / (dy * dy), 1.f / (dz * dz));
    SimParams.HairInvDs = make_float3(1.f / dx, 1.f / dy, 1.f / dz);
    SimParams.HairDs = make_float3(dx, dy, dz);
}

void AMSSolver::AnimTimeToFrame(int& a, int& b, float& lambda) {
    float normalTime = fmin(SimParams.SimTime, SimParams.AnimDuration) / SimParams.AnimDuration;
    a = floor(normalTime * (SimParams.NumFrames - 1));
    b = min(a + 1, SimParams.NumFrames - 1);
    lambda = normalTime * (SimParams.NumFrames - 1) - a;
}

int AMSSolver::DivUp(const int& a, const int& b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

void AMSSolver::UploadHair(weak_ptr<Hair> _hair, weak_ptr<Body> _body) { SendData(_hair, _body); }
void AMSSolver::DownloadHair(weak_ptr<Hair> _hair, weak_ptr<Body> _body) { WritebackData(_hair, _body); }