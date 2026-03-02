#include "Geometry.h"

Kami::SDF::SDF(const SDF& a) : discregridSDF(new Discregrid::CubicLagrangeDiscreteGrid(*a.discregridSDF)) {}

void Kami::SDF::Init(unq_ptr<glm::vec3[]>& vertexPos, unq_ptr<glm::uvec3[]>& triangleIndices, uint32_t numVertices, uint32_t numTriangles,
                     glm::uvec3 SDFDim) {
    std::vector<Eigen::Vector3d> pos(numVertices);
    std::vector<std::array<unsigned int, 3>> triInd(numTriangles);

#pragma omp parallel for
    for (uint32_t i = 0; i < numVertices; ++i) {
        auto p = vertexPos[i];
        pos[i] = Eigen::Vector3d(p.x, p.y, p.z);
    }

#pragma omp parallel for
    for (uint32_t i = 0; i < numTriangles; ++i) {
        auto t = triangleIndices[i];
        triInd[i] = {t.x, t.y, t.z};
    }

    Discregrid::TriangleMesh discrMesh(pos, triInd);
    Discregrid::TriangleMeshDistance md(discrMesh);
    Eigen::AlignedBox3d domain;
    for (auto const& x : discrMesh.vertices()) {
        domain.extend(x);
    }
    domain.max() += 0.1 * Eigen::Vector3d::Ones();
    domain.min() -= 0.1 * Eigen::Vector3d::Ones();

    discregridSDF.reset(new Discregrid::CubicLagrangeDiscreteGrid(domain, std::array<unsigned int, 3>({SDFDim.x, SDFDim.y, SDFDim.z})));
    auto func = Discregrid::DiscreteGrid::ContinuousFunction{};
    func = [&md](Eigen::Vector3d const& xi) { return md.signed_distance(xi).distance; };
    discregridSDF->addFunction(func, false);
}

double Kami::SDF::Distance(const glm::vec3& pos, const float thickness) {
    Eigen::Vector3d normal, doublePos = Eigen::Vector3d(pos.x, pos.y, pos.z);
    double doubleDist = discregridSDF->interpolate(0, doublePos);
    if (doubleDist == std::numeric_limits<float>::max()) return false;
    return doubleDist - thickness;
}

bool Kami::SDF::CollisionTest(const glm::vec3& pos, glm::vec3& cp, glm::vec3& n, float& dist, const float thickness, const float maxDist) {
    Eigen::Vector3d normal, doublePos = Eigen::Vector3d(pos.x, pos.y, pos.z);
    double doubleDist = discregridSDF->interpolate(0, doublePos, &normal);
    if (doubleDist == std::numeric_limits<float>::max()) return false;
    dist = static_cast<float>(doubleDist - thickness);

    if (dist < maxDist) {
        normal.normalize();
        n = glm::vec3(normal[0], normal[1], normal[2]);

        cp = (pos - dist * n);

        return true;
    }
    return false;
}