#pragma once
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>
#include "SimpleTriangle.cuh"
#include "../../Body.h"

#define ASSIMP_LOAD_FLAGS (aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices | aiProcess_PopulateArmatureData)
using std::shared_ptr;
using std::string;
using std::weak_ptr;

class Body;

class SimpleObject {
   public:
    SimpleObject() {};
    SimpleObject(const string& fname, const float3& pos = make_float3(0.f), const float3& scale = make_float3(1.f), const bool& anim = false,
                 const int& numFrames = 1);
    SimpleObject(const string& fname, const float4& pos, const bool& anim = false, const int& numFrames = 1);
    SimpleObject(weak_ptr<Body> body, const string& gridname, const bool& anim = false, const int& numFrames = 1);
    ~SimpleObject();

    // Animation
    void ExportObj(const string& fname);
    void UpdateVertices();
    void UpdateVerticesFromSeq(SimpleVertex* vert);

    // Getters/Setters
    vector<shared_ptr<SimpleTriangle>>& GetTriangles() { return TrianglesObj; }
    vector<shared_ptr<int>>& GetIndices() { return Indices; }
    vector<shared_ptr<SimpleVertex>>& GetVertices() { return Vertices; }
    vector<SimpleTriangle> GetTrianglesRaw();
    vector<SimpleVertex> GetVerticesRaw();
    vector<MiniVertex> GetAnimVerticesRaw();
    int GetNumTriangles() { return NumTriangles; }
    int GetNumVertices() { return NumVertices; }
    int GetNumFrames() { return NumFrames; }
    vector<float3>& GetAABB() { return AABB; }
    vector<vector<float3>>& GetAnimAABB() { return AnimAABB; }
    vector<float3>& GetAABB0() { return AABB0; }

    void WriteBackKamiHead(weak_ptr<Body> body);
    void SendKamiHead(weak_ptr<Body> body);

   private:
    // Animation
    vector<vector<shared_ptr<MiniVertex>>> AnimVertex;
    vector<vector<float3>> AnimAABB;
    vector<bool> InsideGrid;
    bool Animated = false;
    int NumFrames = 1;

    // Mesh Geometry
    vector<shared_ptr<SimpleTriangle>> TrianglesObj;
    vector<shared_ptr<SimpleVertex>> Vertices;
    int NumVertices, NumTriangles, NumIndices;
    vector<shared_ptr<int>> Indices;
    vector<float3> AABB, AABB0;

    // Bones
    vector<hcuMat4> FinalTrans;
    hcuMat4 MasterTrans = hcuMat4::Identity();
    hcuMat4 MasterTransInv = hcuMat4::Identity();
    int NumBones = 0;

    // Animation Helpers
    void ComputeAABBSequence();
    void LoadMeshSequence(const string& fname, const float3& pos, const float3& scale);
    void FillGridInfo(const string& fname);

    // General Helpers
    void ComputeAABB();

    // Bone Helpers
    void GenerateDummyBone();

    // Assimp Helpers
    void LoaderAssimp(const string& fname, const float3& pos, const float3& scale);
    hcuMat4 AssimpToMat4(const aiMatrix4x4& mat);
    void ProcessNode(aiNode* node, const aiScene* scene);
    void ProcessMesh(aiMesh* mesh, const aiScene* scene);
    void ProcessNodeSequence(aiNode* node, const aiScene* scene, const int& frame);
    void ProcessMeshSequence(aiMesh* mesh, const aiScene* scene, const int& frame);
    void InitBoneData(SimpleVertex& v);
    void BuildTriangles();
};
