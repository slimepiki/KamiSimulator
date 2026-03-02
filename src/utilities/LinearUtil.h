#ifndef LINEAR_UTIL_H_
#define LINEAR_UTIL_H_

#include "../Kami.h"
#include <Eigen/Sparse>
#include "../extern/glm/glm/gtc/random.hpp"

namespace Kami::LinearUtil {
const float EPSILON = 1e-6;

// Vector3<float> 1Darray wrapper.
class V3f1DArray {
   public:
    V3f1DArray(uint32_t _size) : size(_size) {};

    uint32_t GetSize() const;
    uint32_t GetVertCount() const;

    virtual void SetEntryToArray(glm::vec3 entry, uint32_t x) = 0;
    virtual void AddEntryToArray(glm::vec3 entry, uint32_t x) = 0;

    virtual void Reset(float val) = 0;
    virtual void Resize(uint32_t _size, float emptyVal = 0) = 0;
    virtual glm::vec3 GetEntryVal(uint32_t x) const = 0;
    virtual glm::vec3& GetEntryRef(uint32_t x) = 0;

    virtual void* GetRawArray() = 0;
    virtual void* GetRawArray() const = 0;

   protected:
    uint32_t size;
    bool CheckIndexIsValid(uint32_t x) const;

   private:
    V3f1DArray() : size(0) {};
};

// glm::vec3[] wrapper
class GLMV3f1DArray : V3f1DArray {
   public:
    GLMV3f1DArray(uint32_t _size);
    GLMV3f1DArray(const GLMV3f1DArray& src);

    void SetEntryToArray(glm::vec3 entry, uint32_t x) override;
    void AddEntryToArray(glm::vec3 entry, uint32_t x) override;

    void Reset(float val = 0) override;  // isn't tested
    void Resize(uint32_t _size, float emptyVal = 0) override;

    glm::vec3 GetEntryVal(uint32_t x) const override;
    glm::vec3& GetEntryRef(uint32_t x) override;

    void* GetRawArray() override;
    void* GetRawArray() const override;

    GLMV3f1DArray& operator=(GLMV3f1DArray& src);

   private:
    unq_ptr<glm::vec3[]> rawArray;
    GLMV3f1DArray() : V3f1DArray(0) {};
};

// vec3 2Darray like EigenVectorXf wrapper
class EigenVecX3f1DArray : public V3f1DArray {
   public:
    EigenVecX3f1DArray(uint32_t _size);
    EigenVecX3f1DArray(const EigenVecX3f1DArray& src);

    void SetEntryToArray(glm::vec3 entry, uint32_t x) override;
    void AddEntryToArray(glm::vec3 entry, uint32_t x) override;

    void Reset(float val = 0) override;  // isn't tested
    void Resize(uint32_t _size, float emptyVal = 0) override;

    glm::vec3 GetEntryVal(uint32_t x) const override;
    // This function is unavailable in EigenVecX3f2DArray.
    glm::vec3& GetEntryRef(uint32_t x) override;

    void* GetRawArray() override;
    void* GetRawArray() const override;

    Eigen::VectorXf& GetEigenVecXf();
    const Eigen::VectorXf& GetEigenVecXf() const;

    EigenVecX3f1DArray& operator=(EigenVecX3f1DArray& src);

   private:
    Eigen::VectorXf rawVec;
    glm::vec3 errorVec = glm::vec3(0.0f, 0.0f, 0.0f);

    EigenVecX3f1DArray();
};

// Vector3<float> 2Darray wrapper.
class V3f2DArray {
   public:
    V3f2DArray(glm::uvec2 _size) : size(_size) {};

    glm::uvec2 GetSize() const;
    uint32_t GetVertCount() const;
    uint32_t GetIndex(uint32_t x, uint32_t y) const;

    virtual void SetEntryToArray(glm::vec3 entry, uint32_t x, uint32_t y) = 0;
    virtual void AddEntryToArray(glm::vec3 entry, uint32_t x, uint32_t y) = 0;

    virtual void Reset(float val) = 0;  // isn't tested
    virtual void Resize(glm::uvec2 _size, float emptyVal = 0) = 0;

    virtual glm::vec3 GetEntryVal(uint32_t x, uint32_t y) const = 0;
    virtual glm::vec3& GetEntryRef(uint32_t x, uint32_t y) = 0;

    virtual void* GetRawArray() = 0;
    virtual void* GetRawArray() const = 0;

   protected:
    glm::uvec2 size;

   private:
    bool CheckIndexIsValid(uint32_t x, uint32_t y) const;
    V3f2DArray() : size(glm::uvec2(0, 0)) {};
};

// glm::vec3[][] wrapper
class GLMV3f2DArray : public V3f2DArray {
   public:
    GLMV3f2DArray(glm::uvec2 _size);
    GLMV3f2DArray(const GLMV3f2DArray& src);

    void SetEntryToArray(glm::vec3 entry, uint32_t x, uint32_t y) override;
    void AddEntryToArray(glm::vec3 entry, uint32_t x, uint32_t y) override;

    void Reset(float val = 0) override;  // isn't tested
    void Resize(glm::uvec2 _size, float emptyVal = 0) override;

    glm::vec3 GetEntryVal(uint32_t x, uint32_t y) const override;
    glm::vec3& GetEntryRef(uint32_t x, uint32_t y) override;

    void* GetRawArray() override;
    void* GetRawArray() const override;

    GLMV3f2DArray& operator=(GLMV3f2DArray& src);

   private:
    unq_ptr<glm::vec3[]> rawArray;

    GLMV3f2DArray();
};

// vec3 2Darray like EigenVectorXf wrapper
class EigenVecX3f2DArray : public V3f2DArray {
   public:
    EigenVecX3f2DArray(glm::uvec2 _size);
    EigenVecX3f2DArray(const EigenVecX3f2DArray& src);

    void SetEntryToArray(glm::vec3 entry, uint32_t x, uint32_t y) override;
    void AddEntryToArray(glm::vec3 entry, uint32_t x, uint32_t y) override;

    void Reset(float val = 0) override;  // isn't tested
    void Resize(glm::uvec2 _size, float emptyVal = 0) override;

    glm::vec3 GetEntryVal(uint32_t x, uint32_t y) const override;
    // This function is unavailable in EigenVecX3f2DArray.
    glm::vec3& GetEntryRef(uint32_t x, uint32_t y) override;

    void* GetRawArray() override;
    void* GetRawArray() const override;

    Eigen::VectorXf& GetEigenVecXf();
    const Eigen::VectorXf& GetEigenVecXf() const;

    EigenVecX3f2DArray& operator=(EigenVecX3f2DArray& src);

   private:
    Eigen::VectorXf rawVec;
    glm::vec3 errorVec = glm::vec3(0.0f, 0.0f, 0.0f);

    EigenVecX3f2DArray();
};

// Vector3<float> 3Darray wrapper.
class V3f3DArray {
   public:
    // If useXYas UV == true, x is u, y is v and z is w.
    // In other words, [v][u][w].
    V3f3DArray(glm::uvec3 _size, bool _useXYasUV = false) : size(_size), useXYasUV(_useXYasUV) {};

    glm::uvec3 GetSize() const;
    bool GetUVCoordOrNot() const;
    uint32_t GetVertCount() const;
    uint32_t GetIndex(uint32_t x, uint32_t y, uint32_t z) const;

    virtual void SetEntryToArray(glm::vec3 entry, uint32_t x, uint32_t y, uint32_t z) = 0;
    virtual void AddEntryToArray(glm::vec3 entry, uint32_t x, uint32_t y, uint32_t z) = 0;

    virtual void Reset(float val) = 0;  // isn't tested
    virtual void Resize(glm::uvec3 _size, float emptyVal = 0) = 0;

    virtual glm::vec3 GetEntryVal(uint32_t x, uint32_t y, uint32_t z) const = 0;
    virtual glm::vec3& GetEntryRef(uint32_t x, uint32_t y, uint32_t z) = 0;

    virtual void* GetRawArray() = 0;
    virtual void* GetRawArray() const = 0;

   protected:
    glm::uvec3 size;
    const bool useXYasUV;

   private:
    bool CheckIndexIsValid(uint32_t x, uint32_t y, uint32_t z) const;
    V3f3DArray() : size(glm::vec3(0, 0, 0)), useXYasUV(false) {};
};

// glm::vec3[][][] wrapper
// This class haven't been tested.
class GLMV3f3DArray : public V3f3DArray {
   public:
    // If useXYas UV == true, x is u, y is v and z is w.
    // In other words, [v][u][w].
    GLMV3f3DArray(glm::uvec3 _size, bool useXYasUV = false);
    GLMV3f3DArray(const GLMV3f3DArray& src);

    void SetEntryToArray(glm::vec3 entry, uint32_t x, uint32_t y, uint32_t z) override;
    void AddEntryToArray(glm::vec3 entry, uint32_t x, uint32_t y, uint32_t z) override;

    void Reset(float val = 0) override;  // isn't tested
    void Resize(glm::uvec3 _size, float emptyVal = 0) override;

    glm::vec3 GetEntryVal(uint32_t x, uint32_t y, uint32_t z) const override;
    glm::vec3& GetEntryRef(uint32_t x, uint32_t y, uint32_t z) override;

    void* GetRawArray() override;
    void* GetRawArray() const override;
    GLMV3f3DArray& operator=(GLMV3f3DArray& src);

   private:
    unq_ptr<glm::vec3[]> rawArray;

    GLMV3f3DArray();
};

// vec3 3Darray like EigenVectorXf wrapper
class EigenVecX3f3DArray : public V3f3DArray {
   public:
    // If useXYas UV == true, x is u, y is v and z is w.
    // In other words, [v][u][w].
    EigenVecX3f3DArray(glm::uvec3 _size, bool useXYasUV = false);
    EigenVecX3f3DArray(const EigenVecX3f3DArray& src);

    void SetEntryToArray(glm::vec3 entry, uint32_t x, uint32_t y, uint32_t z) override;
    void AddEntryToArray(glm::vec3 entry, uint32_t x, uint32_t y, uint32_t z) override;

    void Reset(float val = 0) override;  // isn't tested
    void Resize(glm::uvec3 _size, float emptyVal = 0) override;

    glm::vec3 GetEntryVal(uint32_t x, uint32_t y, uint32_t z) const override;
    // This function is unavailable in EigenVecX3Df3DArray.
    glm::vec3& GetEntryRef(uint32_t x, uint32_t y, uint32_t z) override;

    void* GetRawArray() override;
    void* GetRawArray() const override;
    Eigen::VectorXf& GetEigenVecXf();
    const Eigen::VectorXf& GetEigenVecXf() const;

    EigenVecX3f3DArray& operator=(EigenVecX3f3DArray& src);

   private:
    Eigen::VectorXf rawVec;
    glm::vec3 errorVec = glm::vec3(0.0f, 0.0f, 0.0f);

    EigenVecX3f3DArray();
};

inline glm::vec3 CalcDir(glm::vec3 start, glm::vec3 end) {
    glm::vec3 edge = end - start;
    if (glm::length(edge) < EPSILON) {
        return glm::ballRand(1.f);
    } else {
        return glm::normalize(edge);
    }
}
// This function returns zero vector if the edge is shorter than EPSILON.
inline glm::vec3 CalcDirZ(glm::vec3 start, glm::vec3 end) {
    glm::vec3 edge = end - start;
    if (glm::length(edge) < EPSILON) {
        return glm::vec3(0.0f, 0.0f, 0.0f);
    } else {
        return glm::normalize(edge);
    }
}

class Eigen3x3BlockMat {
   public:
    // x * y [3x3Matrix]
    Eigen3x3BlockMat(uint32_t row, uint32_t column);
    void PushEntry(glm::mat3 mat, uint32_t row, uint32_t column);
    void ApplyEntry();
    glm::mat3 GetEntry(uint32_t row, uint32_t column);
    void Reset();

    unq_ptr<Eigen::SparseMatrix<float>>& GetEigenMatrix();

   private:
    unq_ptr<Eigen::SparseMatrix<float>> eigenSparseMat;
    unq_ptr<vector<Eigen::Triplet<float>>> entryQueue;
    glm::uvec2 size;

    inline void IndexValidation(uint32_t row, uint32_t column) {
#ifdef DEBUG_MODE
        if (row < 0 || row >= size.x) {
            cout << __func__ << ": " << to_string(row) + " is Invalid index(row)." << endl;
            throw std::runtime_error("");
        }
        if (column < 0 || column >= size.x) {
            cout << __func__ << ": " << to_string(column) + " is Invalid index(column)." << endl;
            throw std::runtime_error("");
        }
#endif
    };
};

// void SolveLinearEquation();
glm::vec3 GetTranslateFrom4x4Mat(const glm::mat4 mat);

}  // namespace Kami::LinearUtil

#endif /* LINEAR_UTIL_H_ */