#include "LinearUtil.h"
#include "Notification.h"

using Eigen::Triplet;
// ########################################
// ####################V3f1DArray##########
// ########################################
uint32_t Kami::LinearUtil::V3f1DArray::GetSize() const { return size; }

uint32_t Kami::LinearUtil::V3f1DArray::GetVertCount() const { return size; }

bool Kami::LinearUtil::V3f1DArray::CheckIndexIsValid(uint32_t x) const {
#ifdef DEBUG_MODE
    if (x >= size) {
        Kami::Notification::Warn(__func__, "index excession");
        return false;
    };
#endif
    return true;
}

// ########################################
// ####################GLMV3f1DArray#######
// ########################################

Kami::LinearUtil::GLMV3f1DArray::GLMV3f1DArray(uint32_t _size) : V3f1DArray(_size) {
    rawArray = make_unique<glm::vec3[]>(GetVertCount());
    Reset();
}

Kami::LinearUtil::GLMV3f1DArray::GLMV3f1DArray(const GLMV3f1DArray& src) : V3f1DArray(src.GetSize()) {
    rawArray = make_unique<glm::vec3[]>(this->GetVertCount());
    std::memcpy(rawArray.get(), src.GetRawArray(), this->GetVertCount() * sizeof(glm::vec3));
}

void Kami::LinearUtil::GLMV3f1DArray::SetEntryToArray(glm::vec3 entry, uint32_t x) {
    CheckIndexIsValid(x);
    rawArray[x] = entry;
}
void Kami::LinearUtil::GLMV3f1DArray::AddEntryToArray(glm::vec3 entry, uint32_t x) {
    CheckIndexIsValid(x);
    rawArray[x] += entry;
}

void Kami::LinearUtil::GLMV3f1DArray::Reset(float val) {
    for (uint32_t i = 0; i < GetVertCount(); i++) {
        rawArray[i] = glm::vec3(val, val, val);
    }
}

void Kami::LinearUtil::GLMV3f1DArray::Resize(uint32_t _size, float emptyVal) {
    unq_ptr<glm::vec3[]> newArray = make_unique<glm::vec3[]>(_size);

    for (uint32_t i = 0; i < _size; i++) {
        newArray[i] = glm::vec3(emptyVal, emptyVal, emptyVal);
    }

    uint32_t copysize = std::min(_size, size);
    std::copy(rawArray.get(), rawArray.get() + copysize, newArray.get());
    rawArray = std::move(newArray);

    size = _size;
}

glm::vec3 Kami::LinearUtil::GLMV3f1DArray::GetEntryVal(uint32_t x) const {
    CheckIndexIsValid(x);
    return rawArray[x];
}

glm::vec3& Kami::LinearUtil::GLMV3f1DArray::GetEntryRef(uint32_t x) {
    CheckIndexIsValid(x);
    return rawArray[x];
}

void* Kami::LinearUtil::GLMV3f1DArray::GetRawArray() { return (void*)rawArray.get(); }

void* Kami::LinearUtil::GLMV3f1DArray::GetRawArray() const { return (void*)rawArray.get(); }

Kami::LinearUtil::GLMV3f1DArray& Kami::LinearUtil::GLMV3f1DArray::operator=(GLMV3f1DArray& src) {
    new (this) GLMV3f1DArray(src);
    return *this;
}

// ########################################
// ############EigenVecX3f1DArray#########
// ########################################

Kami::LinearUtil::EigenVecX3f1DArray::EigenVecX3f1DArray(uint32_t _size) : V3f1DArray(_size) { rawVec.resize(GetVertCount() * 3); }

Kami::LinearUtil::EigenVecX3f1DArray::EigenVecX3f1DArray(const EigenVecX3f1DArray& src) : V3f1DArray(src.GetSize()) { rawVec = src.GetEigenVecXf(); }

void Kami::LinearUtil::EigenVecX3f1DArray::SetEntryToArray(glm::vec3 entry, uint32_t x) {
    CheckIndexIsValid(x);

    rawVec[x * 3] = entry.x;
    rawVec[x * 3 + 1] = entry.y;
    rawVec[x * 3 + 2] = entry.z;
}

void Kami::LinearUtil::EigenVecX3f1DArray::AddEntryToArray(glm::vec3 entry, uint32_t x) {
    CheckIndexIsValid(x);

    rawVec[x * 3] += entry.x;
    rawVec[x * 3 + 1] += entry.y;
    rawVec[x * 3 + 2] += entry.z;
}

void Kami::LinearUtil::EigenVecX3f1DArray::Reset(float val) {
    if (val == 0)
        rawVec.setZero();
    else
        rawVec.setConstant(val);
}

void Kami::LinearUtil::EigenVecX3f1DArray::Resize(uint32_t _size, float emptyVal) {
    rawVec.conservativeResize(_size * 3);
    uint32_t vertSize = rawVec.size() / 3;

    if (emptyVal != 0 && _size > vertSize) {
        for (uint32_t i = std::min(vertSize, _size) * 3; i < _size; i += 3) {
            rawVec[i * 3] += emptyVal;
            rawVec[i * 3 + 1] += emptyVal;
            rawVec[i * 3 + 2] += emptyVal;
        }
    }
}

glm::vec3 Kami::LinearUtil::EigenVecX3f1DArray::GetEntryVal(uint32_t x) const {
    CheckIndexIsValid(x);
    return glm::vec3(rawVec[x], rawVec[x + 1], rawVec[x + 2]);
}

glm::vec3& Kami::LinearUtil::EigenVecX3f1DArray::GetEntryRef(uint32_t x) {
    Kami::Notification::Warn(__func__, "This function is undifined in EigenVecX3Df1DArray. Please use \033[31m GetEntryRefA() \033[0m.");
    return errorVec;
}

void* Kami::LinearUtil::EigenVecX3f1DArray::GetRawArray() { return (void*)rawVec.data(); }

void* Kami::LinearUtil::EigenVecX3f1DArray::GetRawArray() const { return (void*)rawVec.data(); }

Eigen::VectorXf& Kami::LinearUtil::EigenVecX3f1DArray::GetEigenVecXf() { return rawVec; }

const Eigen::VectorXf& Kami::LinearUtil::EigenVecX3f1DArray::GetEigenVecXf() const { return rawVec; }

Kami::LinearUtil::EigenVecX3f1DArray& Kami::LinearUtil::EigenVecX3f1DArray::operator=(EigenVecX3f1DArray& src) {
    new (this) EigenVecX3f1DArray(src);
    return *this;
}

// ########################################
// ####################V3f2DArray##########
// ########################################

glm::uvec2 Kami::LinearUtil::V3f2DArray::GetSize() const { return size; }

uint32_t Kami::LinearUtil::V3f2DArray::GetVertCount() const { return size.x * size.y; }

uint32_t Kami::LinearUtil::V3f2DArray::GetIndex(uint32_t x, uint32_t y) const { return size.y * x + y; }

bool Kami::LinearUtil::V3f2DArray::CheckIndexIsValid(uint32_t x, uint32_t y) const {
#ifdef DEBUG_MODE
    if (x >= size.x) {
        Kami::Notification::Warn(__func__, "x index excession");
        return false;
    }
    if (y >= size.y) {
        Kami::Notification::Warn(__func__, "y index excession");
        return false;
    }
#endif
    return true;
}

// ########################################
// ####################GLMV3f2DArray#######
// ########################################

Kami::LinearUtil::GLMV3f2DArray::GLMV3f2DArray(glm::uvec2 _size) : V3f2DArray(_size) {
    rawArray = make_unique<glm::vec3[]>(GetVertCount());
    Reset();
}

Kami::LinearUtil::GLMV3f2DArray::GLMV3f2DArray(const GLMV3f2DArray& src) : V3f2DArray(src.GetSize()) {
    rawArray = make_unique<glm::vec3[]>(this->GetVertCount());
    std::memcpy(rawArray.get(), src.GetRawArray(), this->GetVertCount() * sizeof(glm::vec3));
}

void Kami::LinearUtil::GLMV3f2DArray::SetEntryToArray(glm::vec3 entry, uint32_t x, uint32_t y) { rawArray[GetIndex(x, y)] = entry; }
void Kami::LinearUtil::GLMV3f2DArray::AddEntryToArray(glm::vec3 entry, uint32_t x, uint32_t y) { rawArray[GetIndex(x, y)] += entry; }

void Kami::LinearUtil::GLMV3f2DArray::Reset(float val) {
    for (uint32_t i = 0; i < GetVertCount(); i++) {
        rawArray[i] = glm::vec3(val, val, val);
    }
}

void Kami::LinearUtil::GLMV3f2DArray::Resize(glm::uvec2 _size, float emptyVal) {
    auto newSisze = _size.x * _size.y;
    unq_ptr<glm::vec3[]> newArray = make_unique<glm::vec3[]>(newSisze);

    for (uint32_t i = 0; i < newSisze; i++) {
        newArray[i] = glm::vec3(emptyVal, emptyVal, emptyVal);
    }

    uint32_t copysizeX = std::min(size.x, _size.x);
    uint32_t copysizeY = std::min(size.y, _size.y);

    for (uint32_t i = 0; i < copysizeX; i++) {
        auto newStart = size.y * i;
        auto start = GetIndex(i, 0);
        auto end = GetIndex(i, copysizeY);
        std::copy(rawArray.get() + start, rawArray.get() + end, newArray.get() + newStart);
    }
    rawArray = std::move(newArray);

    size = _size;
}

glm::vec3 Kami::LinearUtil::GLMV3f2DArray::GetEntryVal(uint32_t x, uint32_t y) const {
    uint32_t index = GetIndex(x, y);
    return rawArray[index];
}

glm::vec3& Kami::LinearUtil::GLMV3f2DArray::GetEntryRef(uint32_t x, uint32_t y) {
    uint32_t index = GetIndex(x, y);
    return rawArray[index];
}

void* Kami::LinearUtil::GLMV3f2DArray::GetRawArray() { return (void*)rawArray.get(); }

void* Kami::LinearUtil::GLMV3f2DArray::GetRawArray() const { return (void*)rawArray.get(); }

Kami::LinearUtil::GLMV3f2DArray& Kami::LinearUtil::GLMV3f2DArray::operator=(GLMV3f2DArray& src) {
    new (this) GLMV3f2DArray(src);
    return *this;
}

// ########################################
// ############EigenVecX3f2DArray#########
// ########################################

Kami::LinearUtil::EigenVecX3f2DArray::EigenVecX3f2DArray(glm::uvec2 _size) : V3f2DArray(_size) { rawVec.resize(GetVertCount() * 3); }

Kami::LinearUtil::EigenVecX3f2DArray::EigenVecX3f2DArray(const EigenVecX3f2DArray& src) : V3f2DArray(src.GetSize()) { rawVec = src.GetEigenVecXf(); }

void Kami::LinearUtil::EigenVecX3f2DArray::SetEntryToArray(glm::vec3 entry, uint32_t x, uint32_t y) {
    uint32_t index = GetIndex(x, y);

    rawVec[index * 3] = entry.x;
    rawVec[index * 3 + 1] = entry.y;
    rawVec[index * 3 + 2] = entry.z;
}

void Kami::LinearUtil::EigenVecX3f2DArray::AddEntryToArray(glm::vec3 entry, uint32_t x, uint32_t y) {
    uint32_t index = GetIndex(x, y);

    rawVec[index * 3] += entry.x;
    rawVec[index * 3 + 1] += entry.y;
    rawVec[index * 3 + 2] += entry.z;
}

void Kami::LinearUtil::EigenVecX3f2DArray::Reset(float val) {
    if (val == 0)
        rawVec.setZero();
    else
        rawVec.setConstant(val);
}

void Kami::LinearUtil::EigenVecX3f2DArray::Resize(glm::uvec2 _size, float emptyVal) {
    auto newVecSize = _size.x * _size.y * 3;
    Eigen::VectorXf newVec(newVecSize);

    if (emptyVal != 0)
        newVec.setConstant(newVecSize);
    else
        newVec.setZero();

    uint32_t copysizeX = std::min(size.x, _size.x);
    uint32_t copysizeY = std::min(size.y, _size.y);

    for (uint32_t i = 0; i < copysizeX; ++i) {
        auto newStart = _size.y * i * 3;
        auto start = GetIndex(i, 0) * 3;
        auto end = GetIndex(i, copysizeY) * 3;
        std::copy(rawVec.data() + start, rawVec.data() + end, newVec.data() + newStart);
    }

    size = _size;
    rawVec = newVec;
}

glm::vec3 Kami::LinearUtil::EigenVecX3f2DArray::GetEntryVal(uint32_t x, uint32_t y) const {
    auto index = GetIndex(x, y) * 3;
    return glm::vec3(rawVec[index], rawVec[index + 1], rawVec[index + 2]);
}

glm::vec3& Kami::LinearUtil::EigenVecX3f2DArray::GetEntryRef(uint32_t x, uint32_t y) {
    Kami::Notification::Warn(__func__, "This function is undifined in EigenVecX3Df2DArray. Please use \033[31m GetEntryRefA() \033[0m.");
    return errorVec;
}

void* Kami::LinearUtil::EigenVecX3f2DArray::GetRawArray() { return (void*)rawVec.data(); }

void* Kami::LinearUtil::EigenVecX3f2DArray::GetRawArray() const { return (void*)rawVec.data(); }

Eigen::VectorXf& Kami::LinearUtil::EigenVecX3f2DArray::GetEigenVecXf() { return rawVec; }

const Eigen::VectorXf& Kami::LinearUtil::EigenVecX3f2DArray::GetEigenVecXf() const { return rawVec; }

Kami::LinearUtil::EigenVecX3f2DArray& Kami::LinearUtil::EigenVecX3f2DArray::operator=(EigenVecX3f2DArray& src) {
    new (this) EigenVecX3f2DArray(src);
    return *this;
}

// ########################################
// ####################V3f3DArray##########
// ########################################
glm::uvec3 Kami::LinearUtil::V3f3DArray::GetSize() const { return size; }

bool Kami::LinearUtil::V3f3DArray::GetUVCoordOrNot() const { return useXYasUV; }

uint32_t Kami::LinearUtil::V3f3DArray::GetVertCount() const { return size.x * size.y * size.z; }

uint32_t Kami::LinearUtil::V3f3DArray::GetIndex(uint32_t x, uint32_t y, uint32_t z) const {
#ifdef DEBUG_MODE
    CheckIndexIsValid(x, y, z);
#endif
    if (useXYasUV) {
        return size.z * size.x * y + size.z * x + z;
    } else {
        return size.z * size.y * x + size.z * y + z;
    }
}

bool Kami::LinearUtil::V3f3DArray::CheckIndexIsValid(uint32_t x, uint32_t y, uint32_t z) const {
#ifdef DEBUG_MODE
    if (x >= size.x) {
        Kami::Notification::Warn(__func__, "x index excession");
        return false;
    }
    if (y >= size.y) {
        Kami::Notification::Warn(__func__, "y index excession");
        return false;
    }
    if (z >= size.z) {
        Kami::Notification::Warn(__func__, "z index excession");
        return false;
    }
#endif
    return true;
}

// ########################################
// ####################GLMV3f3DArray#############
// ########################################

Kami::LinearUtil::GLMV3f3DArray::GLMV3f3DArray(glm::uvec3 _size, bool _useXYasUV) : V3f3DArray(_size, _useXYasUV) {
    rawArray = make_unique<glm::vec3[]>(GetVertCount());
    Reset();
}

Kami::LinearUtil::GLMV3f3DArray::GLMV3f3DArray(const GLMV3f3DArray& src) : V3f3DArray(src.GetSize(), src.GetUVCoordOrNot()) {
    rawArray = make_unique<glm::vec3[]>(this->GetVertCount());
    std::memcpy(rawArray.get(), src.GetRawArray(), this->GetVertCount() * sizeof(glm::vec3));
}

void* Kami::LinearUtil::GLMV3f3DArray::GetRawArray() { return (void*)rawArray.get(); }
void* Kami::LinearUtil::GLMV3f3DArray::GetRawArray() const { return (void*)rawArray.get(); }

void Kami::LinearUtil::GLMV3f3DArray::SetEntryToArray(glm::vec3 entry, uint32_t x, uint32_t y, uint32_t z) { rawArray[GetIndex(x, y, z)] = entry; }

void Kami::LinearUtil::GLMV3f3DArray::AddEntryToArray(glm::vec3 entry, uint32_t x, uint32_t y, uint32_t z) { rawArray[GetIndex(x, y, z)] += entry; }

void Kami::LinearUtil::GLMV3f3DArray::Reset(float val) {
    for (uint32_t i = 0; i < GetVertCount(); i++) {
        rawArray[i] = glm::vec3(val, val, val);
    }
}

void Kami::LinearUtil::GLMV3f3DArray::Resize(glm::uvec3 _size, float emptyVal) {
    auto newSisze = _size.x * _size.y * _size.z;
    unq_ptr<glm::vec3[]> newArray = make_unique<glm::vec3[]>(newSisze);

    for (uint32_t i = 0; i < newSisze; i++) {
        newArray[i] = glm::vec3(emptyVal, emptyVal, emptyVal);
    }

    uint32_t copysizeX = std::min(size.x, _size.x);
    uint32_t copysizeY = std::min(size.y, _size.y);
    uint32_t copysizeZ = std::min(size.z, _size.z);

    for (uint32_t i = 0; i < copysizeX; i++) {
        for (uint32_t j = 0; j < copysizeY; j++) {
            uint32_t newStart;

            if (useXYasUV) {
                newStart = _size.z * _size.x * j + _size.z * i;
            } else {
                newStart = _size.z * _size.y * i + _size.z * j;
            }

            auto start = GetIndex(i, j, 0);
            auto end = GetIndex(i, j, copysizeZ);
            std::copy(rawArray.get() + start, rawArray.get() + end, newArray.get() + newStart);
        }
    }
    rawArray = std::move(newArray);

    size = _size;
}

glm::vec3 Kami::LinearUtil::GLMV3f3DArray::GetEntryVal(uint32_t x, uint32_t y, uint32_t z) const { return rawArray[GetIndex(x, y, z)]; }

glm::vec3& Kami::LinearUtil::GLMV3f3DArray::GetEntryRef(uint32_t x, uint32_t y, uint32_t z) { return rawArray[GetIndex(x, y, z)]; }

Kami::LinearUtil::GLMV3f3DArray& Kami::LinearUtil::GLMV3f3DArray::operator=(GLMV3f3DArray& src) {
    new (this) GLMV3f3DArray(src);
    return *this;
}

// ########################################
// ############EigenVecX3f3DArray#########
// ########################################
Kami::LinearUtil::EigenVecX3f3DArray::EigenVecX3f3DArray(glm::uvec3 _size, bool _useXYasUV) : V3f3DArray(_size, _useXYasUV) {
    rawVec.resize(GetVertCount() * 3);
    Reset();
}

Kami::LinearUtil::EigenVecX3f3DArray::EigenVecX3f3DArray(const EigenVecX3f3DArray& src) : V3f3DArray(src.GetSize(), src.GetUVCoordOrNot()) {
    rawVec = src.GetEigenVecXf();
}

void Kami::LinearUtil::EigenVecX3f3DArray::SetEntryToArray(glm::vec3 entry, uint32_t x, uint32_t y, uint32_t z) {
    uint32_t index = GetIndex(x, y, z);

    rawVec[index * 3] = entry.x;
    rawVec[index * 3 + 1] = entry.y;
    rawVec[index * 3 + 2] = entry.z;
}
void Kami::LinearUtil::EigenVecX3f3DArray::AddEntryToArray(glm::vec3 entry, uint32_t x, uint32_t y, uint32_t z) {
    uint32_t index = GetIndex(x, y, z);

    rawVec[index * 3] += entry.x;
    rawVec[index * 3 + 1] += entry.y;
    rawVec[index * 3 + 2] += entry.z;
}

void Kami::LinearUtil::EigenVecX3f3DArray::Reset(float val) {
    if (val == 0)
        rawVec.setZero();
    else
        rawVec.setConstant(val);
}

void Kami::LinearUtil::EigenVecX3f3DArray::Resize(glm::uvec3 _size, float emptyVal) {
    auto newVecSize = _size.x * _size.y * _size.z * 3;

    Eigen::VectorXf newVec(newVecSize);

    if (emptyVal != 0)
        newVec.setConstant(newVecSize);
    else
        newVec.setZero();

    uint32_t copysizeX = std::min(size.x, _size.x);
    uint32_t copysizeY = std::min(size.y, _size.y);
    uint32_t copysizeZ = std::min(size.z, _size.z);

    for (uint32_t i = 0; i < copysizeX; ++i) {
        for (uint32_t j = 0; j < copysizeY; ++j) {
            uint32_t newStart;

            if (useXYasUV) {
                newStart = _size.z * _size.x * j + _size.z * i;
            } else {
                newStart = _size.z * _size.y * i + _size.z * j;
            }
            newStart *= 3;

            auto start = GetIndex(i, j, 0) * 3;
            auto end = GetIndex(i, j, copysizeZ) * 3;
            std::copy(rawVec.data() + start, rawVec.data() + end, newVec.data() + newStart);
        }
    }

    size = _size;
    rawVec = newVec;
}

glm::vec3 Kami::LinearUtil::EigenVecX3f3DArray::GetEntryVal(uint32_t x, uint32_t y, uint32_t z) const {
    auto index = GetIndex(x, y, z) * 3;
    return glm::vec3(rawVec[index], rawVec[index + 1], rawVec[index + 2]);
}

glm::vec3& Kami::LinearUtil::EigenVecX3f3DArray::GetEntryRef(uint32_t x, uint32_t y, uint32_t z) {
    Kami::Notification::Warn(__func__, "This function is undifined in EigenVecX3Df3DArray. Please use \033[31m GetEntryRefA() \033[0m.");
    return errorVec;
}

void* Kami::LinearUtil::EigenVecX3f3DArray::GetRawArray() { return (void*)rawVec.data(); }
void* Kami::LinearUtil::EigenVecX3f3DArray::GetRawArray() const { return (void*)rawVec.data(); }

Eigen::VectorXf& Kami::LinearUtil::EigenVecX3f3DArray::GetEigenVecXf() { return rawVec; }

const Eigen::VectorXf& Kami::LinearUtil::EigenVecX3f3DArray::GetEigenVecXf() const { return rawVec; }

Kami::LinearUtil::EigenVecX3f3DArray& Kami::LinearUtil::EigenVecX3f3DArray::operator=(EigenVecX3f3DArray& src) {
    new (this) EigenVecX3f3DArray(src);
    return *this;
}

// void Kami::LinearUtil::SolveLinearEquation() {}

glm::vec3 Kami::LinearUtil::GetTranslateFrom4x4Mat(const glm::mat4 mat) { return glm::vec3(mat[0][3], mat[1][3], mat[2][3]); }

// ########################################
// #############Eigen3x3BlockMat###########
// ########################################

Kami::LinearUtil::Eigen3x3BlockMat::Eigen3x3BlockMat(uint32_t row, uint32_t column) {
    eigenSparseMat = make_unique<Eigen::SparseMatrix<float>>(row * 3, column * 3);
    entryQueue = make_unique<vector<Triplet<float>>>();
    size = glm::uvec2(row, column);
}

void Kami::LinearUtil::Eigen3x3BlockMat::PushEntry(glm::mat3 mat, uint32_t row, uint32_t column) {
    IndexValidation(row, column);
    uint32_t r = row * 3, c = column * 3;
    for (uint32_t i = 0; i < 3; i++) {
        for (uint32_t j = 0; j < 3; j++) {
            entryQueue->push_back(Triplet<float>(r + i, c + j, mat[i][j]));
        }
    }
}

void Kami::LinearUtil::Eigen3x3BlockMat::ApplyEntry() { eigenSparseMat->setFromTriplets(entryQueue->begin(), entryQueue->end()); }

glm::mat3 Kami::LinearUtil::Eigen3x3BlockMat::GetEntry(uint32_t row, uint32_t column) {
    glm::mat3 ret;
    IndexValidation(row, column);
    uint32_t r = row * 3, c = column * 3;
    for (uint32_t i = 0; i < 3; i++) {
        for (uint32_t j = 0; j < 3; j++) {
            ret[i][j] = eigenSparseMat->coeff(r + i, c + j);
        }
    }
    return ret;
}

void Kami::LinearUtil::Eigen3x3BlockMat::Reset() { eigenSparseMat->setZero(); }

unq_ptr<Eigen::SparseMatrix<float>>& Kami::LinearUtil::Eigen3x3BlockMat::GetEigenMatrix() { return eigenSparseMat; }
