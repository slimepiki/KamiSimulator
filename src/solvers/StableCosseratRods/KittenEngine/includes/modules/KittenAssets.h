#pragma once
// Jerry Hsu, 2021

#include <filesystem>
#include <map>

#include "../../../../../../extern/glm/glm/common.hpp"
#include "Common.h"

namespace Kitten {
class Texture;
typedef struct UBOMat {
    vec4 col;
    vec4 col1;
    vec4 params0;
    vec4 params1;
} UBOMat;

typedef struct Material {
    UBOMat props;
    Texture* texs[8] = {};
} Material;
}  // namespace Kitten

namespace Kitten {
using namespace std;
using namespace std::filesystem;

extern unsigned int meshImportFlags;
extern map<string, void*> resources;

void loadDirectory(path dir);
void loadAsset(path path);
void loadMesh(path path);
void loadTetgenMesh(path path);

// Grabs a file path for a cache file for a specific string key and hash
// If cache does not exist, keep the last numCache files and delete the oldest one if too many
// Both key and hash are used to prevent collisions. Key should be cache name, hash should be hash of inputs
std::filesystem::path getCache(std::string key, size_t hash, int numCache = 8);

template <typename T>
inline T* get(const char* name) {
    printf("Getting resource: %s\n", name);
    auto itr = resources.find(name);
    if (itr == resources.end()) throw runtime_error("Resource not found!");
    return (T*)itr->second;
}
};  // namespace Kitten
