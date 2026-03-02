#include "../includes/modules/KittenAssets.h"
#include "../includes/modules/KittenPreprocessor.h"
#include "../includes/modules/Bound.h"
#include "../includes/modules/Mesh.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../../third_party/stb/stb_image.h"

#include <ratio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <regex>

namespace Kitten {
map<string, void*> resources;

void loadDirectory(path root) {
    if (root.is_relative())
        for (auto& p : recursive_directory_iterator(root)) {
            path f(p);
            if (is_regular_file(f)) {
                loadAsset(f);
                std::cout << "Found file: " << f << std::endl;
            }
        }
    else
        cout << "err: resource path must be relative" << endl;
}

void loadAsset(path path) {
    string ext = path.extension().string();
    if (ext == ".obj" || ext == ".fbx" || ext == ".ply")
        loadMesh(path);
    else if (ext == ".node" || ext == ".face" || ext == ".ele")
        loadTetgenMesh(path);
    else if (ext == ".glsl" || ext == ".include" || ext == ".mtl") {
    } else if (ext == ".csv" || ext == ".txt" || ext == ".cfg" || ext == ".json") {
        cout << "asset: loading text " << path.string().c_str() << endl;
        resources[path.filename().string()] = new string(loadText(path.string()));
    } else
        cout << "err: unknown asset type " << path << endl;

    cout << "register: " << path << endl;
}

void loadMesh(path path) {
    cout << "asset: loading model " << path.string().c_str() << endl;
    loadMeshFrom(path);
}

void loadTetgenMesh(path path) {
    // cout << "asset: loading tetgen " << path.string().c_str() << endl;
    loadTetgenFrom(path);
}

// Returns all the caches in a directory
void getCaches(path p, vector<path>& paths) {
    auto name = p.filename().string();
    // Iterate over all files in the directory
    for (const auto& entry : std::filesystem::directory_iterator(p.parent_path())) {
        if (entry.is_regular_file()) {
            auto ep = entry.path();
            auto cname = ep.filename().string();

            // Check for .tmp extension and matching name
            if (ep.has_extension() && ep.extension() == ".tmp" && cname.compare(0, name.length(), name) == 0) {
                // Check for '_'
                if (cname.length() > name.length() && cname[name.length()] == '_') {
                    // printf("Found %s\n", ep.string().c_str());
                    paths.push_back(ep);
                }
            }
        }
    }
}

std::size_t getLastModifiedEpoch(const std::filesystem::path& path) {
    auto ftime = std::filesystem::last_write_time(path);
    auto sctp = std::chrono::time_point_cast<std::chrono::seconds>(ftime);
    auto epoch = sctp.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::seconds>(epoch).count();
}

std::string size_t2Hex(std::size_t num) {
    std::stringstream stream;
    stream << std::hex << num;
    return stream.str();
}

std::filesystem::path getCache(std::string key, size_t hash, int numCache) {
    auto targetPath = path(key + "_" + size_t2Hex(hash) + ".tmp");
    auto target = targetPath.filename().string();

    // Search if the number of hash
    vector<path> paths;
    getCaches(targetPath.has_parent_path() ? key : "./" + key, paths);

    // Check if we have a matching hash
    for (auto& p : paths)
        if (p.filename().string() == target) {
            // Touch file
            auto now = std::filesystem::file_time_type::clock::now();
            std::filesystem::last_write_time(p, now);

            return p;
        }

    // Cache not found, delete the oldest one if we have too many
    if (paths.size() && paths.size() >= (uint32_t)numCache) {
        // Find the oldest one and delete it
        path oldest = paths[0];
        std::size_t oldestTime = std::numeric_limits<size_t>::max();

        for (auto& p : paths) {
            auto time = getLastModifiedEpoch(p);
            if (time < oldestTime) {
                oldestTime = time;
                oldest = p;
            }
        }

        std::filesystem::remove(oldest);
    }

    return target;
}
}  // namespace Kitten