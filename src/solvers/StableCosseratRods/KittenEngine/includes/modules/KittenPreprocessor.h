#pragma once
// Jerry Hsu, 2021

#include <map>
#include <vector>
#include <string>

#include "../../../../../../extern/glm/glm/common.hpp"

namespace Kitten {
using std::map;
using std::string;
using std::vector;
using namespace glm;

extern vector<string> includePaths;
typedef map<string, ivec4> Tags;
void parseAssetTag(string& ori, string& name, Tags& tags);
bool endsWith(string const& str, string const& ending);
string loadText(string name);
string loadTextWithIncludes(string path);
void printWithLineNumber(string str);
}  // namespace Kitten