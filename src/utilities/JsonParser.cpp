#include <fstream>

#include "JsonParser.h"
#include "FileUtil.h"
#include "Notification.h"

JsonParser::JsonParser(string jsonPath) { OpenJsonFile(jsonPath); }

void JsonParser::OpenJsonFile(string jsonPath) {
    if (!Kami::FileUtil::IsFileExist(jsonPath)) {
        Kami::Notification::Caution(__func__, "Invalid path: " + jsonPath);
        return;
    }
    std::fstream ifs(jsonPath);
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    j = json::parse(str);
    ifs.close();
    return;
}

bool JsonParser::CheckKey(string key) { return j.contains(key); }

bool JsonParser::SaveJson(string jsonPath, bool isSilent) {
    unq_ptr<KamiFile> kamiFile(make_unique<KamiFile>(jsonPath, std::ios_base::out | std::ios_base::trunc, isSilent));
    std::string str = j.dump(4);

    kamiFile->WriteStringWithBreak(str);

    if (!isSilent) Kami::Notification::Notify("Json: The JSON object has been saved at " + jsonPath);

    return false;
}

string JsonParser::dump() { return std::move(j.dump(4)); }

bool JsonParser::GetBool(string key) { return j[key].get<bool>(); }
int JsonParser::GetInteger(string key) { return j[key].get<int>(); }
float JsonParser::GetFloat(string key) { return j[key].get<float>(); }
string JsonParser::GetString(string key) { return j[key].get<string>(); }
vector<bool> JsonParser::GetBoolVec(string key) {
    vector<bool> ret;
    for (auto item : j[key]) {
        ret.push_back(item.get<bool>());
    }
    return ret;
}
vector<int> JsonParser::GetIntVec(string key) {
    vector<int> ret;
    for (auto item : j[key]) {
        ret.push_back(item.get<int>());
    }
    return ret;
}
vector<float> JsonParser::GetFloatVec(string key) {
    vector<float> ret;
    for (auto item : j[key]) {
        ret.push_back(item.get<float>());
    }
    return ret;
}

vector<string> JsonParser::GetStringVec(string key) {
    vector<string> ret;
    for (auto item : j[key]) {
        ret.push_back(item.get<string>());
    }
    return ret;
}

// void JsonParser::PrintItem(string key) {
//     if (j.contains(key)) {
//         cout << key << ": " << j[key] << endl;
//     } else {
//         cout << key << " is not found" << endl;
//     }
// }

void JsonParser::ResetVal(string key, bool val) { j[key] = val; }

void JsonParser::ResetVal(string key, int val) { j[key] = val; }

void JsonParser::ResetVal(string key, float val) { j[key] = val; }

void JsonParser::ResetVal(string key, string val) { j[key] = val; }

void JsonParser::ResetVal(string key, const char* val) {
    string str{val};
    j[key] = str;
}

void JsonParser::ResetVal(string key, vector<int> val) { j[key] = val; }
void JsonParser::ResetVal(string key, vector<float> val) { j[key] = val; }
void JsonParser::ResetVal(string key, vector<string> val) { j[key] = val; }