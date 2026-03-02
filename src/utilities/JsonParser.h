#ifndef JSON_PARSER_H_
#define JSON_PARSER_H_

#include "../Kami.h"
#include "../../extern/json.hpp"

#include "Notification.h"

class JsonParser {
    using json = nlohmann::json;

   private:
    json j;

   public:
    // Please include the extension in filePath.
    JsonParser(string jsonPath);
    JsonParser() {};
    // Please include the extension in filePath.
    void OpenJsonFile(string jsonPath);
    bool CheckKey(string key);

    // Please include the extension in filePath.
    bool SaveJson(string filePath, bool isSilent = false);

    // get JSON as string
    string dump();

    bool GetBool(string key);
    int GetInteger(string key);
    float GetFloat(string key);
    string GetString(string key);
    vector<bool> GetBoolVec(string key);
    vector<int> GetIntVec(string key);
    vector<float> GetFloatVec(string key);
    vector<string> GetStringVec(string key);

    // for test
    // void PrintItem(string key);

    // Only bool, integer, float, string, vector<bool>, vector<int>, vector<float>, and vector<string> are permitted.
    template <typename T>
    void SetVal(string key, T val, bool isSilent = false) {
        if (j.contains(key) && !isSilent) {
            Kami::Notification::Notify("Json:  \"" + key + "\" is going to be overwritten.");
        }
        ResetVal(key, val);
    };

    // Only bool, integer, float, string, vector<bool>, vector<int>, vector<float>, and vector<string> are permitted.
    // This function doesn't generate any notification. Using SetVal() instead of ResetVal() is recommended.
    void ResetVal(string key, bool val);
    void ResetVal(string key, int val);
    void ResetVal(string key, float val);
    void ResetVal(string key, string val);
    void ResetVal(string key, const char* val);
    void ResetVal(string key, vector<int> val);
    void ResetVal(string key, vector<float> val);
    void ResetVal(string key, vector<string> val);
};

#endif /* JSON_PARSER_H_ */
