#ifndef UNION_VEC_H_
#define UNION_VEC_H_

#include "../Kami.h"

struct UnionVec {
    enum vecType { BOOL, INT, FLOAT, STRING, NONE };

    vecType vType = NONE;
    void Set(const vector<bool>& vec);
    void Set(const vector<int>& vec);
    void Set(const vector<float>& vec);
    void Set(const vector<string>& vec);
    bool GetBool(uint32_t index);
    int GetInt(uint32_t index);
    float GetFloat(uint32_t index);
    string GetString(uint32_t index);

    vecType GetType();
    uint32_t GetSize();

   private:
    // ∈{"bool", "int", "float", "string"}
    string type;

    vector<bool> boolVec;
    vector<int> intVec;
    vector<float> floatVec;
    vector<string> stringVec;

    string GetTypeStr();
};

#endif /* UNION_VEC_H_ */