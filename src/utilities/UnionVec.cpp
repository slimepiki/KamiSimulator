#include "UnionVec.h"
#include "Notification.h"

void UnionVec::Set(const vector<bool>& vec) {
    boolVec = vec;
    vType = BOOL;
}

void UnionVec::Set(const vector<int>& vec) {
    intVec = vec;
    vType = INT;
}

void UnionVec::Set(const vector<float>& vec) {
    floatVec = vec;
    vType = FLOAT;
}

void UnionVec::Set(const vector<string>& vec) {
    stringVec = vec;
    vType = STRING;
}

bool UnionVec::GetBool(uint32_t index) {
    if (vType != BOOL) {
        Kami::Notification::Warn(__func__, "Wrong type! The correct type is " + GetTypeStr());
    }
    if (index >= boolVec.size()) {
        Kami::Notification::Warn(__func__, "Int: Invalid index!");
    }
    return boolVec[index];
}

int UnionVec::GetInt(uint32_t index) {
    if (vType != INT) {
        Kami::Notification::Warn(__func__, "Wrong type! The correct type is " + GetTypeStr());
    }
    if (index >= intVec.size()) {
        Kami::Notification::Warn(__func__, "Int: Invalid index!");
    }
    return intVec[index];
}

float UnionVec::GetFloat(uint32_t index) {
    if (vType != FLOAT) {
        Kami::Notification::Warn(__func__, "Wrong type! The correct type is " + GetTypeStr());
    }
    if (index >= floatVec.size()) {
        Kami::Notification::Warn(__func__, "Int: Invalid index!");
    }
    return floatVec[index];
}

string UnionVec::GetString(uint32_t index) {
    if (vType != STRING) {
        Kami::Notification::Warn(__func__, "Wrong type! The correct type is " + GetTypeStr());
    }
    if (index >= stringVec.size()) {
        Kami::Notification::Warn(__func__, "Int: Invalid index!");
    }
    return stringVec[index];
}

string UnionVec::GetTypeStr() {
    if (vType == BOOL) return "BOOL";
    if (vType == INT) return "INT";
    if (vType == FLOAT) return "FLOAT";
    if (vType == STRING) return "STRING";
    return "NONE";
}

UnionVec::vecType UnionVec::GetType() { return vType; }

uint32_t UnionVec::GetSize() {
    if (vType == BOOL) return boolVec.size();
    if (vType == INT) return intVec.size();
    if (vType == FLOAT) return floatVec.size();
    if (vType == STRING) return stringVec.size();
    return 0;
}
