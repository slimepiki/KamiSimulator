#include "Kami.h"
#include "utilities/Notification.h"
#include "utilities/LinearUtil.h"
#include "Hair.h"

int main(int args, char* argv[]) {
    string filepath = "../resources/hairstyles/strands00001.data";
    // string filepath = "test0.data";
    Hair hair(filepath);

    Kami::Notification::PrintHairVertex2D(hair, 9999, 99);
    hair.SaveCurrentSnapShot("test", 0);
    return 0;
}