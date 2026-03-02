#ifndef CAMERA_H_
#define CAMERA_H_

#include "../extern/glm/glm/common.hpp"

namespace Camera {
struct Setting {
   public:
    glm::vec3 origin;
    glm::vec3 lookAt;
    glm::vec3 up;
    constexpr Setting(glm::vec3 _origin, glm::vec3 _lookAt, glm::vec3 _up) : origin(_origin), lookAt(_lookAt), up(_up) {};
};

constexpr Setting HEAD_CAMERA_SETTING(glm::vec3(0.f, 1.6f, 2.2f), glm::vec3(0.f, 1.6f, 0.f), glm::vec3(0, 1, 0));
constexpr Setting BOX_CAMERA_SETTING(glm::vec3(0.12f, .14f, .5f), glm::vec3(0.12f, 0.f, 0.f), glm::vec3(0, 1, 0));
constexpr Setting FULLBODY_CAMERA_SETTING(glm::vec3(0.f, 1.6f, 2.2f), glm::vec3(0.f, 1.6f, 0.f), glm::vec3(0, 1, 0));
}  // namespace Camera

#endif /* CAMERA_H_ */