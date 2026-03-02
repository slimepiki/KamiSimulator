#ifndef MOTION_H_
#define MOTION_H_

#include "Kami.h"

class Motion {
   private:
    // fps
    // Motion

    struct MotionParams {
        string fileName;
    };

   public:
    Motion(string filename);
    // frame GetCurrentFrame(float time);
};

#endif /* MOTION_H_ */