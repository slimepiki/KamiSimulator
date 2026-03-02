#ifndef HAIR_MODIFICATION_H_
#define HAIR_MODIFICATION_H_

#include "../Kami.h"
#include "../Hair.h"

namespace Kami {
namespace HairMods {
// Recreate a hair with "mul" timed elements.
void DivideHair(sh_ptr<Hair> hair, size_t mul);
}  // namespace HairMods
}  // namespace Kami

#endif /* HAIR_MODIFICATION_H_ */