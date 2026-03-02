#ifndef USCHAIRSALON_IMPORTER_H_
#define USCHAIRSALON_IMPORTER_H_

#include "../Kami.h"
#include "../Hair.h"

struct Hair;

// Please include the extension.
void ImportUSCHairSalonToHair(string filepath, wk_ptr<Hair> hair, uint32_t maxStrandCount = 10000u, uint32_t maxLength = 100u);

#endif /* USCHAIRSALON_IMPORTER_H_ */