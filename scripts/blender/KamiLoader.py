import bpy # type: ignore
import sys
import os
from pathlib import Path

script_dir_path = os.path.dirname(bpy.context.space_data.text.filepath)
sys.path.append(script_dir_path)

Path_obj_dir = Path(script_dir_path)

import KamiLoaderModules as Kamimod

Kamimod.cleanUp()
Kamimod.setScene()

hair_path = f"{Path_obj_dir.parent.parent}/resources/hairstyles/strands00001.data"
head_path = f"{Path_obj_dir.parent.parent}/resources/hairstyles/head_model.obj"

Kamimod.loadHair(hair_path)
Kamimod.loadHead(head_path)