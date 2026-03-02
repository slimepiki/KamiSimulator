import mitsuba as mi
import sys
import shutil
import os
import csv

args = sys.argv

mi.set_variant('cuda_ad_rgb')

#for script test
#relative_path = ""

#from bin
relative_path = "../scripts/mitsuba/"

with open(args[4], encoding='utf-8', newline='') as f:
    lst = list(csv.reader(f, delimiter=' '))
    lst = [[float(x) for x in y] for y in lst]

o = mi.ScalarPoint3f(lst[0])
t = mi.ScalarPoint3f(lst[1])
u = mi.ScalarPoint3f(lst[2])

mi.xml.dict_to_xml(
	{
		"type": "scene",
		"mysensor": {
        	"type": "perspective",
        	"to_world": mi.ScalarTransform4f().look_at(origin=o, target=t, up=u),
        	"myfilm": {
            	"type": "hdrfilm",
            	"rfilter": {
        	        "type": "tent"
        	    },
        	    "width": 960,
        	    "height": 540,
                "file_format": "exr",
                "pixel_format": "rgb"
        	}, "mysampler": {
        	    "type": "independent",
        	    "sample_count": 256,
    	    },
	    },"hair": {
    		"type": "linearcurve",
    		"filename": args[2],
    		"hair_bsdf": {
        	"type": "hair",
    		}
		}
		,"body": {
    		"type": "obj",
    		"filename": args[3],
    		"body_bsdf": {
        		"type": "diffuse",
				"reflectance" : {
					"type" : "spectrum",
					"filename" : relative_path + "HumanEpidermis.spd"
				}
			}
		}
	}
, relative_path + "model.xml")

scene = mi.load_file(relative_path + "photo_studio.xml")

image = mi.render(scene)

mi.util.write_bitmap(args[1], image)

os.remove(relative_path + "model.xml")
shutil.rmtree(relative_path + "meshes")