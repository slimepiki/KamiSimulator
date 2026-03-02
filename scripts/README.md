# Description

This scripts are used to rendering.

## LICENSE

USCLoader.py is based on [this script](https://blender.stackexchange.com/questions/262567/how-do-i-add-hair-3d-segments-coord-from-file-to-a-head/263245#263245).

Epidermis_spectra_genelator.ods is based on [This paper](https://dl.acm.org/doi/10.5555/2383894.2383946).

```
@inproceedings{10.5555/2383894.2383946,
	author = {Donner, Craig and Jensen, Henrik Wann},
	title = {A spectral BSSRDF for shading human skin},
	year = {2006},
	isbn = {3905673355},
	publisher = {Eurographics Association},
	address = {Goslar, DEU},
	booktitle = {Proceedings of the 17th Eurographics Conference on Rendering Techniques},
	pages = {409–417},
	numpages = {9},
	location = {Nicosia, Cyprus},
	series = {EGSR '06}
}
```

## USAGE

### blender

Please open Blender with an empty scene, load KamiLoader.py from the script tab, and run with any .data and .ojb path.
You also have to give some volume to the line segments before rendering if you want to render.

### mitsuba

### Epidermis_spectra_genelator.ods

This table calculates human epidermis absorption spectra.
Please set parameters, copy the cells from J2 to K14, and paste them into any file.