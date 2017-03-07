#!/bin/bash
#[examples]
# ./gen_vis_maps.sh $(SHAPENET_DATA)/02958343 car
# ./gen_vis_maps.sh $(SHAPENET_DATA)/new_chair chair
# ./gen_vis_maps.sh $(SHAPENET_DATA)/03790512 motorcycle
# ./gen_vis_maps.sh $(SHAPENET_DATA)/03991062 flowerpot

matlab -nodesktop -nosplash -r "gen_vis_maps('$1');exit"
th gen_vis_maps.lua --image_dir $1 --category $2
