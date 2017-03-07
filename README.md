# Transformation-Grounded Image Generation Network for Novel 3D View Synthesis
[Eunbyung Park](http://www.cs.unc.edu/~eunbyung/), [Jimei Yang](https://eng.ucmerced.edu/people/jyang44/), [Ersin Yumer](http://www.meyumer.com/), [Duygu Ceylan](http://www.duygu-ceylan.com/), and [Alexander C. Berg](http://acberg.com/), CVPR 2017

[[Paper]](http://www.cs.unc.edu/~eunbyung/tvsn/) [[Project Homepage]](http://www.cs.unc.edu/~eunbyung/tvsn/)

Follow below instructions to run and test the codes used in the paper

## 0. Shapenet dataset download
You should have ShapeNetCore.v1 dataset in your local $(SHAPENET_DATA) directory via [shapenet.org](https://shapenet.org/) in your local directory. Once you downloaded, let's make softlinks to them.
```bash
$(tvsn_root)/tvsn/data$>ln -s $(SHAPENET_DATA)/02958343 ./car
$(tvsn_root)/tvsn/data$>ln -s $(SHAPENET_DATA)/03001627 ./chair
```

## 1. Dataset Preparation (Rendering multiple view images)
I adopted [rendering engine](https://github.com/sunweilun/ObjRenderer) used in the [appearance flow network](https://github.com/tinghuiz/appearance-flow), and modified original code a little bit to get the surface normals and object coordinates, which will be used for generating visibility maps. You can download from [here](https://github.com/silverbottlep/ObjRenderer) and edit the 'config.txt' file to tune the engine for your purpose. For example,
```bash
$(tvsn_root)$> git clone git@github.com:silverbottlep/ObjRenderer.git
$(tvsn_root)$> cd ObjRenderer
$(tvsn_root)/ObjRenderer$> cat config.txt
folder_path = $(SHAPENET_DATA)/02958343 "e.g. 'car' category"
envmap_path = envmaps/envmap2.hdr
theta_inc = 20
phi_inc = 10
phi_max = 20
output_coord = 1
output_norm = 1
render_size = 1024
output_size = 256
reverse_normals = 1
brightness = 0.7
```
Build and execute. It will take long time, and requires a lot of space(~10GB)
```bash
$(tvsn_root)/ObjRenderer$> make
$(tvsn_root)/ObjRenderer$> ./dist/Release/GNU-Linux-x86/objrenderer
```
It will generate multiple view images, 3D object coordinates, and normals under 'model_views' directory for each 3D models.
```bash
$(tvsn_root)/ObjRenderer$> ls ($SHAPENET_DATA)/02958343/56c80ed5da821cd0179005454847728d/model_views
0_0_coord.exr   
0_0_norm.exr    
0_0.png         
0_10_coord.exr  
0_10_norm.exr   
0_10.png        
0_20_coord.exr  
0_20_norm.exr   
0_20.png        
100_0_coord.exr 
100_0_norm.exr  
100_0.png       
100_10_coord.exr
100_10_norm.exr 
100_10.png      
100_20_coord.exr
100_20_norm.exr 
.....
```

## 2. Dataset Preparation (Generating visibility maps)
Now, we are going to make visibility maps. First we need to have simple library for reading .exr file. You can download and install from [here](http://www.mit.edu/~kimo/software/matlabexr/). You also need to have matlab and torch installed.
```bash
$(tvsn_root)$> cd gen_vis_maps
$(tvsn_root)/gen_vis_maps$> wget http://www.mit.edu/~kimo/software/matlabexr/MatlabEXR.zip
$(tvsn_root)/gen_vis_maps$> unzip MatlabEXR.zip
$(tvsn_root)/gen_vis_maps$> cd MatlabEXR
$(tvsn_root)/gen_vis_maps/MatlabEXR$> mex exrinfo.cpp -lIlmImf -lIex -lImath -lHalf -I/usr/include/OpenEXR/
$(tvsn_root)/gen_vis_maps/MatlabEXR$> mex exrread.cpp -lIlmImf -lIex -lImath -lHalf -I/usr/include/OpenEXR/
$(tvsn_root)/gen_vis_maps/MatlabEXR$> mex exrwrite.cpp -lIlmImf -lIex -lImath -lHalf -I/usr/include/OpenEXR/
```
Once you have exrread library, you can run the script we provided, it will save visibility maps in 'tvsn/data' directory, e.g. 'tvsn/data/maps_car.t7'
```bash
$(tvsn_root)/gen_vis_maps$>./gen_vis_maps.sh $(SHAPENET_DATA)/02958343 car
$(tvsn_root)/gen_vis_maps$>./gen_vis_maps.sh $(SHAPENET_DATA)/03001627 chair
$(tvsn_root)/gen_vis_maps$>./gen_vis_maps.sh $(SHAPENET_DATA)/03790512 motorcycle
$(tvsn_root)/gen_vis_maps$>./gen_vis_maps.sh $(SHAPENET_DATA)/03991062 flowerpot
$(tvsn_root)/gen_vis_maps$>ls ../tvsn/data/
maps_car.t7  maps_chair.t7 maps_motorcycle.t7 maps_flowerpot.t7
```
Rendering images and computing visibility maps are time consuming jobs, so I highly recommend you to parallelize it across multiple cpus. It can be easily done by modifying the code or splitting the data into different directories.

## 3. Training DOAFN(Disocclusion aware appearance flow network)
```bash
$(tvsn_root)/tvsn/code$>./script_train_doafn.sh
```

## 4. Training TVSN(Transformation-grounded view synthesis network)
First, we need to prepare pretrained vgg16 network. We imported [caffemodel](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md) and translated into torch nngraph format. You can download translated version with provided script.
```bash
$(tvsn_root)/tvsn/code/lossnet$>./download_lossnet.sh
```
Now, you can train!
```bash
$(tvsn_root)/tvsn/code$>./script_train_tvsn.sh
```
