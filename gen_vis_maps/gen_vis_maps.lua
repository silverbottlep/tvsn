require 'paths'
require 'gnuplot'
require 'image'
require 'paths'
mattorch = require('fb.mattorch')

opt = lapp[[
  --image_dir			(default '../ObjRenderer/test/')
  --category			(default 'car')
  --output_dir		(default '../tvsn/data/')
]]

--debugger = require('fb.debugger')
--debugger.enter()

list = paths.dir(opt.image_dir)
table.sort(list)
table.remove(list,1)
table.remove(list,1)
n_models = #list

out = {}
map_indices = {}
for i=1,#list do
	map_indices[list[i]] = i
end

maps = torch.ByteTensor(n_models,3,18,17,64,64)
for i=1,#list do
	print(string.format('processing %s',list[i]))
	for phi=0,20,10 do
		fname = paths.concat(opt.image_dir,list[i],string.format('model_views/maps_%d.mat',phi))
		mat = mattorch.load(fname)
		for in_theta=1,18 do
			for trans=1,17 do
				m = mat.out_map[{{},{},{in_theta},{trans}}]:squeeze()
				--m = image.scale(m,128,128)
				m = m:gt(255*0.3)
				maps[i][(phi/10)+1][in_theta][trans]:copy(m)
			end
		end
	end
end

out.map_indices = map_indices
out.maps = maps

--save to the file
out_file = paths.concat(opt.output_dir,'./maps_' .. opt.category .. '.t7')
torch.save(out_file,out)
