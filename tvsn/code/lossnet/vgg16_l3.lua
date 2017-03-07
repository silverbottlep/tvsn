require 'nngraph'

local model = {}

function model.create()
	local inputs = {}
	local outputs = {}
	local im = nn.Identity()()
	table.insert(inputs,im)

	-- convolution layers
	local conv1_1 = cudnn.ReLU()(cudnn.SpatialConvolution(3,64,3,3,1,1,1,1)(im):annotate{name='conv1_1',bias=true})
	local conv1_2 = cudnn.ReLU()(cudnn.SpatialConvolution(64,64,3,3,1,1,1,1)(conv1_1):annotate{name='conv1_2',bias=true})
	local pool1 = cudnn.SpatialMaxPooling(2,2,2,2,0,0):ceil()(conv1_2)

	local conv2_1 = cudnn.ReLU()(cudnn.SpatialConvolution(64,128,3,3,1,1,1,1)(pool1):annotate{name='conv2_1',bias=true})
	local conv2_2 = cudnn.ReLU()(cudnn.SpatialConvolution(128,128,3,3,1,1,1,1)(conv2_1):annotate{name='conv2_2',bias=true})
	local pool2 = cudnn.SpatialMaxPooling(2,2,2,2,0,0):ceil()(conv2_2)
	
	local conv3_1 = cudnn.ReLU()(cudnn.SpatialConvolution(128,256,3,3,1,1,1,1)(pool2):annotate{name='conv3_1',bias=true})
	local conv3_2 = cudnn.ReLU()(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1)(conv3_1):annotate{name='conv3_2',bias=true})
	local conv3_3 = cudnn.ReLU()(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1)(conv3_2):annotate{name='conv3_3',bias=true})
	local pool3 = cudnn.SpatialMaxPooling(2,2,2,2,0,0):ceil()(conv3_3)
	
	table.insert(outputs,pool1)
	table.insert(outputs,pool2)
	table.insert(outputs,pool3)
	return nn.gModule(inputs, outputs)
end

return model 
