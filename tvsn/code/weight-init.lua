--
-- Different weight initialization methods
--
-- > model = require('weight-init')(model, 'heuristic')
--
require("nn")

w_init = {}

-- "Efficient backprop"
-- Yann Lecun, 1998
function w_init_heuristic(fan_in, fan_out)
   return math.sqrt(1/(3*fan_in))
end


-- "Understanding the difficulty of training deep feedforward neural networks"
-- Xavier Glorot, 2010
function w_init_xavier(fan_in, fan_out)
   return math.sqrt(2/(fan_in + fan_out))
end


-- "Understanding the difficulty of training deep feedforward neural networks"
-- Xavier Glorot, 2010
function w_init_xavier_caffe(fan_in, fan_out)
   return math.sqrt(1/fan_in)
end


-- "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
-- Kaiming He, 2015
function w_init_kaiming(fan_in, fan_out)
   return math.sqrt(4/(fan_in + fan_out))
end


function w_init.nngraph(g, arg)
   -- choose initialization method
   local method = nil
   if     arg == 'heuristic'    then method = w_init_heuristic
   elseif arg == 'xavier'       then method = w_init_xavier
   elseif arg == 'xavier_caffe' then method = w_init_xavier_caffe
   elseif arg == 'kaiming'      then method = w_init_kaiming
   else
      assert(false)
   end
	 
	 for i,node in ipairs(g.forwardnodes) do
		 if node.data.module then
			 -- loop over all convolutional modules
			 local m = node.data.module
			 if m.__typename == 'cudnn.SpatialConvolution' then
				 m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
			 elseif m.__typename == 'cudnn.SpatialFullConvolution' then
				 m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
			 elseif m.__typename == 'cudnn.SpatialBatchNormalization' then
				 if m.weight then m.weight:normal(1.0, 0.02) end
				 if m.bias then m.bias:fill(0) end
			 elseif m.__typename == 'nn.SpatialConvolution' then
				 m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
			 elseif m.__typename == 'nn.SpatialConvolutionMM' then
				 m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
			 elseif m.__typename == 'nn.SpatialDeconvolution' then
				 m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
			 elseif m.__typename == 'nn.LateralConvolution' then
				 m:reset(method(m.nInputPlane*1*1, m.nOutputPlane*1*1))
			 elseif m.__typename == 'nn.VerticalConvolution' then
				 m:reset(method(1*m.kH*m.kW, 1*m.kH*m.kW))
			 elseif m.__typename == 'nn.HorizontalConvolution' then
				 m:reset(method(1*m.kH*m.kW, 1*m.kH*m.kW))
			 elseif m.__typename == 'nn.Linear' then
				 m:reset(method(m.weight:size(2), m.weight:size(1)))
			 elseif m.__typename == 'nn.TemporalConvolution' then
				 m:reset(method(m.weight:size(2), m.weight:size(1)))            
			 end

			 if m.bias then
				 m.bias:zero()
			 end
		 end
	 end
end

function w_init.nn(net, arg)
   -- choose initialization method
   local method = nil
   if     arg == 'heuristic'    then method = w_init_heuristic
   elseif arg == 'xavier'       then method = w_init_xavier
   elseif arg == 'xavier_caffe' then method = w_init_xavier_caffe
   elseif arg == 'kaiming'      then method = w_init_kaiming
   else
      assert(false)
   end

   -- loop over all convolutional modules
   for i = 1, #net.modules do
      local m = net.modules[i]
      if m.__typename == 'cudnn.SpatialConvolution' then
         m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
			elseif m.__typename == 'cudnn.SpatialFullConvolution' then
         m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
			elseif m.__typename == 'cudnn.SpatialBatchNormalization' then
				 if m.weight then m.weight:normal(1.0, 0.02) end
				 if m.bias then m.bias:fill(0) end
			elseif m.__typename == 'nn.SpatialConvolution' then
         m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
      elseif m.__typename == 'nn.SpatialConvolutionMM' then
         m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
			elseif m.__typename == 'nn.SpatialDeconvolution' then
				 m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
      elseif m.__typename == 'nn.LateralConvolution' then
         m:reset(method(m.nInputPlane*1*1, m.nOutputPlane*1*1))
      elseif m.__typename == 'nn.VerticalConvolution' then
         m:reset(method(1*m.kH*m.kW, 1*m.kH*m.kW))
      elseif m.__typename == 'nn.HorizontalConvolution' then
         m:reset(method(1*m.kH*m.kW, 1*m.kH*m.kW))
      elseif m.__typename == 'nn.Linear' then
         m:reset(method(m.weight:size(2), m.weight:size(1)))
      elseif m.__typename == 'nn.TemporalConvolution' then
         m:reset(method(m.weight:size(2), m.weight:size(1)))            
      end

      if m.bias then
         m.bias:zero()
      end
   end
end

function w_init.module(m, arg)
   -- choose initialization method
   local method = nil
   if     arg == 'heuristic'    then method = w_init_heuristic
   elseif arg == 'xavier'       then method = w_init_xavier
   elseif arg == 'xavier_caffe' then method = w_init_xavier_caffe
   elseif arg == 'kaiming'      then method = w_init_kaiming
   else
      assert(false)
   end

	 if m.__typename == 'cudnn.SpatialConvolution' then
		 m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
	 elseif m.__typename == 'cudnn.SpatialFullConvolution' then
			 m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
	 elseif m.__typename == 'nn.SpatialConvolution' then
			 m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
	 elseif m.__typename == 'nn.SpatialConvolutionMM' then
		 m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
	 elseif m.__typename == 'nn.LateralConvolution' then
		 m:reset(method(m.nInputPlane*1*1, m.nOutputPlane*1*1))
	 elseif m.__typename == 'nn.VerticalConvolution' then
		 m:reset(method(1*m.kH*m.kW, 1*m.kH*m.kW))
	 elseif m.__typename == 'nn.HorizontalConvolution' then
		 m:reset(method(1*m.kH*m.kW, 1*m.kH*m.kW))
	 elseif m.__typename == 'nn.Linear' then
		 m:reset(method(m.weight:size(2), m.weight:size(1)))
	 elseif m.__typename == 'nn.TemporalConvolution' then
		 m:reset(method(m.weight:size(2), m.weight:size(1)))            
	 end
	 if m.bias then
		 m.bias:zero()
	 end
end

return w_init
