require 'nngraph'

local DCGAN = {}

function DCGAN.create(opts)
	netG = DCGAN.create_netG(opts)
	netD = DCGAN.create_netD(opts)
	return netG, netD
end

function DCGAN.create_netG(opts)
	local inputs = {}
	local input_im = nn.Identity()()
	local input_im_feat = nn.Identity()()
	local input_view = nn.Identity()()
	local mean = nn.Identity()()
	table.insert(inputs,input_im)
	table.insert(inputs,input_im_feat)
	table.insert(inputs,input_view)
	table.insert(inputs,mean)

  -- 3 x 256 x 256 
	local en_conv1 = nn.LeakyReLU(0.2, true)(cudnn.SpatialBatchNormalization(16)(cudnn.SpatialConvolution(3,16,4,4,2,2,1,1)(input_im)))
  -- 16 x 128 x 128
	local en_conv2 = nn.LeakyReLU(0.2, true)(cudnn.SpatialBatchNormalization(32)(cudnn.SpatialConvolution(16,32,4,4,2,2,1,1)(en_conv1)))
  -- 32 x 64 x 64
	local en_conv3 = nn.LeakyReLU(0.2, true)(cudnn.SpatialBatchNormalization(64)((cudnn.SpatialConvolution(32,64,4,4,2,2,1,1)(en_conv2))))
  -- 64 x 32 x 32
	local en_conv4 = nn.LeakyReLU(0.2, true)(cudnn.SpatialBatchNormalization(128)((cudnn.SpatialConvolution(64,128,4,4,2,2,1,1)(en_conv3))))
  -- 128 x 16 x 16 
	local en_conv5 = nn.LeakyReLU(0.2, true)(cudnn.SpatialBatchNormalization(256)((cudnn.SpatialConvolution(128,256,4,4,2,2,1,1)(en_conv4))))
  -- 256 x 8 x 8
	local en_conv6 = nn.LeakyReLU(0.2, true)(cudnn.SpatialBatchNormalization(512)((cudnn.SpatialConvolution(256,512,4,4,2,2,1,1)(en_conv5))))
  -- 512 x 4 x 4
	
	-- view
	local view_fc1 = cudnn.ReLU()(nn.Linear(17,128)(input_view))
	local view_fc2 = nn.Reshape(opts.batchSize, 128, 1, 1)(cudnn.ReLU()(nn.Linear(128,128)(view_fc1)))
	local view_conv = cudnn.SpatialFullConvolution(128,128,4,4)(view_fc2)
	local view_conv = cudnn.ReLU()(cudnn.SpatialBatchNormalization(128)(view_conv))
	local concat1 = nn.JoinTable(2)({en_conv6,input_im_feat,view_conv})

	-- code
	-- (512+128) x 4 x 4
	local concat2 = cudnn.SpatialFullConvolution(512+512+128,512,3,3,1,1,1,1)(concat1)
	local concat2 = cudnn.ReLU()(cudnn.SpatialBatchNormalization(512)(concat2))
	-- (512+128) x 4 x 4
	local concat3 = cudnn.SpatialFullConvolution(512,512,3,3,1,1,1,1)(concat2)
	local concat3 = cudnn.ReLU()(cudnn.SpatialBatchNormalization(512)(concat3))
	
	-- decoder
	-- 512 x 4 x 4
	local de_deconv1 = cudnn.SpatialFullConvolution(512,256,4,4,2,2,1,1)(concat3)
	local de_deconv1 = cudnn.ReLU()(cudnn.SpatialBatchNormalization(256)(de_deconv1))
	local de_deconv1 = cudnn.SpatialConvolution(256,256,3,3,1,1,1,1)(de_deconv1)
	local de_deconv1 = cudnn.ReLU()(cudnn.SpatialBatchNormalization(256)(de_deconv1))
	-- 256 x 8 x 8
	local de_deconv2 = cudnn.SpatialFullConvolution(256,128,4,4,2,2,1,1)(de_deconv1)
	local de_deconv2 = cudnn.ReLU()(cudnn.SpatialBatchNormalization(128)(de_deconv2))
	local de_deconv2 = cudnn.SpatialConvolution(128,128,3,3,1,1,1,1)(de_deconv2)
	local de_deconv2 = cudnn.ReLU()(cudnn.SpatialBatchNormalization(128)(de_deconv2))
	-- 128 x 16 x16 
	local de_deconv3 = cudnn.SpatialFullConvolution(128,64,4,4,2,2,1,1)(de_deconv2)
	local de_deconv3 = cudnn.ReLU()(cudnn.SpatialBatchNormalization(64)(de_deconv3))
	local de_skip3 = nn.JoinTable(2)({de_deconv3,en_conv3})
	local de_skip3 = cudnn.SpatialConvolution(64+64,64,3,3,1,1,1,1)(de_skip3)
	local de_skip3 = cudnn.ReLU()(cudnn.SpatialBatchNormalization(64)(de_skip3))
	-- 64 x 32 x 32 
	local de_deconv4 = cudnn.SpatialFullConvolution(64,32,4,4,2,2,1,1)(de_skip3)
	local de_deconv4 = cudnn.ReLU()(cudnn.SpatialBatchNormalization(32)(de_deconv4))
	local de_skip4 = nn.JoinTable(2)({de_deconv4,en_conv2})
	local de_skip4 = cudnn.SpatialConvolution(32+32,32,3,3,1,1,1,1)(de_skip4)
	local de_skip4 = cudnn.ReLU()(cudnn.SpatialBatchNormalization(32)(de_skip4))
	-- 32 x 64 x 64
	local de_deconv5 = cudnn.SpatialFullConvolution(32,16,4,4,2,2,1,1)(de_skip4)
	local de_deconv5 = cudnn.ReLU()(cudnn.SpatialBatchNormalization(16)(de_deconv5))
	local de_skip5 = nn.JoinTable(2)({de_deconv5,en_conv1})
	local de_skip5 = cudnn.SpatialConvolution(16+16,16,3,3,1,1,1,1)(de_skip5)
	local de_skip5 = cudnn.ReLU()(cudnn.SpatialBatchNormalization(16)(de_skip5))
	-- 16 x 128 x 128 
	local de_deconv6 = cudnn.SpatialFullConvolution(16,16,4,4,2,2,1,1)(de_skip5)
	local de_deconv6 = cudnn.ReLU()(cudnn.SpatialBatchNormalization(16)(de_deconv6))
	local de_skip6 = nn.JoinTable(2)({de_deconv6,input_im})
	local de_skip6 = cudnn.SpatialConvolution(16+3,16,3,3,1,1,1,1)(de_skip6)
	local de_skip6 = cudnn.ReLU()(cudnn.SpatialBatchNormalization(16)(de_skip6))
	local tanh_out = nn.Tanh()(cudnn.SpatialConvolution(16,3,3,3,1,1,1,1)(de_skip6)):annotate{name='tanh_out'}
	-- 3 x 256 x 256

	local im = nn.MulConstant(127.5,false)(nn.AddConstant(1,false)(tanh_out))
	local trans_im = nn.CSubTable()({im,mean})

	local outputs = {}
	table.insert(outputs,trans_im)

	return nn.gModule(inputs, outputs)
end

function DCGAN.create_netD(opts)
	local input_im = nn.Identity()()

  -- 3 x 256 x 256 
	local en_conv1 = nn.LeakyReLU(0.2, true)(cudnn.SpatialBatchNormalization(64)(cudnn.SpatialConvolution(3,64,4,4,2,2,1,1)(input_im))):annotate{name='feat1'}
  -- 16 x 128 x 128
	local en_conv2 = nn.LeakyReLU(0.2, true)(cudnn.SpatialBatchNormalization(64)(cudnn.SpatialConvolution(64,64,4,4,2,2,1,1)(en_conv1))):annotate{name='feat2'}
  -- 32 x 64 x 64
	local en_conv3 = nn.LeakyReLU(0.2, true)(cudnn.SpatialBatchNormalization(64)((cudnn.SpatialConvolution(64,64,4,4,2,2,1,1)(en_conv2)))):annotate{name='feat3'}
  -- 64 x 32 x 32
	local en_conv4 = nn.LeakyReLU(0.2, true)(cudnn.SpatialBatchNormalization(128)((cudnn.SpatialConvolution(64,128,4,4,2,2,1,1)(en_conv3)))):annotate{name='feat4'}
  -- 128 x 16 x 16 
	local en_conv5 = nn.LeakyReLU(0.2, true)(cudnn.SpatialBatchNormalization(256)((cudnn.SpatialConvolution(128,256,4,4,2,2,1,1)(en_conv4)))):annotate{name='feat5'}
  -- 256 x 8 x 8
	local en_conv6 = nn.LeakyReLU(0.2, true)(cudnn.SpatialBatchNormalization(512)((cudnn.SpatialConvolution(256,512,4,4,2,2,1,1)(en_conv5)))):annotate{name='feat6'}
  -- 512 x 4 x 4
	local out = nn.View(1)(nn.Sigmoid()(cudnn.SpatialConvolution(512,1,4,4)((en_conv6))))

	local inputs = {}
	local outputs = {}
	table.insert(inputs,input_im)
	table.insert(outputs,en_conv1)
	table.insert(outputs,en_conv2)
	table.insert(outputs,en_conv3)
	table.insert(outputs,out)

	return nn.gModule(inputs, outputs)
end
return DCGAN 
