require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'image'
require 'paths'
require 'nngraph'
require 'gnuplot'
require 'stn'

opt = lapp[[
  --dataset						(default 'tvsn')
	--split							(default 'train')
  --saveFreq          (default 20)
  --modelString       (default 'TVSN')
  -g, --gpu           (default 0)
  --imgscale          (default 256)
  --background				(default 0)
  --nThreads					(default 8)
  --maxEpoch          (default 200)
  --iter_per_epoch		(default 1000)
  --batchSize         (default 15)
  --lossnet						(default 'vgg16')
  --lr								(default 0.0001)
  --beta1							(default 0.5)
  --lambda1						(default 100)
  --lambda2						(default 0.001)
	--iterG							(default 2)
  --loss_layer				(default 3)
	--tv_weight					(default 0.0001)
	--pixel_weight			(default 0)
  --modelDir					(default '../snapshots/')
	--data_dir					(default '../data/')
  --category          (default 'car')
	--resume						(default 0)
	-d, --debug					(default 0)
]]

print(opt)
if opt.debug > 0 then
	debugger = require('fb.debugger')
	debugger.enter()
end

-- initial setup
opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(opt.nThreads)
torch.setdefaulttensortype('torch.FloatTensor')

opt.modelName = string.format('%s_%s_%s_bs%03d', opt.modelString, opt.imgscale, opt.category, opt.batchSize)
opt.doafnName = string.format('DOAFN_SYM_%s_%s_bs025',opt.imgscale, opt.category)
opt.modelPath = opt.modelDir .. opt.modelName
if not paths.dirp(opt.modelPath) then
  paths.mkdir(opt.modelPath)
end
if not paths.dirp(opt.modelPath .. '/training/') then
  paths.mkdir(opt.modelPath .. '/training/')
end

if opt.gpu < 0 or opt.gpu > 3 then opt.gpu = -1 end
if opt.gpu >= 0 then
  cutorch.setDevice(opt.gpu+1)
  print('<gpu> using device ' .. opt.gpu)
end

opt.metadata = string.format('../data/metadata_%s.cache',opt.category)

local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

-- This code is copied from https://github.com/jcjohnson/fast-neural-style  
local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')
function TVLoss:__init(strength)
  parent.__init(self)
  self.strength = strength
  self.x_diff = torch.Tensor()
  self.y_diff = torch.Tensor()
end
function TVLoss:updateOutput(input)
  self.output = input
  return self.output
end
-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  local B, C, H, W = input:size(1), input:size(2), input:size(3), input:size(4)
	for i=1,B do
		self.x_diff:resize(3, H - 1, W - 1)
		self.y_diff:resize(3, H - 1, W - 1)
		self.x_diff:copy(input[i][{{}, {1, -2}, {1, -2}}])
		self.x_diff:add(-1, input[i][{{}, {1, -2}, {2, -1}}])
		self.y_diff:copy(input[i][{{}, {1, -2}, {1, -2}}])
		self.y_diff:add(-1, input[i][{{}, {2, -1}, {1, -2}}])
		self.gradInput[i][{{}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
		self.gradInput[i][{{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
		self.gradInput[i][{{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
		self.gradInput[i]:mul(self.strength)
		--self.gradInput[i]:add(gradOutput[i])
	end
  return self.gradInput
end

print('loading lossnet for perceptual loss...')
lossnet = torch.load(string.format('lossnet/%s_l%d.t7',opt.lossnet,opt.loss_layer))
if opt.tv_weight > 0 then
	tvloss = nn.TVLoss(opt.tv_weight):float()
end

print(string.format('Initializing data loader nthread:%d',opt.nThreads))
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
local data_sizes = data:size()
local ntrain = data_sizes[2]
local ntest = data_sizes[3]
print("Dataset: " .. opt.dataset .. " nTotal: " .. ntrain+ntest .. " nTrain: " .. ntrain .. " nTest: " .. ntest)

print('loading pretrained doafn ...')
local doafn_loader = torch.load(opt.modelDir .. opt.doafnName .. '/net-epoch-200.t7')
local doafn = doafn_loader.net

-- load model from current learning stage
epoch = 0
if opt.resume > 0 then
	for i = opt.maxEpoch, 1, -opt.saveFreq do
		if paths.filep(opt.modelPath .. string.format('/net-epoch-%d.t7', i)) then
			epoch = i
			loader = torch.load(opt.modelPath .. string.format('/net-epoch-%d.t7', i))
			netD = loader.netD
			if opt.resetD > 0 then
				netD:apply(weights_init)
			end
			netG = loader.netG
			loss_listD = loader.loss_listD
			loss_listG = loader.loss_listG
			print(opt.modelPath .. string.format('/net-epoch-%d.t7', i))
			print(string.format('resuming from epoch %d', i))
			break
		end
	end
else
	-- create model
	print('creating model...')
	net_module = dofile('models/' .. string.format('%s_%s.lua', opt.modelString, opt.imgscale))
	netG, netD = net_module.create(opt)
	netG:apply(weights_init)
	netD:apply(weights_init)
	loss_listD = torch.Tensor(1,1):fill(100)
	loss_listG = torch.Tensor(1,1):fill(100)
end
local criterion_l1 = nn.AbsCriterion()
criterion_l1.sizeAverage = true
local criterion_pixel = nn.MSECriterion()
criterion_pixel.sizeAverage = true
local criterionGAN = nn.BCECriterion()

local optimStateD = { learningRate = opt.lr, beta1 = opt.beta1 }
local optimStateG = { learningRate = opt.lr, beta1 = opt.beta1 }
local plot_err_gap = 100

local feat_err, feat_err_per, pixel_err
local real_label = 1
local fake_label = 0
local batch_doafn_out_masked = torch.Tensor(opt.batchSize, 3, opt.imgscale, opt.imgscale)
local batch_doafn_feat = torch.Tensor(opt.batchSize, 512, 4, 4)
local batch_im_fake = torch.Tensor(opt.batchSize, 3, opt.imgscale, opt.imgscale)
local label = torch.Tensor(opt.batchSize)
local batch_im_out_per = torch.Tensor(opt.batchSize, 3, opt.imgscale, opt.imgscale)
local mean_pixel = torch.Tensor({103.939,116.779,123.68})
mean_pixel = mean_pixel:view(3,1,1):expandAs(torch.Tensor(3,opt.imgscale,opt.imgscale))
mean_pixel = torch.repeatTensor(mean_pixel,opt.batchSize,1,1,1)
local perm = torch.LongTensor{3,2,1}
local tm = torch.Timer()
local data_tm = torch.Timer()
local doafn_feat_idx = 18
-- finding index of output(Tanh())
local tanh_out_idx
for i,node in ipairs(netG.forwardnodes) do
	name = node.data.annotations.name
	if name == 'tanh_out' then
		tanh_out_idx = i
	end
end
print(tanh_out_idx)

if opt.gpu >= 0 then
  print('<gpu> using device ' .. opt.gpu)
  cutorch.setDevice(opt.gpu+1)
	mean_pixel = mean_pixel:cuda()
	if opt.tv_weight > 0 then
		tvloss = tvloss:cuda()
	end
	batch_doafn_out_masked = batch_doafn_out_masked:cuda()
	batch_doafn_feat = batch_doafn_feat:cuda()
	batch_im_fake = batch_im_fake:cuda()
	batch_im_out_per = batch_im_out_per:cuda()
	lossnet = lossnet:cuda()
	label = label:cuda()
  netG = netG:cuda()
	netD = netD:cuda()
	doafn = doafn:cuda()
	criterionGAN = criterionGAN:cuda()
	criterion_pixel = criterion_pixel:cuda()
	criterion_l1= criterion_l1:cuda()
end
local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()

	 -- processing real images
   label:fill(real_label)
   local out = netD:forward(batch_im_out_per)
	 local df_do={}
	 for l=1,opt.loss_layer do
		 table.insert(df_do,out[l]:clone():fill(0))
	 end
	 local output_label = out[opt.loss_layer+1]
   local errD_real = criterionGAN:forward(output_label, label)
   local df = criterionGAN:backward(output_label, label)
	 df_do[opt.loss_layer+1] = df
   netD:backward(batch_im_out_per, df_do)

	 -- generating fake images from generator
   local fake = netG:forward({batch_doafn_out_masked, batch_doafn_feat, batch_view_in, mean_pixel})
   batch_im_fake:copy(fake)
   
	 -- processing fake images
	 label:fill(fake_label)
   local out = netD:forward(batch_im_fake)
   local output_label = out[opt.loss_layer+1]
   local errD_fake = criterionGAN:forward(output_label, label)
   local df = criterionGAN:backward(output_label, label)
	 df_do[opt.loss_layer+1] = df
   netD:backward(batch_im_fake, df_do)

   return errD_real + errD_fake, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersD:zero()
   gradParametersG:zero()

   local fake = netG:forward({batch_doafn_out_masked, batch_doafn_feat, batch_view_in, mean_pixel})
   batch_im_fake:copy(fake)

	 -- GAN LOSS and FEATURE MATCHING
   -- get features for real images
	 local out = netD:forward(batch_im_out_per)
	 local feat_real={}
	 for l=1,opt.loss_layer do
		 table.insert(feat_real,out[l]:clone())
	 end
   -- get features for fake images
	 local out = netD:forward(batch_im_fake)
	 local feat_fake={}
	 for l=1,opt.loss_layer do
		 table.insert(feat_fake,out[l]:clone())
	 end
   -- compute gradient of feature matching
	 local df_do= {}
	 feat_err = 0
	 for l=1,opt.loss_layer do
		 local err = criterion_pixel:forward(feat_fake[l], feat_real[l])
		 local derr = criterion_pixel:backward(feat_fake[l], feat_real[l])
		 feat_err = feat_err + err
		 table.insert(df_do,derr:clone():mul(opt.lambda1))
	 end

	 -- fake labels are real for generator cost
   local output_label = out[opt.loss_layer+1]
	 label:fill(real_label)
	 local errG = criterionGAN:forward(output_label, label)
	 local df = criterionGAN:backward(output_label, label)
	 table.insert(df_do,df:clone())

	 -- compute gradient of GAN + feature loss, add tv regularization
	 local df_do1 = netD:updateGradInput(batch_im_fake, df_do)

	 -- PERCEPTUAL LOSS
	 -- compute gt features for perceptual loss
	 local feat_gt_per={}
	 local feat = lossnet:forward(batch_im_out_per)
	 for l=1,opt.loss_layer do
		 table.insert(feat_gt_per,feat[l]:clone())
	 end
	 -- compute fake features for perceptual loss
	 local feat_fake_per = lossnet:forward(batch_im_fake)
	 -- compute error in feature level
	 local d_feat_per={}
	 feat_err_per = 0
	 for l=1,opt.loss_layer do
		 local err = criterion_pixel:forward(feat_fake_per[l], feat_gt_per[l])
		 local derr = criterion_pixel:backward(feat_fake_per[l], feat_gt_per[l])
		 feat_err_per = feat_err_per + err
		 table.insert(d_feat_per,derr:clone())
	 end
	 local df_do2 = lossnet:updateGradInput(batch_im_fake,d_feat_per)
	 if opt.tv_weight > 0 then
		 local d_tvloss = tvloss:updateGradInput(batch_im_fake,0)
		 --print(torch.abs(df_do2):mean(), torch.abs(df_do2):max(), torch.abs(df_do2):min())
		 --print(torch.abs(d_tvloss):mean(),torch.abs(d_tvloss):max(), torch.abs(d_tvloss):min() )
		 df_do2:add(d_tvloss)
	 end
	 df_do2:mul(opt.lambda2)

	 -- compute error in pixel level
	 --local d_pxloss
	 --pixel_err = 0
	 --if opt.pixel_weight > 0 then
		 pixel_err = criterion_l1:forward(batch_im_fake, batch_im_out_per)
		 local d_pxloss = criterion_l1:backward(batch_im_fake, batch_im_out_per)
		 d_pxloss:mul(opt.pixel_weight)
	 --end
	 --print(pixel_err)
	 --print(d_pxloss:mean(),d_pxloss:max(), d_pxloss:min() )
	 --print(torch.abs(df_do1):mean(),torch.abs(df_do1):max(), torch.abs(df_do1):min() )
	 --print(torch.abs(df_do2):mean(),torch.abs(df_do2):max(), torch.abs(df_do2):min() )

	 netG:backward( {batch_doafn_out_masked, batch_doafn_feat, batch_view_in, mean_pixel}, df_do1:add(df_do2):add(d_pxloss) )
	 
   return errG, gradParametersG
end

local get_batch = function()
	batch_im_in, batch_im_out, batch_view_in = data:getBatch()
	batch_im_out_per:copy(batch_im_out)
	-- make it [0, 1] -> [0, 255], and BGR, subtract mean
	batch_im_out_per:mul(255)
	batch_im_out_per = batch_im_out_per:index(2,perm)	
	batch_im_out_per:add(-1,mean_pixel)

	-- make it [0, 1] -> [-1, 1]
	batch_im_in:mul(2):add(-1)
	batch_im_out:mul(2):add(-1)
	if opt.gpu >= 0 then
		batch_im_in = batch_im_in:cuda()
		batch_im_out = batch_im_out:cuda()
		batch_view_in = batch_view_in:cuda()
	end
	local f = doafn:forward({batch_im_in, batch_view_in})
	-- [-1, 1] -> mask -> [0, 2]
	batch_doafn_out_masked:copy(f[1]):add(1)
	for j=1,opt.batchSize do
		batch_doafn_out_masked[j]:cmul(f[2][j]:repeatTensor(3,1,1))
	end
	-- [0, 2] -> mask -> [-1, 1]
	batch_doafn_out_masked:add(-1)

	batch_doafn_feat:copy(doafn.forwardnodes[doafn_feat_idx].data.module.output)
end

for t = epoch+1, opt.maxEpoch do
	-- training
  netG:training()
  netD:training()
	doafn:evaluate()
	for i = 1, opt.iter_per_epoch do
		local iter = i+(t-1)*opt.iter_per_epoch
		data_tm:reset(); data_tm:resume()
		get_batch()
		data_tm:stop()

		tm:reset(); tm:resume()
		collectgarbage()
		-- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
		_,errD = optim.adam(fDx, parametersD, optimStateD)
		-- (2) Update G network: maximize log(D(G(z)))
		_,errG = optim.adam(fGx, parametersG, optimStateG)
		for iterG=2,opt.iterG do
			-- train with real
			data_tm:reset(); data_tm:resume()
			get_batch()
			data_tm:stop()
			_,errG = optim.adam(fGx, parametersG, optimStateG)
		end
		tm:stop()
		print(string.format('#### epoch (%d) iter (%d) pixel_err = %.3f, feat_err = %.3f, feat_err_per = %.3f, errD = %.3f, errG = %.3f Time: %.3f DataTime: %.3f ', t, iter, pixel_err, feat_err, feat_err_per, errD[1], errG[1], tm:time().real, data_tm:time().real))
		loss_listD = torch.cat(loss_listD, torch.Tensor(1,1):fill(errD[1]),1)
		loss_listG = torch.cat(loss_listG, torch.Tensor(1,1):fill(errG[1]),1)

		-- plot 
		if iter % 250 == 0 then
			local nrow = 4
			local to_plot={}
			local pred = netG.forwardnodes[tanh_out_idx].data.module.output:clone()
			pred = pred:index(2,perm)
			for k=1,10 do
				to_plot[(k-1)*nrow + 1] = batch_doafn_out_masked[k]:clone()
				to_plot[(k-1)*nrow + 1]:add(1):mul(0.5)
				to_plot[(k-1)*nrow + 2] = pred[k]
				to_plot[(k-1)*nrow + 2]:add(1):mul(0.5)
				to_plot[(k-1)*nrow + 3] = batch_im_in[k]:clone()
				to_plot[(k-1)*nrow + 3]:add(1):mul(0.5)
				to_plot[(k-1)*nrow + 4] = batch_im_out[k]:clone()
				to_plot[(k-1)*nrow + 4]:add(1):mul(0.5)
			end
			formatted = image.toDisplayTensor({input=to_plot, nrow = nrow})
			image.save((opt.modelPath .. '/training/' .. 
					string.format('training_output_%05d.jpg',iter)), formatted)

			-- plot errors
			print('plotting errs...')
			local listD = loss_listD[{{2,loss_listD:size(1)},{}}]
			local listG = loss_listG[{{2,loss_listG:size(1)},{}}]
			local len = listD:size(1)/plot_err_gap
			local lossD = torch.Tensor(len)
			local lossG = torch.Tensor(len)
			for k=1,len do
				lossD[k] = listD[(k-1)*plot_err_gap+1][1]
				lossG[k] = listG[(k-1)*plot_err_gap+1][1]
			end
			gnuplot.pngfigure(opt.modelPath .. string.format('/gan_loss_epoch-%d.png',t))
			gnuplot.plot({'D',lossD,'-'},{'G',lossG,'-'})
			gnuplot.plotflush()
			
			--io.stdin:read('*l')
		end

	end
	
  if t % opt.saveFreq == 0 then
			collectgarbage()
			parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
			parametersG, gradParametersG = nil, nil
			torch.save(opt.modelPath .. string.format('/net-epoch-%d.t7', t),
				{netG = netG:clearState(), netD = netD:clearState(), loss_listD = loss_listD, 
				loss_listG = loss_listG})
			parametersD, gradParametersD = netD:getParameters()
			parametersG, gradParametersG = netG:getParameters()
	end
end
