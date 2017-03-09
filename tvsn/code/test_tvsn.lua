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
require 'lfs'
assert(loadfile("image_error_measures.lua"))(true)
local ffi = require 'ffi'

torch.setdefaulttensortype('torch.FloatTensor')

opt = lapp[[
  -g, --gpu           (default 0)
  --imgscale          (default 256)
  --background				(default 0)
  --modelDir					(default '../snapshots/')
	--data_dir					(default '../data/')
	--doafn_path				(default '../snapshots/pretrained/doafn_car_epoch200.t7')
	--tvsn_path					(default '../snapshots/pretrained/tvsn_car_epoch220.t7')
  --category          (default 'car')
  --plot							(default 0)
	-d, --debug					(default 0)
]]

if opt.debug > 0 then
	debugger = require('fb.debugger')
	debugger.enter()
end

opt.expPath = string.format('./results_%s/',opt.category)
if not paths.dirp(opt.expPath) then
  paths.mkdir(opt.expPath)
end

opt.doafn_path = string.format('DOAFN_SYM_%s_%s_bs025/net-epoch-200.t7',opt.imgscale,opt.category)

print('Loading metadata')
opt.metadata = string.format('../data/metadata_%s.cache',opt.category)
local metadata = torch.load(opt.metadata)
local testset_info = torch.load(string.format('../data/testset_%s.t7',opt.category))
local n_samples = testset_info:size(1)

if opt.gpu < 0 or opt.gpu > 3 then opt.gpu = -1 end
print('set gpu...')
if opt.gpu >= 0 then
  cutorch.setDevice(opt.gpu+1)
  print('<gpu> using device ' .. opt.gpu)
end

print('loading DOAFN...')
loader = torch.load(opt.modelDir .. opt.doafn_path)
local doafn = loader.net

print('loading TVSN...')
loader = torch.load(opt.modelDir .. opt.tvsn_path)
local comp_dcgan_per = loader.netG

local criterion_l1 = nn.AbsCriterion()
criterion_l1.sizeAverage = false

local doafn_feat_idx = 18
local comp_dcgan_per_out_idx
for i,node in ipairs(comp_dcgan_per.forwardnodes) do
	name = node.data.annotations.name
	if name == 'tanh_out' then
		comp_dcgan_per_out_idx = i
	end
end

local transforms = {20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340}

local n_batch = 15

local batch_im_in = torch.Tensor(25,3,opt.imgscale,opt.imgscale)
local batch_im_out = torch.Tensor(25,3,opt.imgscale,opt.imgscale)
local batch_im_out_alpha = torch.Tensor(n_batch,3,opt.imgscale,opt.imgscale)
local batch_view_in_doafn = torch.Tensor(25,17)
local batch_doafn_out_masked = torch.Tensor(25, 3, opt.imgscale, opt.imgscale)
local batch_doafn_feat = torch.Tensor(25, 512, 4, 4)
local mean_pixel = torch.Tensor({103.939,116.779,123.68})
mean_pixel = mean_pixel:view(3,1,1):expandAs(torch.Tensor(3,opt.imgscale,opt.imgscale))
mean_pixel = torch.repeatTensor(mean_pixel,25,1,1,1)

if opt.gpu >= 0 then
	batch_im_in = batch_im_in:cuda()
	batch_im_out = batch_im_out:cuda()
	batch_im_out_alpha = batch_im_out_alpha:cuda()
	batch_view_in_doafn = batch_view_in_doafn:cuda()
	batch_doafn_out_masked = batch_doafn_out_masked:cuda()
	batch_doafn_feat = batch_doafn_feat:cuda()
	doafn = doafn:cuda()
	comp_dcgan_per = comp_dcgan_per:cuda()
	mean_pixel = mean_pixel:cuda()
	criterion_l1 = criterion_l1:cuda()
end

print('testing...')
N = metadata.n_test

local test_output_l1 = torch.Tensor(n_samples,2)
local test_output_ssim = torch.Tensor(n_samples,2)

for i=1,n_samples,15 do
	collectgarbage()
	print('processing ' .. i)
	for j=1,15 do
		local idx = (i-1) + j
		local model_id = testset_info[idx][1]
		local model_name = ffi.string(torch.data(metadata.models[model_id]))

		local src_theta_id = testset_info[idx][2]
		local transform_id = testset_info[idx][3]
		local phi = testset_info[idx][4]

		local src_theta = (src_theta_id-1)*20
		local one_hot_doafn = torch.zeros(17)
		one_hot_doafn:scatter(1,torch.LongTensor{transform_id},1)
		local dst_theta = src_theta + transforms[transform_id]
		if dst_theta < 0 then
			dst_theta = 360 + dst_theta 
		elseif dst_theta > 350 then
			dst_theta = dst_theta - 360
		end
		local imgpath1 = opt.data_dir .. opt.category .. '/' .. model_name .. 
		'/model_views/' .. string.format('%d_%d.png',src_theta, phi)
		local imgpath2 = opt.data_dir .. opt.category .. '/' .. model_name .. 
		'/model_views/' .. string.format('%d_%d.png',dst_theta, phi)
		local im1 = image.load(imgpath1, 4, 'float')
		im1 = image.scale(im1,opt.imgscale,opt.imgscale)
		local im2 = image.load(imgpath2, 4, 'float')
		im2 = image.scale(im2,opt.imgscale,opt.imgscale)
		-- rgba -> rgb with white background
		local im1_rgb = im1[{{1,3},{},{}}]
		local im2_rgb = im2[{{1,3},{},{}}]
		local alpha1 = im1[4]:repeatTensor(3,1,1)
		local alpha2 = im2[4]:repeatTensor(3,1,1)
		im1_rgb:cmul(alpha1):add(1-alpha1)
		im2_rgb:cmul(alpha2):add(1-alpha2)
		batch_im_in[j]:copy(im1_rgb)
		batch_im_out[j]:copy(im2_rgb)
		batch_im_out_alpha[j]:copy(alpha2:gt(0):cuda())
		batch_view_in_doafn[j]:copy(one_hot_doafn)
	end
	batch_im_in:mul(2):add(-1)
	batch_im_out:mul(2):add(-1)

	-- fill rest of the batch
	batch_im_in[{{16,25},{},{},{}}]:copy(batch_im_in[{{1,10},{},{},{}}])
	batch_view_in_doafn[{{16,25},{}}]:copy(batch_view_in_doafn[{{1,10},{}}])
	
	local doafn_f = doafn:forward({batch_im_in, batch_view_in_doafn})
	batch_doafn_feat:copy(doafn.forwardnodes[doafn_feat_idx].data.module.output)

	-- [-1,1] -> [0, 2] -> mask -> [-1, 1]
	batch_doafn_out_masked:copy(doafn_f[1]):add(1)
	for j=1,n_batch do
		batch_doafn_out_masked[j]:cmul(doafn_f[2][j]:repeatTensor(3,1,1))
	end
	batch_doafn_out_masked:add(-1)
	
	local input1 = batch_doafn_out_masked[{{1,n_batch},{},{},{}}]
	local input2 = batch_doafn_feat[{{1,n_batch},{},{},{}}]
	local input3 = batch_view_in_doafn[{{1,n_batch},{}}]
	local input4 = mean_pixel[{{1,n_batch},{},{},{}}]
	comp_dcgan_per:forward({input1, input2, input3, input4})
	local comp_dcgan_per_f = comp_dcgan_per.forwardnodes[comp_dcgan_per_out_idx].data.module.output:clone()
	comp_dcgan_per_f = comp_dcgan_per_f:index(2,torch.LongTensor{3,2,1})
	
	batch_im_in:add(1):mul(0.5)
  batch_im_out:add(1):mul(0.5)
	doafn_f[1]:add(1):mul(0.5)
	comp_dcgan_per_f:add(1):mul(0.5)
	batch_doafn_out_masked:add(1):mul(0.5)

	if opt.plot==1 then
		local to_plot={}
		local nrow = 6
		for j=1,n_batch do
			to_plot[(j-1)*nrow + 1] = batch_im_in[j]:clone()
			to_plot[(j-1)*nrow + 2] = batch_im_out[j]:clone()
			to_plot[(j-1)*nrow + 3] = doafn_f[1][j]:clone()
			to_plot[(j-1)*nrow + 4] = doafn_f[2][j]:repeatTensor(3,1,1):clone()
			to_plot[(j-1)*nrow + 5] = batch_doafn_out_masked[j]:clone()
			to_plot[(j-1)*nrow + 6] = comp_dcgan_per_f[j]
		end
		local formatted = image.toDisplayTensor({input=to_plot, nrow=nrow})
		image.save(opt.expPath .. string.format('/test%03d.jpg',i), formatted)
	end

  batch_im_in[{{1,n_batch},{},{},{}}]:cmul(batch_im_out_alpha)
  batch_im_out[{{1,n_batch},{},{},{}}]:cmul(batch_im_out_alpha)
	doafn_f[1][{{1,n_batch},{},{},{}}]:cmul(batch_im_out_alpha)
	comp_dcgan_per_f[{{1,n_batch},{},{},{}}]:cmul(batch_im_out_alpha)
	
	-- measure the similarity
	for k=1,n_batch do
		local idx = (i-1) + k
		local n_fg = batch_im_out_alpha[{{k},{1},{},{}}]:sum()
		local l1_err = torch.Tensor(2):fill(0)
		local ssim = torch.Tensor(2):fill(0)

		l1_err[1] = criterion_l1:forward(doafn_f[1][k], batch_im_out[k])/(3*n_fg)
		l1_err[2] = criterion_l1:forward(comp_dcgan_per_f[k], batch_im_out[k])/(3*n_fg)
	
		ssim[1] = SSIM(doafn_f[1][k]:float(), batch_im_out[k]:float())
		ssim[2] = SSIM(comp_dcgan_per_f[k]:float(), batch_im_out[k]:float())

		test_output_l1[idx]:copy(l1_err)
		test_output_ssim[idx]:copy(ssim)
	end

end

result={}
result.test_output_l1 = test_output_l1
result.test_output_ssim = test_output_ssim
torch.save(string.format('test_output_%s.t7',opt.category),result)
