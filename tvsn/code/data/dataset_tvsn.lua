--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
require 'image'
require 'lfs'
local ffi = require 'ffi'

local dataset = torch.class('dataLoader')

function dataset:__init(data_dir, category, loadSize, split, background)
	 self.data_dir = data_dir
	 self.category = category 
	 self.loadSize = loadSize
	 self.theta_inc = 20
	 self.theta_max = 340
	 self.n_theta = 18
	 self.phi_inc = 10
	 self.phi_max = 20
	 self.n_phi = 3
	 self.split = split
	 self.background = background
end

-- size()
function dataset:size()
	return {self.metadata.n_samples, self.metadata.n_train, self.metadata.n_test}
end

-- sampler, samples from the training set.
function dataset:sample(quantity)
   assert(quantity)
   local data1 = torch.Tensor(quantity,
		       self.loadSize[1], self.loadSize[2], self.loadSize[3])
   local data2 = torch.Tensor(quantity,
		       self.loadSize[1], self.loadSize[2], self.loadSize[3])
   local trans = torch.Tensor(quantity,17)
   for i=1,quantity do
      local out1, out2, transform = self:get()
      data1[i]:copy(out1)
      data2[i]:copy(out2)
			trans[i]:copy(transform)
   end
   return data1, data2, trans 
end

function dataset:get()
	local model_id
	if self.split == 'train' then
		model_id = self.metadata.train_indices[torch.random(self.metadata.n_train)]
	elseif self.split == 'val' then
		model_id = self.metadata.val_indices[torch.random(self.metadata.n_val)]
	elseif self.split == 'test' then
		model_id = self.metadata.test_indices[torch.random(self.metadata.n_test)]
	end
	--local phi = (torch.random(self.n_phi)-1)*self.phi_inc
	local phi_id = torch.random(self.n_phi) 
	local phi = (phi_id-1)*self.phi_inc
	local src_theta_id = torch.random(self.n_theta) 
	local src_theta = (src_theta_id-1)*self.theta_inc
	local transform_id = torch.random(17)
	local one_hot = torch.zeros(17)
	one_hot:scatter(1,torch.LongTensor{transform_id},1)

	local dst_theta = src_theta + transform_id*20
	if dst_theta < 0 then
		dst_theta = 360 + dst_theta 
	elseif dst_theta > 350 then
		dst_theta = dst_theta - 360
	end


	--source image and target image
	local model_name = ffi.string(torch.data(self.metadata.models[model_id]))
	local imgpath1 = self.data_dir .. self.category .. '/' .. model_name .. 
			'/model_views/' .. string.format('%d_%d.png',src_theta, phi)
	local imgpath2 = self.data_dir .. self.category .. '/' .. model_name .. 
			'/model_views/' .. string.format('%d_%d.png',dst_theta, phi)
	local im1 = self:sampleHookTrain(imgpath1)
	local im2 = self:sampleHookTrain(imgpath2)
	local im1_rgb = im1[{{1,3},{},{}}]
	local im2_rgb = im2[{{1,3},{},{}}]
	local alpha1 = im1[4]:repeatTensor(3,1,1)
	local alpha2 = im2[4]:repeatTensor(3,1,1)
	if self.background == 1 then
		-- rgba -> rgb with random background
		local bgpath = self.data_dir .. '/background/' .. 
		string.format('%06d.jpg',torch.random(10000))
		local bg = image.load(bgpath, 3, 'float')
		if self.loadSize[2] ~= bg:size(2) then
			bg= image.scale(bg, self.loadSize[2], self.loadSize[2])
		end
		local bg_temp = (1-alpha1):cmul(bg)
		im1_rgb:cmul(alpha1):add(bg_temp)
		local bg_temp = (1-alpha2):cmul(bg)
		im2_rgb:cmul(alpha2):add(bg_temp)
	else
		im1_rgb:cmul(alpha1):add(1-alpha1)
		im2_rgb:cmul(alpha2):add(1-alpha2)
	end

	return im1_rgb,im2_rgb,one_hot
end

return dataset
