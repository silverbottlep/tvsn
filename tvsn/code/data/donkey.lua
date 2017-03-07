--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
if opt.dataset == 'doafn' then
	paths.dofile('dataset_doafn.lua')
elseif opt.dataset == 'tvsn' then
	paths.dofile('dataset_tvsn.lua')
end

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
-- Check for existence of opt.data
print('Loading train metadata')
metadata_cache = torch.load(opt.metadata)
--------------------------------------------------------------------------------------------
-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
   collectgarbage()
   local im = image.load(path, 4, 'float')
	 if self.loadSize[2] ~= im:size(2) then
		 im = image.scale(im, self.loadSize[2], self.loadSize[2])
	 end
   return im
end

print('Creating train metadata')
trainLoader = dataLoader(opt.data_dir, opt.category, 
		{3, opt.imgscale, opt.imgscale}, opt.split, opt.background)
trainLoader.sampleHookTrain = trainHook
trainLoader.sampleHookTest = trainHook
trainLoader.metadata = metadata_cache
trainLoader.maps = opt.maps
trainLoader.map_indices = opt.map_indices
