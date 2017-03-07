require 'torch'
require 'lfs'
ffi = require 'ffi'

opt = lapp[[
  --category          (default 'car')
	--data_dir					(default '../data/')
]]

data_dir = opt.data_dir .. '/' .. opt.category

models={}
for f in lfs.dir(data_dir) do
  if f ~= "." and f ~= ".." then
 	 table.insert(models,f)
  end
end
n_samples= #models

metadata={}
n_train = torch.ceil(n_samples*0.8)
n_test = n_samples - n_train 
metadata.n_train = n_train
metadata.n_test = n_test
random_indices = torch.randperm(n_samples)
metadata.train_indices = random_indices[{{1,n_train}}]
metadata.test_indices = random_indices[{{n_train+1,n_samples}}]

lengths = torch.Tensor(#models)
for i=1,#models do
	lengths[i] = string.len(models[i])
end
max_len = lengths:max()+1
model_names = torch.CharTensor()
model_names:resize(#models,max_len):fill(0)
s_data = model_names:data()
for i=1,#models do
	print(models[i])
	ffi.copy(s_data, models[i])
	s_data = s_data + max_len
end

metadata.models = model_names
out_file = string.format('../data/metadata_%s.cache',opt.category)
torch.save(out_file,metadata)
