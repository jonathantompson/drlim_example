-- Most of this code was the result of an unrelated NYU project by Ross 
-- Gorishin and Jonathan Tompson.  It has been simplied for readability and to
-- reproduce the drlim results only (it was originally an autoencoder that 
-- utilized the drlim loss criterion).  

-- THIS CODE IS A MESS!!!!... particularly the data loading and processing.
-- It is also pretty slow.  I've made it public just a a working example
-- of DRLIM.

require 'cunn'
require 'image'
require 'parallel'
require 'xlua'
require 'mattorch'
require 'optim'
dofile('pbar.lua')

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)
torch.manualSeed(1)
math.randomseed(1)
cutorch.setDevice(2)
if math.mod == nil then math.mod = math.fmod end

-- *************************************
-- PARAMETERS
-- *************************************
data_dir = './data/' 
results_dir = './results/' 
os.execute('mkdir -p '..results_dir)
os.execute('mkdir -p '..data_dir)
network_dim = {500, 200, 2}  -- 1st, 2nd, 3rd layer dimensions
margin = 1.5  -- HingeEmbeddingCriterion parameter
M = 16   -- Number of dissimilar pairs parameter when generating epoch data 
         -- (defines ratio of similar to dissimilar pairs)
learning_rate = 0.001  -- SGD learning rate 
epochs = 40   -- Number of training epochs

-- *************************************
-- LOAD DATA (norb dataset)
-- *************************************
dofile("data.lua")
factors, norb = load_small_norb(data_dir) 
-- note factors are: dim1 = sample, dim2 = sample_parameter.
-- sample_parameter = [1.sample #][2.instance 4,6,7,8,9][3.elevation][4.azimuth]
--                    [5.lighting][6.class]
factors_full = factors:clone() 
-- Choose a subset of the dataset (one or more instances of rotating objects)
class = 2
instances ={4}  -- class 2 has instances 4,5,7,8,9 (select 1 or more)
factors, kNN = select_norb_subset(factors, instances, class)
-- plot_data()
--Basic preprocessing 
X = norb:reshape(norb:size(1),norb:size(3)*norb:size(4)):float():squeeze()
width = math.sqrt(X:size(2))
height = math.sqrt(X:size(2))
X:add(-X:mean())  
X:div(X:max())
im_size = X:size(2)
collectgarbage()  

-- Plot some examples
plot_data() -- black indicates they are NOT close. white indicates they are 

-- *************************************
-- DEFINE THE MODEL
-- *************************************
dofile("model.lua")
model, criterion, encoder = create_model()
optimState = {
  learningRate = learning_rate,
  weightDecay = 0,
  momentum = 0,
  dampening = 0,
  learningRateDecay = 0
}
optimMethod = optim.sgd
parameters, gradParameters = model:getParameters()

-- *************************************
-- PERFORM TRAINING
-- *************************************
dofile("train.lua")
print("Training...")

for iter = 1, epochs do 
  print('Iteration ' .. iter .. ' of ' .. epochs)

  av_error = train()
  print("\nave_error = " .. av_error)
  
  collectgarbage()
  torch.save(results_dir..'model', model)
end

-- *************************************
-- ANALYSIS
-- *************************************

-- Save the lower dimension points to disk
n = factors:size(1)
x_low_dim = torch.FloatTensor(n, network_dim[#network_dim])
model_encoder = encoder:clone()
sample = torch.CudaTensor(im_size)
for i = 1,n do
  sample:copy(X[factors[i][1]])
  x_low_dim[{i,{}}]:copy(model_encoder:forward(sample))
end
-- Save the low dim points to binary, as well as the elevation and azimuth
o = torch.DiskFile(results_dir..'/x_low_dim.bin', 'w')
o:binary()
o:writeFloat(x_low_dim:storage())
o:close()
o = torch.DiskFile(results_dir..'/factors.bin', 'w')
o:binary()
o:writeInt(factors:storage())
o:close()

print('Results have been saved to .bin files in results directory.') 
print('Run analysis.mat to see the full results')
