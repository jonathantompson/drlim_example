create_model = function()
  
  --Encoder 
  local encoder = nn.Sequential()
  encoder:add(nn.Linear(im_size,network_dim[1]))
  encoder:get(encoder:size()).bias:add(-encoder:get(encoder:size()).bias:min())
  encoder:add(nn.Threshold())
  encoder:add(nn.Linear(network_dim[1],network_dim[2])) 
  encoder:get(encoder:size()).bias:add(-encoder:get(encoder:size()).bias:min())
  encoder:add(nn.Threshold())
  encoder:add(nn.Linear(network_dim[2],network_dim[3])) 
  encoder:cuda()

  -- Full network --> Split two input images
  local model = nn.Sequential()

  -- Create the parallel siamese network and add it to the full network
  local encoder_siamese = nn.ParallelTable()
  encoder_siamese:add(encoder)
  encoder_siamese:add(encoder:clone('weight','bias','gradWeight','gradBias'))
  model:add(encoder_siamese) 

  -- Create the L2 distance function and add it to the full network
  local dist = nn.PairwiseDistance(2)
  model:add(dist)

  model:cuda()

  -- Criterion
  local criterion = nn.HingeEmbeddingCriterion(margin):cuda() 

  return model, criterion, encoder
end
