train = function()
  local point_pairs = gen_epoch_data(kNN,M)
  local av_error = 0 
  local nsamples = 0

  for i = 1,point_pairs:size(1) do
    if (math.mod(i,100)==0 or i == point_pairs:size(1)) then 
      progress(i,point_pairs:size(1))
    end
    
    if (point_pairs[i][1]~=0 and point_pairs[i][2]~=0 and
         point_pairs[i][1]~=point_pairs[i][2]) then
      local data = {torch.FloatTensor(im_size), torch.FloatTensor(im_size)}
      data[1] = X[point_pairs[i][1]]:cuda()
      data[2] = X[point_pairs[i][2]]:cuda()
      local target = point_pairs[i][3]

      local feval = function(x)
        gradParameters:zero()
        local pred = model:forward(data)
        local err = criterion:forward(pred, target)
        av_error = av_error + err
        local grad = criterion:backward(pred, target)
        model:backward(data, grad)
        return err, gradParameters
      end

      -- optimize on current sample
      optimMethod(feval, parameters, optimState)

      nsamples = nsamples + 1
    end
  end
   
  av_error = av_error / nsamples 
  return av_error  
end

