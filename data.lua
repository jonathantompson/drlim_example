dofile("lushio.lua")

--Returns NORB dataset and labels 
load_small_norb = function(data_dir) 
  local norb_dir = data_dir..'/small_norb/'

  if not paths.dirp(norb_dir) then
    os.execute('mkdir -p ' .. norb_dir)
  end

  if not paths.filep(norb_dir.."processed_smallnorb.tar.gz") then
     local www = 'http://cims.nyu.edu/~tompson/data/processed_smallnorb.tar.gz'
     local tar = sys.basename(www)
     os.execute('wget -P ' .. norb_dir .. ' ' .. www)
     os.execute('tar -C ' .. norb_dir .. ' -xvf ' .. norb_dir .. tar)
  end

  local norb = lushio.read(norb_dir..'norb_dat.mat') 
  norb = norb:select(2,1)
  norb = norb:clone():resize(norb:size(1),1,96,96)

  --Load labels 
  --factors =[sample #][instance 4,6,7,8,9][elevation][azimuth][lighting][class]
  local class = lushio.read(norb_dir..'norb_cat.mat') 
  local info = lushio.read(norb_dir..'norb_info.mat') 
  local factors = torch.cat(info,class)
  factors = torch.cat(torch.range(1,factors:size(1)):int(), factors, 2) 

  collectgarbage()
  return factors, norb  

end

-- ****************************************************************
-- NOTE: THE VAST MAJORITY OF THE CODE WRITTEN BELOW WAS BY ROSS G.
--       (IT NEEDS A LOT OF CLEANING UP!)
-- ****************************************************************

-- This function was EXP2_data in the original codebase (Ross and Jonathan's)
-- single-instances of rotating objects
-- factors =[1.sample #][2.instance 4,6,7,8,9][3.elevation][4.azimuth][5.lighting][6.class]
select_norb_subset = function(factors, instances)
    local kNN = torch.zeros(#instances,162,5):int()

    for idx,inst in ipairs(instances) do

        local filter = {[2] = inst, [5] = 1, [6] = 2}
        local inst_factors = apply_filter(factors, filter)
        kNN[idx] = find_fNN(inst_factors)

    end

    --single lighting condition, single class
    local filt_factors = apply_filter(factors, {[5] = 1, [6] = 2})

    --remove instance information  
    filt_factors = torch.cat(filt_factors:select(2,1), filt_factors:narrow(2,3,2), 2)

    return filt_factors, shuffle(kNN)
end

--Filter the [factors] to include
--only one [value] of factor [f]
filter_factor = function(factors, f, value)

    local n = 0

    for i = 1, factors:size(1) do

        if factors[i][f] == value then

            n = n + 1

            factors:select(1,n):copy(factors:select(1,i))

        end

    end

    --remove extra rows 
    if n>0 then

        factors = factors:narrow(1,1,n)
        collectgarbage()

    else

        print('Warning: factor value not found, no filtering was performed!')

    end


    return factors

end

--Filter [factors] according to the [filter] 
--table, i.e. if filter = {[2] = 4} then 
--factor 2 will only contain instances of 4
--factors =[1.sample #][2.instance 4,6,7,8,9][3.elevation][4.azimuth][5.lighting][6.class]
apply_filter = function(factors, filter)

    local factors = factors:clone()
    local nfac = factors:size(2)
    local nfil = 0

    for f,v in pairs(filter) do

        factors = filter_factor(factors, f, v)
        nfil = nfil + 1

    end

    --Copy all unfiltered factors 
    local filtered_factors = torch.Tensor(factors:size(1),nfac-nfil):int()
    local n = 1

    for i = 1, nfac do

        if filter[i]==nil then
            filtered_factors:select(2,n):copy(factors:select(2,i))
            n = n + 1
        end

    end

    return filtered_factors

end
--Finds neighbors in factor space 
--2 neighbors [high] and [low] for 
--each factor, i.e. 2*[nfac] neighbors 
find_fNN = function(factors)

    local nsamp = factors:size(1)
    local nfac = factors:size(2)
    local fNN = torch.zeros(nsamp,2*(nfac-1)):int()
    fNN = torch.cat(factors:select(2,1), fNN, 2)

    print('Finding neighbors...')

    for i = 1, nsamp do

        progress(i,nsamp)

        local n = 2

        for j = 2, nfac do

            local ff = factors:clone()

            local f_val = ff[i][j]

            ff = get_slice(ff,i,j)

            local val, idx = torch.sort(ff:select(2,j))

            local t = find(val,f_val)

            --check if the factor is azimuth (only circular factor) 
            if ff:select(2,j):max() == 34 and ff:select(2,j):min() == 0  then

                --low neighbor 
                if t>1 then
                    n1 = ff[idx[t-1]][1]
                elseif t == 1 then
                    n1 = ff[idx[val:size(1)]][1]
                else
                    n1 = 0
                end

                --high neighbor 
                if t<val:size(1) then
                    n2 = ff[idx[t+1]][1]
                elseif t==val:size(1) then
                    n2 = ff[idx[1]][1]
                else
                    n2=0
                end

            else

                --low neighbor 
                if t>1 then n1 = ff[idx[t-1]][1] else n1=0 end
                --high neighbor 
                if t<val:size(1) then n2 = ff[idx[t+1]][1] else n2=0 end

            end

            fNN[i][n] = n1
            fNN[i][n+1] = n2

            n = n + 2

        end

    end

    collectgarbage()

    return fNN

end

--returns a slice of the data along factor [f] at [sample_idx] 
get_slice = function(factors, sample_idx, f)

    local sample = factors[sample_idx]
    local slice = factors:clone()

        for j = 2, factors:size(2) do

            if j ~= f then

                slice = filter_factor(slice,j,sample[j])

            end

        end

    return slice

end

find = function(x,target)

    for i = 1,x:size(1) do

        if x[i]==target then

            return i

        end

    end

end

shuffle = function(dataset)

    local shuffled_dataset = dataset:clone()
    local order = torch.randperm(shuffled_dataset:size(1))

    for i = 1,dataset:size(1) do

        shuffled_dataset[i] = dataset[order[i]]

    end

    return shuffled_dataset

end

--Epoch Data Set Generation 
--randomly generates mostly dissimilar 
--pairs, and concactenates them to all
--similar pairs 
gen_epoch_data = function(kNN, ndis)

    if kNN:dim()==2 then

        local n = kNN:size(1)
        local nfac = kNN:size(2)

        --Similar Pairs 
        local sim = kNN:narrow(2,1,2):clone()

        for i = 3, nfac do

            sim = torch.cat(sim, torch.cat(kNN:select(2,1), kNN:select(2,i), 2), 1)

        end

        sim = torch.cat(sim, torch.ones(sim:size(1)):int(), 2)

        --Mostly Dissimilar Pairs
        local data

        if ndis > 0 then

            local dis = torch.cat(shuffle(kNN:select(2,1)),shuffle(kNN:select(2,1)), 2)

            for i = 1,ndis-1 do

                dis = torch.cat(dis, torch.cat(shuffle(kNN:select(2,1)), shuffle(kNN:select(2,1)), 2), 1)

            end

            dis = torch.cat(dis, torch.ones(dis:size(1)):mul(-1):int(), 2)

            data = dis:cat(sim,1)

        else
            data = sim
        end

        return shuffle(data)

   elseif kNN:dim()==3 then

        local data = gen_epoch_data2(kNN, ndis)

        return shuffle(data)

    else

        print('ERROR: what kind of kNN tensor is this?')

        return nil
   end


end

gen_epoch_data2 = function(kNN, ndis)

    local pairs = gen_epoch_data(kNN[1], ndis)

    for i = 2, kNN:size(1) do

        pairs = torch.cat(pairs, gen_epoch_data(kNN[i], ndis), 1)

    end

    return shuffle(pairs)

end

plot_data = function()
  nexamples = 10
  input_pairs = gen_epoch_data(kNN,M)
  image_pairs = torch.zeros(nexamples,3,96,96)
  
  k = 1; i = 1
  while (k <= nexamples) do
      if input_pairs[i][1]~=0 and input_pairs[i][2]~= 0 then
          image_pairs:select(1,k):select(1,1):copy(norb[input_pairs[i][1]])
          image_pairs:select(1,k):select(1,2):copy(norb[input_pairs[i][2]])
          if input_pairs[i][3]==1 then
            image_pairs:select(1,k):select(1,3):fill(255)
          end
          k = k + 1
      end
      i = i + 1
  end
   
  image_pairs:resize(3*nexamples,1,96,96)
  image_pairs_disp = image.toDisplayTensor{scaleeach=true,nrow=3,input=image_pairs,padding=1} 
  image.display(image_pairs_disp)
end

project_points = function(X,factors,encoder,selector)

    local n = factors:size(1)
    local Xproj = torch.Tensor(n,2)
    local net = encoder:clone():add(selector)

    for i = 1,n do

       Xproj[i]:copy(net:forward(X[factors[i][1]]:cuda()))

    end

    return Xproj

end

--Center 2D Points at the Origin 
center = function(X)

    X:select(2,1):add(-X:select(2,1):mean())
    X:select(2,2):add(-X:select(2,2):mean())

    X:select(2,1):div(-X:select(2,1):max())
    X:select(2,2):div(-X:select(2,2):max())

    return(X)
end

--Plots points in 2D with color 
color_scatter = function(x, color)

    --Convert Tensor to String 
    --the format of [s] is x,y,color  

    local s = ''

        for i = 1, x:size(1) do

            s = s..tostring(x[i][1])..' '..tostring(x[i][2])..' '..tostring(color[i])..'\n'

        end

    local file = io.open(results_dir .. "/temp.txt", "w")
    file:write(s)
    file:close()

    gnuplot.raw('set palette rgb 3,11,16; plot "'..results_dir..'/temp.txt" using 1:2:3 with points palette unset colorbox')

end

