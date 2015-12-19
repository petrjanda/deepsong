nn = require 'nn'
npy4th = require 'npy4th'

wide = 512 -- 4096

x = torch.Tensor(20, wide, 128)
yt = torch.Tensor(20)

i = 1
for a in paths.iterdirs("spect/") do
    for s in paths.iterfiles("spect/" .. a) do
        input = npy4th.loadnpy('spect/' .. a .. '/' .. s)
        x[i] = input:transpose(1, 2):resize(wide, 128)
        yt[i] = a
        i = i + 1
    end
end

s = wide * 30

model = nn.Sequential()

model:add(nn.TemporalConvolution(128, 256, 4))
model:add(nn.ReLU())
model:add(nn.TemporalMaxPooling(1,4))

model:add(nn.TemporalConvolution(256, 256, 4))
model:add(nn.ReLU())
model:add(nn.TemporalMaxPooling(1,2))

model:add(nn.TemporalConvolution(256, 512, 4))
model:add(nn.ReLU())
model:add(nn.TemporalMaxPooling(1,2))

model:add(nn.Reshape(s))
model:add(nn.ReLU())
model:add(nn.Linear(s, 2))
model:add(nn.LogSoftMax())

criterion= nn.ClassNLLCriterion()

for e = 1, 1 do
    print("epoch: " .. e)
    perm = torch.randperm(yt:size(1))
    loss = 0
    
    for i = 1, yt:size(1) do
        item = perm[{i}]
        
        y = model:forward(x[{{item}}])

        print(yt[{item}], y[1][1], y[1][2])

        local err=criterion:forward(y, yt[{item}])
        loss = loss + err
        local gradCriterion = criterion:backward(y,yt[{item}]);

        model:zeroGradParameters()
        model:backward(x[{{item}}], gradCriterion)
        model:updateParameters(0.01)
    end
    
    -- print("loss: " .. loss / yt:size(1))
end
