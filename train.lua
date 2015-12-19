nn = require 'nn'
npy4th = require 'npy4th'

--wide = 4096
wide = 128


-- input, model, training config
local conf = {
  -- input
  i = {
    num_samples = 20,
    frequency_width = 128,
    path = "spect/"
  },

  -- model
  m = {
    num_classes = 2,
    num_hidden = 1000
  },

  -- training
  t = {
    epochs = 125
  }
}


x = torch.Tensor(conf.i.num_samples, wide, conf.i.frequency_width)
yt = torch.Tensor(conf.i.num_samples)

i = 1
for a in paths.iterdirs(conf.i.path) do
    for s in paths.iterfiles(conf.i.path .. a) do
        input = npy4th.loadnpy(conf.i.path .. a .. '/' .. s):transpose(1, 2):resize(wide, conf.i.frequency_width)

        x[i] = input
        yt[i] = a

        i = i + 1
    end
end

s = 6 * 512 -- wide * conf.i.frequency_width / 4 / 2 / 2

model = nn.Sequential()

model:add(nn.TemporalConvolution(conf.i.frequency_width, 256, 4))
model:add(nn.ReLU())
model:add(nn.TemporalMaxPooling(1,4))

model:add(nn.TemporalConvolution(256, 256, 4))
model:add(nn.ReLU())
model:add(nn.TemporalMaxPooling(1,2))

model:add(nn.TemporalConvolution(256, 512, 4))
model:add(nn.ReLU())
model:add(nn.TemporalMaxPooling(1,2))

model:add(nn.Reshape(s))
model:add(nn.Linear(s, conf.m.num_hidden))
model:add(nn.Sigmoid())
model:add(nn.Linear(conf.m.num_hidden, conf.m.num_classes))
model:add(nn.LogSoftMax())

criterion= nn.ClassNLLCriterion()

for e = 1, conf.t.epochs do
    local perm = torch.randperm(yt:size(1))
    local loss = 0
    
    for i = 1, conf.i.num_samples do
        item = perm[{i}]
        y = model:forward(x[{{item}}])

        local err=criterion:forward(y, yt[{item}])
        loss = loss + err
        local gradCriterion = criterion:backward(y,yt[{item}]);

        model:zeroGradParameters()
        model:backward(x[{{item}}], gradCriterion)
        model:updateParameters(0.01)
    end
    
    print("epoch: " .. e .. ", loss: " .. loss/conf.i.num_samples )
end

p = model:forward(x[{1}])

local mt, mi = p:max(1)
print(mt, mi[0])
