nn = require 'nn'
npy4th = require 'npy4th'

-- TODO: move to conf
--wide = 4096
wide = 4096

-- input, model, training config
local conf = {
  -- input
  i = {
    num_samples = 30,
    frequency_width = 128,
    path = "spect/"
  },

  -- model
  m = {
    num_classes = 3,
    num_hidden = 1000
  },

  -- training
  t = {
    epochs = 30
  }
}

-- load data
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

-- TODO: Calculate this
s = 254 * 512 -- wide * conf.i.frequency_width / 4 / 2 / 2

-- model
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

criterion = nn.ClassNLLCriterion()

-- training
for e = 1, conf.t.epochs do
    local permutation = torch.randperm(yt:size(1))
    local loss = 0
    local item, y, err, grad
    
    for i = 1, conf.i.num_samples do
        item = permutation[{i}]
        y = model:forward(x[{{item}}])

        err = criterion:forward(y, yt[{item}])
        loss = loss + err
        grad = criterion:backward(y,yt[{item}]);

        model:zeroGradParameters()
        model:backward(x[{{item}}], grad)
        model:updateParameters(0.01)
    end
    
    print("epoch: " .. e .. ", loss: " .. loss/conf.i.num_samples )
end


-- prediction
local v_path = 'spect/989787.LOFI.mp3.npy'
local v_input = npy4th.loadnpy(v_path):transpose(1, 2):resize(wide, conf.i.frequency_width)
local v_prob = model:forward(v_input)
local v_mt, v_mi = v_prob:max(1)

print(v_mi)

v_path = 'spect/3322004.LOFI.mp3.npy'
v_input = npy4th.loadnpy(v_path):transpose(1, 2):resize(wide, conf.i.frequency_width)
v_prob = model:forward(v_input)
v_mt, v_mi = v_prob:max(1)

print(v_mi)
