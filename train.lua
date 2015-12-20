nn = require 'nn'
npy4th = require 'npy4th'

-- TODO: move to conf
--wide = 4096
wide = 512

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
    num_hidden = 300
  },

  -- training
  t = {
    epochs = 75
  }
}

-- load data
x = torch.Tensor(conf.i.num_samples, wide, conf.i.frequency_width)
yt = torch.Tensor(conf.i.num_samples)

i = 1
for a in paths.iterdirs(conf.i.path) do
    for s in paths.iterfiles(conf.i.path .. a) do
        input = npy4th.loadnpy(conf.i.path .. a .. '/' .. s):transpose(1, 2)
        w = math.floor((input:size(1) - wide) / 2)

        x[i] = input[{{w, w + wide - 1}, {}}]
        yt[i] = a

        i = i + 1
    end
end

-- TODO: Calculate this
s = 30 * 512 -- wide * conf.i.frequency_width / 4 / 2 / 2

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

-- validate
function validate() 
  function sample(path, yt)
    local input = npy4th.loadnpy(path):transpose(1, 2):resize(wide, conf.i.frequency_width)
    local prob = model:forward(input)
    local loss = criterion:forward(prob, yt)
    local mt, mi = prob:max(1)

    return mi[1], loss
  end

  
  local j, jl = sample('spect/989787.LOFI.mp3.npy',1) -- jamie
  local m, ml = sample('spect/moved.mp3.npy',2) -- move d
  local d, dl = sample('spect/dixon.mp3.npy',3)

  print(j,m,d,(jl+ml+dl)/3)
end

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
    
    validate()
    print("epoch: " .. e .. ", loss: " .. loss/conf.i.num_samples )
end


