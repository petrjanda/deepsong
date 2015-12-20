nn = require 'nn'
npy4th = require 'npy4th'

require 'ds'

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
local num_training_samples, x, yt = loadDs(conf)

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

-- model:add(nn.Reshape(12))

global = nn.ConcatTable()
global:add(nn.Mean(2))
global:add(nn.Max(2))

model:add(global)
model:add(nn.JoinTable(2))

model:add(nn.Linear(2 * 512, conf.m.num_hidden))
model:add(nn.Sigmoid())
model:add(nn.Linear(conf.m.num_hidden, conf.m.num_classes))
model:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()

-- validate
function validate() 
  function sample(path, yt)
    local input = npy4th.loadnpy(path):transpose(1, 2)
    w = math.floor((input:size(1) - wide) / 2)
    input = input[{{w, w + wide - 1}, {}}]

    local prob = model:forward(input)
    local loss = criterion:forward(prob, yt)
    local mt, mi = prob:max(1)

    return mi[1], loss
  end

  
  local j, jl = sample('spect/989787.LOFI.mp3.npy',1) -- jamie
  local m, ml = sample('spect/moved.mp3.npy',2) -- move d
  local d, dl = sample('spect/dixon.mp3.npy',3)

  print(j .. 1,m .. 2,d .. 3,(jl+ml+dl)/3)
end

-- training
for e = 1, conf.t.epochs do
    local permutation = torch.randperm(yt:size(1))
    local loss = 0
    local item, y, err, grad
    
    for i = 1, num_training_samples do
        item = permutation[{i}]

        -- forwardprop
        y = model:forward(x[{{item}}])
        err = criterion:forward(y, yt[{item}])
        loss = loss + err

        -- reset gradients
        model:zeroGradParameters()
        
        -- backprop
        grad = criterion:backward(y,yt[{item}]);
        model:backward(x[{{item}}], grad)
        model:updateParameters(0.01)
    end
    
    -- validate()
    print("epoch: " .. e .. ", loss: " .. loss/num_training_samples)
end
