nn = require 'nn'
npy4th = require 'npy4th'

ds = require 'ds'

-- input, model, training config
local conf = {
  -- input
  i = {
    frequency_width = 128,
    path = "train/",
    width = 256
  },

  -- model
  m = {
    num_classes = 8,
    num_hidden = {512, 256}
  },

  -- training
  t = {
    epochs = 75
  }
}

-- load data
local num_training_samples, x, yt = ds.load(conf)

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

global = nn.ConcatTable()
global:add(nn.Mean(2))
global:add(nn.Max(2))
model:add(global)
model:add(nn.JoinTable(2))

model:add(nn.Linear(2 * 512, conf.m.num_hidden[1]))
model:add(nn.Sigmoid())

model:add(nn.Linear(conf.m.num_hidden[1], conf.m.num_hidden[2]))
model:add(nn.Sigmoid())

-- model:add(nn.Reshape(14 * 512))
-- model:add(nn.Sigmoid())
-- model:add(nn.Linear(14 * 512, conf.m.num_hidden))

model:add(nn.Linear(conf.m.num_hidden[2], conf.m.num_classes))
model:add(nn.LogSoftMax())


criterion = nn.ClassNLLCriterion()


print(model)

-- training
for e = 1, conf.t.epochs do
    local permutation = torch.randperm(num_training_samples)
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
