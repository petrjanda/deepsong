require 'nn'
require 'image'
require 'optim'

ds = require 'ds'

-- config
local conf = {
  -- input
  training = {
    frequency_width = 128,
    path = "train/",
    width = 256
  },

  -- model
  m = {
    num_classes = 2,
    num_hidden = {512, 256}
  },

  -- training
  t = {
    epochs = 75,
    optim = {
      learningRate = 2e-3, 
      alpha = 0.95 
    }
  }
}

-- load data
local training = ds.load(conf.training)

-- model
model = nn.Sequential()

model:add(nn.TemporalConvolution(conf.training.frequency_width, 256, 4))
model:add(nn.ReLU())
model:add(nn.TemporalMaxPooling(4))

model:add(nn.TemporalConvolution(256, 256, 4))
model:add(nn.ReLU())
model:add(nn.TemporalMaxPooling(2))

model:add(nn.TemporalConvolution(256, 512, 4))
model:add(nn.ReLU())
model:add(nn.TemporalMaxPooling(2))

global = nn.ConcatTable()
global:add(nn.Mean(2))
global:add(nn.Max(2))
model:add(global)
model:add(nn.JoinTable(2))

-- classifier
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

local params, grad_params = model:getParameters()

-- training
classes = {1,2} -- ,3,4,5,6,7,8}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)
 
for e = 1, conf.t.epochs do
    local permutation = torch.randperm(training.count)
    local total_loss = 0
    local x, y, err, grad
    
    for i = 1, training.count do
      item = permutation[{i}]

      local feval = function(x)
        -- get new parameters
        if x ~= params then
          params:copy(x)
        end

        -- reset gradients
        grad_params:zero()

        -- evaluate
        x = training.x[{{item}}]
        yt = training.yt[{item}]

        local y = model:forward(x)
        local err = criterion:forward(y, yt)

        -- estimate df/dW
        local df_do = criterion:backward(y, yt)
        model:backward(x, df_do)

        -- update confusion
        confusion:add(y[{1}], yt)

        -- activate after mini-batching
        -- gradParameters:div(1)

        -- return f and df/dX
        return err, grad_params
      end

      local _, loss = optim.rmsprop(feval, params, conf.t.optim)
  
      total_loss = total_loss + loss[1]
    end
    
    -- validate()
    local weights = model:get(1):parameters()[1]:clone():resize(1, 256, 512)
    image.save("tmp/weights" .. e .. ".png", weights:div(weights:mean()))
    print("epoch: " .. e .. ", loss: " .. total_loss/training.count)
    print(confusion)

    confusion:zero()
end
