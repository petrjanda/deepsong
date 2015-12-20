require 'nn'
require 'image'
require 'optim'

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
local num_training_samples, dsx, dsyt = ds.load(conf)

-- model
model = nn.Sequential()

model:add(nn.TemporalConvolution(conf.i.frequency_width, 256, 4))
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


local optim_state = {
  learningRate = 2e-3, 
  alpha = 0.95 
}

local params, grad_params = model:getParameters()


-- training
classes = {1,2,3,4,5,6,7,8}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)
 
for e = 1, conf.t.epochs do
    local permutation = torch.randperm(num_training_samples)
    local tloss = 0
    local x, y, err, grad
    
    for i = 1, num_training_samples do
      item = permutation[{i}]

      local feval = function(x)
        -- get new parameters
        if x ~= params then
          params:copy(x)
        end

        -- reset gradients
        grad_params:zero()

        -- f is the average of all criterions
        local f = 0

       -- evaluate function for complete mini batch
          -- estimate f
          x = dsx[{{item}}]
          yt = dsyt[{item}]

          local output = model:forward(x)
          local err = criterion:forward(output, yt)
          f = f + err

          -- estimate df/dW
          local df_do = criterion:backward(output, yt)
          model:backward(x, df_do)

          -- update confusion
          confusion:add(output[{1}], yt)

       -- normalize gradients and f(X)
       -- gradParameters:div(1)
       -- f = f/1

       -- return f and df/dX
       return f, grad_params
    end

      local _, loss = optim.rmsprop(feval, params, optim_state)
  
      tloss = tloss + loss[1]
    end
    
    -- validate()
    local weights = model:get(1):parameters()[1]:clone():resize(1, 256, 512)
    image.save("tmp/weights" .. e .. ".png", weights:div(weights:mean()))
    print("epoch: " .. e .. ", loss: " .. tloss/num_training_samples)
    print(confusion)

    confusion:zero()
end
