require 'nn'
require 'image'
require 'optim'

exp = require 'experiment'

ds = require 'ds'

-- config
local conf = {
  -- input
  training = {
    frequency_width = 128,
    path = "train/",
    width = 512
  },

  validation = {
    frequency_width = 128,
    path = "valid/",
    width = 512
  },


  -- model
  m = {
    num_classes = 3,
    num_hidden = {256, 64},
    conv_size = {256, 128, 64},
    filters = {4,4,4},
    pooling = {4,2,2},
    classifier_dropout = {.5, .5}
  },

  -- training
  t = {
    epochs = 50,
    optim = {
      learningRate = 2e-3, 
      alpha = .95
    }
  }
}

-- load data
local training = ds.load(conf.training)
local validation = ds.load(conf.validation)

-- model
local m = conf.m
model = nn.Sequential()

model:add(nn.TemporalConvolution(conf.training.frequency_width, m.conv_size[1], m.filters[1]))
model:add(nn.ReLU())
model:add(nn.TemporalMaxPooling(m.pooling[1]))

model:add(nn.TemporalConvolution(m.conv_size[1], m.conv_size[2], m.filters[2]))
model:add(nn.ReLU())
model:add(nn.TemporalMaxPooling(m.pooling[2]))

model:add(nn.TemporalConvolution(m.conv_size[2], m.conv_size[3], m.filters[3]))
model:add(nn.ReLU())
model:add(nn.TemporalMaxPooling(m.pooling[3]))

global = nn.ConcatTable()
global:add(nn.Mean(2))
global:add(nn.Max(2))
global:add(nn.Min(2))
model:add(global)
model:add(nn.JoinTable(2))

-- classifier
model:add(nn.Linear(3 * m.conv_size[3], m.num_hidden[1]))
model:add(nn.Sigmoid())
model:add(nn.Dropout(m.classifier_dropout[1]))

model:add(nn.Linear(m.num_hidden[1], m.num_hidden[2]))
model:add(nn.Sigmoid())
model:add(nn.Dropout(m.classifier_dropout[2]))

model:add(nn.Linear(m.num_hidden[2], m.num_classes))
model:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()


print("Conf:")
print(conf)

print("")
print("Model:")
print(model)

print("")
print("Criterion:")
print(criterion)

local params, grad_params = model:getParameters()

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(training.classes)
local path = os.date("%y%m%d_%H%M%S", os.time()) .. '_exp.txt'
local w = exp.CsvWriter(path)
local exp = exp.Experiment({'epoch', 't_valid', 'v_valid'}, w)

print(path)

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
        -- print(y, yt, df_do)
        model:backward(x, df_do)

        -- update confusion
        confusion:add(y[{1}], yt)

        -- activate after mini-batching
        -- gradParameters:div(1)

        -- return f and df/dX
        return err, grad_params
      end

      local _, loss = optim.sgd(feval, params, conf.t.optim)
      -- print(grad_params:norm() / params:norm())

      total_loss = total_loss + loss[1]
    end

    -- calculate validation set
    validation_confusion = optim.ConfusionMatrix(training.classes)
    for i = 1, validation.count do
        x = validation.x[{{i}}]
        yt = validation.yt[{i}]
        y = model:forward(x)

        validation_confusion:add(y[{1}], yt)
    end

    local weights = model:get(1):parameters()[1]:clone() --:resize(1, 256, 512)
    image.save("tmp/weights" .. e .. ".png", weights:div(weights:mean()))
    print("")
    print("----------")
    print("epoch: " .. e .. ", loss: " .. total_loss/training.count)
    print(confusion)
    print(validation_confusion)

  exp:t{e, 1-confusion.totalValid, 1-validation_confusion.totalValid}

    confusion:zero()
    validation_confusion:zero()
end
