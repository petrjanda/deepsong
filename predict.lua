require 'nn'
require 'optim'
ds = require 'ds'

local conf = {
  validation = {
    frequency_width = 128,
    path = arg[1],
    width = 512
  }
}

model = torch.load(arg[2])
model:evaluate()

local validation = ds.load(conf.validation)
local validation_confusion = optim.ConfusionMatrix(validation.classes)
local labels = {"Deep House","Nu Disco"}

local j = 0
for i = 1, validation.count do
    x = validation.x[{{i}}]
    yt = validation.yt[{i}]
    y = model:forward(x)

    _, max = torch.max(y, 2)


    if(yt ~= max[1][1]) then
      print(validation.files[i], labels[yt], '-->', labels[max[1][1]])
      j = j + 1
    end

    validation_confusion:add(y[{1}], yt)
end

print(j, '/', validation.count)
-- print(validation_confusion)
