npy4th = require 'npy4th'
require 'set'

local function load(conf)
  local count = 0
  for s in paths.iterfiles(conf.path) do
    count = count + 1
  end

  local classes = {}
  local files = {}

  i = 1
  for s in paths.iterfiles(conf.path) do
    input = npy4th.loadnpy(conf.path .. '/' .. s)
    w = math.floor((input:size(1) - conf.width) / 2)

    label = tonumber(string.sub(s, 1, 1))
    classes = Set.union(classes, Set.new{label})

    table.insert(files, s)

    i = i + 1
  end

  return {count=count, files=files, classes=Set.keys(classes)}
end

local function batch(ds, i, size, conf)
  local from = i * size
  local to = (i+1) * size
  local count = size 

  local x = torch.Tensor(count, conf.width, conf.frequency_width)
  local yt = torch.Tensor(count)

  local i = 1
  for j=from, to - 1 do
    local s = ds.files[j]

    input = npy4th.loadnpy(conf.path .. '/' .. s)
    w = math.floor((input:size(1) - conf.width) / 2)

    label = tonumber(string.sub(s, 1, 1))

    x[i] = input[{{w+1, w + conf.width}, {}}]
    yt[i] = label

    i = i + 1
  end

  return x, yt
end

return {
  load = load,
  batch = batch
}
