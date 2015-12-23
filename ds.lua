npy4th = require 'npy4th'
require 'set'

local function load(conf)
  local count = 0
  for s in paths.iterfiles(conf.path) do
    count = count + 1
  end

  local x = torch.Tensor(count, conf.width, conf.frequency_width)
  local yt = torch.Tensor(count)
  local classes = {}

  i = 1
  for s in paths.iterfiles(conf.path) do
    input = npy4th.loadnpy(conf.path .. '/' .. s)
    w = math.floor((input:size(1) - conf.width) / 2)

    label = tonumber(string.sub(s, 1, 1))

    x[i] = input[{{w+1, w + conf.width}, {}}]
    yt[i] = label

    classes = Set.union(classes, Set.new{label})

    -- print(type_count())

    i = i + 1
  end

  return {count=count, x=x, yt=yt, classes=Set.keys(classes)}
end

function count_all(f)
  local seen = {}
  local count_table
  count_table = function(t)
    if seen[t] then return end
    f(t)
    seen[t] = true
    for k,v in pairs(t) do
      if type(v) == "table" then
        count_table(v)
      elseif type(v) == "userdata" then
        f(v)
      end
    end
  end
  count_table(_G)
end


function type_count()
  local counts = {}
  local enumerate = function (o)
    local t = type(o)
    counts[t] = (counts[t] or 0) + 1
  end
  count_all(enumerate)
  return counts
end


return {
  load = load
}
