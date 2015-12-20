npy4th = require 'npy4th'

local function l(conf) 
  c = 0
  for a in paths.iterfiles(conf.path) do
    c = c + 1
  end

  local x = torch.Tensor(c, conf.width, conf.frequency_width)
  local yt = torch.Tensor(c)

  i = 1
  for s in paths.iterfiles(conf.path) do
    input = npy4th.loadnpy(conf.path .. '/' .. s)
    w = math.floor((input:size(1) - conf.width) / 2)

    label = tonumber(string.sub(s, 1, 1))

    x[i] = input[{{w+1, w + conf.width}, {}}]
    yt[i] = label

    i = i + 1
  end

  return {count=c, x=x, yt=yt}
end

return {
  load = l
}
