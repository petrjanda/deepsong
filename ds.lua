npy4th = require 'npy4th'

local function load(conf) 
  local count = 0
  for s in paths.iterfiles(conf.path) do
    label = tonumber(string.sub(s, 1, 1))

    if(label == 1 or label == 2) then
      count = count + 1
    end
  end

  local x = torch.Tensor(count, conf.width, conf.frequency_width)
  local yt = torch.Tensor(count)

  i = 1
  for s in paths.iterfiles(conf.path) do
    input = npy4th.loadnpy(conf.path .. '/' .. s)
    w = math.floor((input:size(1) - conf.width) / 2)

    label = tonumber(string.sub(s, 1, 1))

    if(label == 1 or label == 2) then
      x[i] = input[{{w+1, w + conf.width}, {}}]
      yt[i] = label

      i = i + 1
    end
  end

  return {count=count, x=x, yt=yt}
end

return {
  load = load
}
