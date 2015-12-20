function loadDs(conf) 
  c = 0
  for a in paths.iterdirs(conf.i.path) do
      for s in paths.iterfiles(conf.i.path .. a) do
           c = c + 1
      end
  end

  local x = torch.Tensor(c, wide, conf.i.frequency_width)
  local yt = torch.Tensor(c)

  i = 1
  for a in paths.iterdirs(conf.i.path) do
      for s in paths.iterfiles(conf.i.path .. a) do
          input = npy4th.loadnpy(conf.i.path .. a .. '/' .. s):transpose(1, 2)
          w = math.floor((input:size(1) - wide) / 2)

          x[i] = input[{{w, w + wide - 1}, {}}]
          yt[i] = a

          i = i + 1
      end
  end

  return c, x, yt
end
