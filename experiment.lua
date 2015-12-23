local class = require 'class'

local Experiment = class('Experiment')

function Experiment:__init(variables, w)
  self.size = #variables

  self.w = w
  self.w:writeHeaders(variables)
end

function Experiment:t(values)
  assert(#values == self.size, "Need " .. self.size .. " values, got " .. #values)

  self.w:write(values)
end

local CsvWriter = class('FileWriter')

function CsvWriter:__init(path)
  self.file = io.open(path, "w")
end

function CsvWriter:close()
  self.file:close()
end

function CsvWriter:writeHeaders(headers)
  self.file:write(table.concat(headers,",") .. '\n')
end

function CsvWriter:write(values)
  self.file:write(table.concat(values,",") .. '\n')
  self.file:flush()
end

return {
  Experiment = Experiment,
  CsvWriter = CsvWriter
}
