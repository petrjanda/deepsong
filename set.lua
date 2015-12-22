Set = {}

function Set.new (t)
  local set = {}
  for _, l in ipairs(t) do set[l] = true end
  return set
end

function Set.union (a,b)
  local res = Set.new{}

  for k in pairs(a) do res[k] = true end
  for k in pairs(b) do res[k] = true end

  return res
end

function Set.intersection (a,b)
  local res = Set.new{}

  for k in pairs(a) do
    res[k] = b[k]
  end

  return res
end

function Set.keys(a)
  local keys = {}
  for k in pairs(a) do table.insert(keys, k) end

  return keys
end
