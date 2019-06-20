t = [1, 2; 4, 2]
[v, d] = eig(t)
v
v = fliplr(v)
v(:, 1:1)