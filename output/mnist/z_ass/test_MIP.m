k = 30;
n = 300;
m = 2;
A = zeros(m * k * (k-1) + n + k, n * k + 1);

b = zeros(m * k * (k-1) + n + k, 1);
f = zeros(n * k + 1, 1);
f(n * k + 1, 1) = 1;
x = intlinprog(f,intcon,A,b);
lb = zeros(n * k + 1, 1);
ub = ones(n * k + 1, 1);
ub(n*k + 1, 1) = Inf;


