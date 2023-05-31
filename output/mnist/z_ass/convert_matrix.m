path = "./";
file = path + "MIP_z_n300part10.csv";
Tbl = readtable(file);
a = Tbl{:,:};
k = size(a, 1); n = size(a, 2);
disp(k);
disp(n);
u = 1; b= zeros(n * k , 1);
for j = 1:n
    for i = 1:k
        b(u) = a(i,j);
        u = u + 1;
    end
end
c = zeros(k,n);
for j = 1:k
    for i = 1:n
        c(j,i) = b((j-1) * n + i, 1);
    end
end
disp(sum(c, 1));%y = 2.760254e+00


