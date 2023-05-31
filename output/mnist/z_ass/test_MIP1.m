
path = "./";
point_file = path + "wPCA_MLP2_G10_partition_noniid90_nclient300.csv";

%point_file = "wPCA_MLP2_G10_partition_noniid90_nclient300.csv"

path = "./";
%x0_file = "MIP_z_n300part10.csv";

x0_file = "1.MIP_z_n300part10.csv";
Tbl = readtable(path + x0_file);
a = Tbl{:,:};
%k = size(a, 1); n = size(a, 2);
%disp(k); disp(n);
x0= zeros(n * k + 1 , 1);
for u = 1:size(a,1)
    x0(a(u, 1)) = a(u, 2);
end
%u = 1
%for j = 1:n
%    for i = 1:k
%        x0(u,1) = a(i,j);
%        u = u + 1;
%    end
%end

%x0(n * k + 1, 1) =  2.760254e+00;
Tbl = readtable(point_file);
w = Tbl{:,2:3};
mean_c = mean(w);
 
%k = 2;
%n = 20; %size(v, 1); %#rows = 300 clients for this case;
m = size(w, 2); % #columns = #dimensions (=2 for this case)
v = w;
for i = 1: n
    v(i,:) = v(i, :) - mean_c;
end
disp("mean_w, w_sum, v_sum: ");
disp(mean_c);
disp(sum(w));
disp(sum(v));

A = sparse(m * k * (k-1) + k, n * k + 1);

b = zeros(m * k * (k-1) + k, 1);
f = zeros(n * k + 1, 1);
f(n * k + 1, 1) = 1;

lb = zeros(n * k + 1, 1); % x>=0
ub = ones(n * k + 1, 1);  % x <= 1
ub(n*k + 1, 1) = Inf;     % 0 <= y <= Inf

%eq constraint
Aeq = sparse(n, n * k + 1);
beq = ones(n, 1);

for i = 1:n
    for j = 1:k
        Aeq(i, (j - 1) * n + i) = 1;
    end
end

%ieq constraint: non-empty cluster
b(1:k, 1) = -1;
c = 1; % index for constraint
for j = 1:k
    for i = 1:n
        A(c, (j-1) * n +i ) = -1;
    end
    c = c + 1;
end

%ieq constraint: 
for l = 1:m
    for j1 = 1:k-1
        for j2 = j1+1:k
            for i = 1:n
                A(c, (j1 - 1) * n + i) = v(i,l);
                A(c, (j2 - 1) * n + i) = - v(i,l);                
            end
            A(c, n * k + 1) = -1;
            c = c + 1;
        end 
    end
end
%ieq constraint: 

for l = 1:m
    for j1 = 1:k-1
        for j2 = j1+1:k
            for i = 1:n
                A(c, (j1 - 1) * n + i) = - v(i,l);
                A(c, (j2 - 1) * n + i) =  v(i,l);                
            end
            A(c, n * k + 1) = -1;
            c = c + 1;
        end 
    end
end


intcon = 1:(n * k);

options = optimoptions('intlinprog','MaxTime',10, 'MaxNodes', 1e20, 'RelativeGapTolerance', 0.02);

[x,fval,exitflag,output] = intlinprog(f,intcon,A,b,Aeq,beq,lb,ub, x0, options);

disp('sum_x : ');
disp(sum(x))

%xx = reshape(x(1:n*k), k, n);
%writematrix(xx, path + "MIP_z_n" + num2str(n) + "part" + num2str(k) + ".csv")

%disp("z = ")
%disp(z)

%disp(sum(xx, 2));
%disp(sum(xx, 1))

xx = zeros(n + 1, 2);
j = 1;
for i = 1: n* k + 1
    if x(i,1) > 0
        xx(j, 1) = i; xx(j,2) = x(i,1);
        j = j + 1;
    end
end

%writematrix(xx, path + "1." + x0_file);

%check feasiblity

count = 0;
for l = 1:m
    for j1 = 1:k-1
        for j2 = j1+1:k
            s = 0;
            for i = 1:n
                s = s +  (x((j1 - 1) * n + i) - x((j2 - 1) * n + i)) * v(i,l);                
            end
            if s > x(n * k + 1, 1)
                count = count + 1;
                disp(j1); disp(j2)
            end
        end 
    end
end



%ieq constraint: 
for l = 1:m
    for j1 = 1:k-1
        for j2 = j1+1:k
            s = 0;
            for i = 1:n
                s = s +  (-x((j1 - 1) * n + i) +x((j2 - 1) * n + i)) * v(i,l);                
            end
            if s > x(n * k + 1, 1)
                count = count + 1;
                disp(j1); disp(j2);disp(l);
            end
        end 
    end
end

disp("#violations: ")
disp(count)




 