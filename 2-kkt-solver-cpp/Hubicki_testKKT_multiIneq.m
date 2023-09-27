% testKKT_multiIneq.m
% Christian Hubicki
% 2023-06-04

clc
clear
close all

%% Constrained Program Definition
% Define your decision vector
x = sym('x',[2 1])
x0 = [-0.9; -0.1];

% Define your cost function
f = x(1).^2 + x(2).^2

% Define your inequality constraints (g(x)<=0)
g = [x(1)-1; x(2)-0; -x(1)+1]


%% Solver Setup

% Count the number of decision variables
Nx = numel(x);
% Count the number of inequality constraints
Ng = numel(g);

% Symbolically Define our Lagrange multipliers
lam = sym('lam',[Ng 1]);

% Symbolically Define our KKT slack variables
s = sym('s',[Ng 1]);

% Define our Lagrange function (Lagrangian)
L = f + sum(lam.*(g + s.^2));

% Define the KKT conditions
dL_dx = jacobian(L,x).';
dL_dlam = jacobian(L,lam).';
dL_ds = jacobian(L,s).';
KKT = [dL_dx; dL_dlam; dL_ds];

% Define our vector of independent variables for the KKT Conditions
w = [x;lam;s];

% Define our Jacobian of the KKT with respect to all variables
J = jacobian(KKT,w);

%% Generate MATLAB functions for computation
f_fun = matlabFunction(f,'Vars',{x});
g_fun = matlabFunction(g,'Vars',{x});
KKT_fun = matlabFunction(KKT,'Vars',{x,lam,s});
J_fun = matlabFunction(J,'Vars',{x,lam,s});


%% Run Newton's Method Solver
lam0 = ones(size(lam));
s0 = ones(size(s));

w_current = [x0; lam0; s0];
for iter = 1:20

    KKT_current = KKT_fun(x0,lam0,s0);
    J_current = J_fun(x0,lam0,s0);
    
    
    w_current = w_current - inv(J_current)*KKT_current;
    cost = f_fun(x0);
    const_viol = max([0; g_fun(x0)]);

    disp(sprintf(['iter ' num2str(iter)  ...
        '\t cost: ' num2str(cost,'%.2e') ...
        '\t viol: ' num2str(const_viol,'%.2e') ...
        '\t KKT norm: ' num2str(norm(KKT_current),'%.2e')
        ]))

    x0 = w_current(1:Nx,:);
    lam0 = w_current(Nx+1:Nx+Ng,:);
    s0 = w_current(Nx+Ng+1:Nx+Ng+Ng,:);
    
    
    

%     pause
end

x_solution = x0
