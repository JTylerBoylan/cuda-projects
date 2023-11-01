clc
clear
close all
%%

% Params
Nns = 4;
Ns = 3;
Nn = Nns * Ns;

q = sym('q', [Nn*4 1]); % state
u = sym('u', [Nn*2 1]); % control
sx = sym('sx', [Ns 1]); % step x
sy = sym('sy', [Ns 1]); % step y
rho_step = sym('rho_step', [Nn 1]); % slack step
rho_avoid = sym('rho_avoid', [Nn 1]); % slack avoid
rho_input = sym('rho_input', [Nn 1]); % slack input

z = [q; u; sx; sy; rho_step; rho_avoid; rho_input]; % decision variables

sx_ref = sym('sx_ref', [Ns 1]); % reference step x
Wx_step = sym('wx_step', [Ns 1]);
Jx_step = sum(Wx_step.*(sx - sx_ref).^2);

sy_ref = sym('sy_ref', [Ns 1]); % reference step y
Wy_step = sym('wy_step', [Nn 1]);
Jy_step = sum(Wy_step.*(rho_step).^2);

q_ref = sym('q_ref', [Nn*4 1]); % reference state
W_target = sym('w_target', [Nn*4 1]);
J_target = sum(W_target.*(q - q_ref).^2);

syms p_footx p_footy
u_ref = p_footx*ones(2*Nn,1);
for i = 1:Nn
    step_i = floor((i-1) / Nns);
    u_ref(2*i-1) = p_footx + sum(sx(1:step_i));
    u_ref(2*i) = p_footy + sum(sy(1:step_i));
end

W_effort = sym('w_effort', [2*Nn 1]);
J_effort = sum(W_effort.*(u - u_ref).^2);

W_avoid = sym('w_avoid', [Nn 1]);
J_avoid = sum(W_avoid.*(rho_avoid).^2);

W_input = sym('w_input', [Nn 1]);
J_input = sum(W_input.*(rho_input).^2);

J = Jx_step + Jy_step + J_target + J_effort + J_avoid + J_input;

hess = hessian(J, z);
grad = subs(gradient(J, z), z, zeros(size(z)));
