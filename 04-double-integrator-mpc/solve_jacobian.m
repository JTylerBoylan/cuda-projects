clear
clc
close all
%%

N_nodes = 1;
HorizonTime = 1.0;
DeltaTime = HorizonTime / N_nodes;

N_states = 3*(N_nodes + 1);
N_constraints = 4*(N_nodes + 1);

x = sym('x',[N_nodes+1 1]);
v = sym('v',[N_nodes+1 1]);
u = sym('u',[N_nodes+1 1]);
lam = sym('lam',[N_constraints 1]);
s = sym('s',[N_constraints 1]);

state = [x; v; u];

w = [state; lam; s];

F = sum(state.^2);

x0 = 0.5;
v0 = 0.5;

G(1) = state(1) - x0;
G(2) = x0 - state(1);
G(3) = state(N_nodes+1) - v0;
G(4) = v0 - state(N_nodes+1);

for i = 1:N_nodes
    xi = i + 1;
    vi = N_nodes + i + 1;
    ui = 2*N_nodes + i + 1;
    ci = 4*i + 1;
    G(ci) = state(xi) - state(xi - 1) - state(vi - 1)*DeltaTime;
    G(ci+1) = -state(xi) + state(xi - 1) + state(vi - 1)*DeltaTime;
    G(ci+2) = state(vi) - state(vi - 1) - state(ui - 1)*DeltaTime;
    G(ci+3) = -state(vi) + state(vi - 1) + state(ui - 1)*DeltaTime;
end

L = F + sum(lam.*(G' + s.^2));