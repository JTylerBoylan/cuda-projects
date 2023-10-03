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

F = sum(state.^2);

x0 = 0.5;
v0 = 0.5;

G(1) = state(0) - x0;
G(2) = x0 - state(0);
G(3) = state(N_nodes) - v0;
G(4) = v0 - state(N_nodes);

for i = 2:N_nodes
    xi = i;
    vi = N_nodes + i;
    ui = 2*N_nodes + i;
    ci = 4*(i-1) + 1;
    G(ci) = state(xi) - state(xi - 1) - state(vi - 1)*DeltaTime;
    G(ci+1) = -state(xi) + state(xi - 1) + state(vi - 1)*DeltaTime;
    G(ci+2) = state(vi) - state(vi - 1) - state(ui - 1)*DeltaTime;
    G(ci+3) = -state(vi) + state(vi - 1) + state(ui - 1)*DeltaTime;
end