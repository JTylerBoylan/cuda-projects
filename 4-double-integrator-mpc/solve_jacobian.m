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
G(2)