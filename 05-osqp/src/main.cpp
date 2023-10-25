#include <OsqpEigen/OsqpEigen.h>

#include "symbol_vector.hpp"
#include "osqp_constraints.hpp"
#include "osqp_util.hpp"

#define NUMBER_OF_NODES 2

#define HORIZON_TIME 1.0F
#define DELTA_TIME HORIZON_TIME / float(NUMBER_OF_NODES - 1)

using namespace boylan;

matrix P_matrix(const int n)
{
    matrix P(n, n);
    for (int r = 0; r < n; r++)
    {
        for (int c = 0; c < n; c++)
        {
            if (r == c)
            {
                P(r, c) = 2.0;
            }
            else
            {
                P(r, c) = 0.0;
            }
        }
    }
    return P;
}

matrix q_vector(const int n)
{
    matrix q(n, 1);
    for (int i = 0; i < n; i++)
    {
        q(i, 0) = 0.0;
    }
    return q;
}

osqp_constraints equality_constraints(const symbol_vector x)
{
    osqp_constraints eq_con(x, 2 * NUMBER_OF_NODES);

    const float x0 = 1.0;
    const float v0 = 1.0;

    // Boundary constraints
    eq_con[0] = {x[0], x0, x0};
    eq_con[NUMBER_OF_NODES] = {x[NUMBER_OF_NODES], v0, v0};

    // Dynamics constraints
    for (int i = 1; i < NUMBER_OF_NODES; i++)
    {
        const int xi = i;
        const int vi = NUMBER_OF_NODES + i;
        const int ui = 2 * NUMBER_OF_NODES + i;
        eq_con[xi] = {x[xi] - x[xi - 1] - x[vi - 1] * DELTA_TIME, 0.0, 0.0};
        eq_con[vi] = {x[vi] - x[vi - 1] - x[ui - 1] * DELTA_TIME, 0.0, 0.0};
    }

    return eq_con;
}

int main()
{

    std::cout << "OSQP Version:" << osqp_version() << "\n";

    symbol_vector x("x", NUMBER_OF_NODES);
    symbol_vector v("v", NUMBER_OF_NODES);
    symbol_vector u("u", NUMBER_OF_NODES - 1);

    symbol_vector w = x.append(v).append(u);

    osqp_constraints h_x = equality_constraints(w);

    const int n = 3 * NUMBER_OF_NODES - 1;
    const int m = 2 * NUMBER_OF_NODES;
    matrix P = P_matrix(n);
    matrix q = q_vector(n);
    matrix A = h_x.get_A();
    matrix lb = h_x.get_l();
    matrix ub = h_x.get_u();

    OSQPCscMatrix Pcsc, Acsc;
    toCSC(P, &Pcsc);
    toCSC(A, &Acsc);

    std::vector<float> qvec = toVector(q),
                       lbvec = toVector(lb),
                       ubvec = toVector(ub);

    OSQPSettings settings;
    osqp_set_default_settings(&settings);

    OSQPSolver *solver = NULL;
    osqp_setup(&solver, &Pcsc, qvec.data(), &Acsc, lbvec.data(), ubvec.data(), m, n, &settings);

    return EXIT_SUCCESS;
}