#include <iostream>
#include <convex_optimization/index_vector.hpp>

using namespace boylan;

int main()
{

    std::cout << "Hello world!\n";

    auto x = IndexVector("x", 2);

    std::cout << x.i(0) << "\n";
    std::cout << x.i(1) << "\n";

    ex objective = x.i(0)*x.i(0) + x.i(1)*x.i(1);

    return EXIT_SUCCESS;
}