#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <vector>
#include <assert.h>
#include <ginac/ginac.h>

using namespace GiNaC;

void print_j(const matrix mat)
{
    for (int i = 0; i < mat.rows(); i++)
    {
        std::cout << mat(i,0) << "\n";
    }
}