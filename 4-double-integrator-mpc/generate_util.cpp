#include <sstream>
#include <ginac/ginac.h>

using namespace GiNaC;

std::string generate_HEADER(const int num_obj, const int num_vars, const int num_coeffs)
{
    std::ostringstream oss;
    oss 
    << "#ifndef KKT_SOLVER_GENERATED_LOOKUP_CU_\n"
    << "#define KKT_SOLVER_GENERATED_LOOKUP_CU_\n"
    << "\n"
    << "#define NUM_OBJECTIVES " << num_obj << "\n"
    << "#define NUM_VARIABLES " << num_vars << "\n"
    << "#define NUM_COEFFICIENTS " << num_coeffs << "\n"
    << "\n"
    << "typedef float (*VAL_PTR)(float*, float*);\n"
    << "\n"
    << "#define GET_WI(w) WI_EVALUATE<<<NUM_VARIABLES, NUM_OBJECTIVES>>>(w)\n"
    << "#define GET_KKT(KKT,w,c) KKT_EVALUATE<<<NUM_VARIABLES, NUM_OBJECTIVES>>>(KKT,w,c)\n"
    << "#define GET_J(J,w,c) J_EVALUATE<<<NUM_VARIABLES*NUM_VARIABLES, NUM_OBJECTIVES>>>(J,w,c)\n"
    << "\n"
    << "#define FORMAT_KKT(KKT,d_KKT) KKT_FORMAT<<<NUM_OBJECTIVES, 1>>>(KKT, d_KKT)\n"
    << "#define FORMAT_J(J,d_J) J_FORMAT<<<NUM_OBJECTIVES, 1>>>(J, d_J)\n"
    << "\n"
    ;
    return oss.str();
}

std::string generate_WI_P_N(const float val, const int np, const int nv)
{
    std::ostringstream oss;
    oss
    << std::setprecision(5) << std::fixed
    << "#define WI_" << np << "_" << nv << " " << val << "F\n"
    ;
    return oss.str(); 
}

std::string generate_COST_P(const ex expression, const int np)
{
    std::ostringstream oss;
    oss
    << "\n"
    << "__device__\n"
    << "float COST_" << np << "(float * w, float * c)\n"
    << "{\n"
    << "return " << csrc_float << expression << ";\n"
    << "}\n"
    << "\n"
    ;
    return oss.str();
}

std::string generate_KKT_P_N(const ex expression, const int np, const int nv)
{
    std::ostringstream oss;
    oss
    << "__device__\n"
    << "float KKT_" << np << "_" << nv << "(float * w, float * c)\n"
    << "{\n"
    << "return " << csrc_float << expression << ";\n"
    << "}\n"
    << "\n"
    ;
    return oss.str();
}

std::string generate_J_P_N_M(const ex expression, const int np, const int row, const int col)
{
    std::ostringstream oss;
    oss
    << "__device__\n"
    << "float J_" << np << "_" << row << "_" << col << "(float * w, float * c)\n"
    << "{\n"
    << "return " << csrc_float << expression << ";\n"
    << "}\n"
    << "\n"
    ;
    return oss.str();
}

std::string generate_WI_LOOKUP(const int num_probs, const int num_vars)
{
        std::ostringstream oss;
    oss
    << "__constant__\n"
    << "float WI_LOOKUP[" << num_probs*num_vars << "] =\n"
    << "{\n";
    for (int p = 0; p < num_probs; p++)
    {
        for (int v = 0; v < num_vars; v++)
        {
            oss << "WI_" << p << "_" << v << ", ";
        }
        oss << "\n";
    }
    oss
    << "};\n"
    << "\n"
    ;
    return oss.str();
}

std::string generate_COST_LOOKUP(const int num_probs)
{
    std::ostringstream oss;
    oss
    << "__constant__\n"
    << "VAL_PTR COST_LOOKUP[" << num_probs << "] =\n"
    << "{\n";
    for (int p = 0; p < num_probs; p++)
    {
        oss << "COST_" << p << ", ";
        oss << "\n";
    }
    oss
    << "};\n"
    << "\n"
    ;
    return oss.str();
}

std::string generate_KKT_LOOKUP(const int num_probs, const int num_vars)
{
    std::ostringstream oss;
    oss
    << "__constant__\n"
    << "VAL_PTR KKT_LOOKUP[" << num_probs*num_vars << "] =\n"
    << "{\n";
    for (int np = 0; np < num_probs; np++)
    {
        for (int nv = 0; nv < num_vars; nv++)
        {
            oss << "KKT_" << np << "_" << nv << ", ";
        }
        oss << "\n";
    }
    oss
    << "};\n"
    << "\n"
    ;
    return oss.str();
}

std::string generate_J_LOOKUP(const int num_probs, const int num_vars)
{
    std::ostringstream oss;
    oss
    << "__constant__\n"
    << "VAL_PTR J_LOOKUP[" << num_probs*num_vars*num_vars << "] =\n"
    << "{\n";
    for (int np = 0; np < num_probs; np++)
    {
        for (int nvr = 0; nvr < num_vars; nvr++)
        {
            for (int nvc = 0; nvc < num_vars; nvc++)
            {
                oss << "J_" << np << "_" << nvr << "_" << nvc << ", ";
            }
            oss << "\n";
        }
    }
    oss
    << "};\n"
    << "\n"
    ;
    return oss.str();
}

std::string generate_WI_EVALUATE()
{
    std::ostringstream oss;
    oss
    << "__global__\n"
    << "void WI_EVALUATE(float * w)\n"
    << "{\n"
    << "const int idx = blockDim.x*blockIdx.x + threadIdx.x;\n"
    << "w[idx] = WI_LOOKUP[idx];\n"
    << "};\n"
    << "\n"
    ;
    return oss.str();
}

std::string generate_KKT_EVALUATE()
{
    std::ostringstream oss;
    oss
    << "__global__\n"
    << "void KKT_EVALUATE(float * KKT, float * w, float * c)\n"
    << "{\n"
    << "const int idx = blockDim.x*blockIdx.x + threadIdx.x;\n"
    << "KKT[idx] = KKT_LOOKUP[idx](w, c);\n"
    << "};\n"
    << "\n"
    ;
    return oss.str();
}

std::string generate_KKT_FORMAT()
{
    std::ostringstream oss;
    oss
    << "__global__\n"
    << "void KKT_FORMAT(float ** KKT, float * d_KKT)\n"
    << "{\n"
    << "const int idx = blockDim.x*blockIdx.x + threadIdx.x;\n"
    << "KKT[idx] = &d_KKT[NUM_VARIABLES*idx];\n"
    << "};\n"
    << "\n"
    ;
    return oss.str();
}

std::string generate_J_EVALUATE()
{
    std::ostringstream oss;
    oss
    << "__global__\n"
    << "void J_EVALUATE(float * J, float * w, float * c)\n"
    << "{\n"
    << "const int idx = blockDim.x*blockIdx.x + threadIdx.x;\n"
    << "J[idx] = J_LOOKUP[idx](w, c);\n"
    << "};\n"
    << "\n"
    ;
    return oss.str();
}

std::string generate_J_FORMAT()
{
    std::ostringstream oss;
    oss
    << "__global__\n"
    << "void J_FORMAT(float ** J, float * d_J)\n"
    << "{\n"
    << "const int idx = blockDim.x*blockIdx.x + threadIdx.x;\n"
    << "J[idx] = &d_J[NUM_VARIABLES*NUM_VARIABLES*idx];\n"
    << "};\n"
    << "\n"
    ;
    return oss.str();
}

std::string generate_ENDER()
{
    std::ostringstream oss;
    oss 
    << "#endif"
    ;
    return oss.str();
}