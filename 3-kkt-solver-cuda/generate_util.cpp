#include <sstream>
#include <ginac/ginac.h>

using namespace GiNaC;

std::string generate_lookup_header(const int num_obj, const int num_vars, const int num_coeffs)
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
    << "typedef float (*intercept_ptr)(float*, float*);\n"
    << "\n"
    ;
    return oss.str();
}

std::string generate_cost_function(const ex expression, const int np)
{
    std::ostringstream oss;
    oss
    << "__device__\n"
    << "float COST_" << np << "(float * w, float * c)\n"
    << "{\n"
    << "return " << csrc_float << expression << ";\n"
    << "}\n"
    << "\n"
    ;
    return oss.str();
}

std::string generate_winitial_definition(const float val, const int np, const int nv)
{
    std::ostringstream oss;
    oss
    << std::setprecision(5) << std::fixed
    << "#define WI_" << np << "_" << nv << " " << val << "F\n"
    ;
    return oss.str(); 
}

std::string generate_expression_function(const ex expression, const int np, const int nv)
{
    std::ostringstream oss;
    oss
    << "__device__\n"
    << "float j_" << np << "_" << nv << "(float * w, float * c)\n"
    << "{\n"
    << "return " << csrc_float << expression << ";\n"
    << "}\n"
    << "\n"
    ;
    return oss.str();
}

std::string generate_lookup_intercept(const int num_prob, const int num_vars)
{
    std::ostringstream oss;
    oss
    << "__constant__\n"
    << "intercept_ptr LOOKUP_INTERCEPT[" << num_prob*num_vars << "] =\n"
    << "{\n";
    for (int p = 0; p < num_prob; p++)
    {
        for (int v = 0; v < num_vars; v++)
        {
            oss << "j_" << p << "_" << v << ", ";
        }
        oss << "\n";
    }
    oss
    << "};\n"
    << "\n"
    ;
    return oss.str();
}

std::string generate_lookup_initials(const int num_prob, const int num_vars)
{
        std::ostringstream oss;
    oss
    << "__constant__\n"
    << "float LOOKUP_INITIAL[" << num_prob*num_vars << "] =\n"
    << "{\n";
    for (int p = 0; p < num_prob; p++)
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

std::string generate_lookup_objective(const int num_prob)
{
    std::ostringstream oss;
    oss
    << "__constant__\n"
    << "intercept_ptr LOOKUP_OBJECTIVE[" << num_prob << "] =\n"
    << "{\n";
    for (int p = 0; p < num_prob; p++)
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

std::string generate_lookup_ender()
{
    std::ostringstream oss;
    oss 
    << "#endif"
    ;
    return oss.str();
}