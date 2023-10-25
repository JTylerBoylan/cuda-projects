#ifndef KKT_SOLVER_GENERATED_LOOKUP_CU_
#define KKT_SOLVER_GENERATED_LOOKUP_CU_

#define NUM_OBJECTIVES 3
#define NUM_VARIABLES 22
#define NUM_COEFFICIENTS 0

typedef float (*VAL_PTR)(float*, float*);

#define GET_WI(w) WI_EVALUATE<<<NUM_VARIABLES, NUM_OBJECTIVES>>>(w)
#define GET_KKT(KKT,w,c) KKT_EVALUATE<<<NUM_VARIABLES, NUM_OBJECTIVES>>>(KKT,w,c)
#define GET_J(J,w,c) J_EVALUATE<<<NUM_VARIABLES*NUM_VARIABLES, NUM_OBJECTIVES>>>(J,w,c)

#define FORMAT_KKT(KKT,d_KKT) KKT_FORMAT<<<NUM_OBJECTIVES, 1>>>(KKT, d_KKT)
#define FORMAT_J(J,d_J) J_FORMAT<<<NUM_OBJECTIVES, 1>>>(J, d_J)

#define WI_0_0 1.00000F
#define WI_0_1 1.00000F
#define WI_0_2 1.00000F
#define WI_0_3 1.00000F
#define WI_0_4 1.00000F
#define WI_0_5 1.00000F
#define WI_0_6 1.00000F
#define WI_0_7 1.00000F
#define WI_0_8 1.00000F
#define WI_0_9 1.00000F
#define WI_0_10 1.00000F
#define WI_0_11 1.00000F
#define WI_0_12 1.00000F
#define WI_0_13 1.00000F
#define WI_0_14 1.00000F
#define WI_0_15 1.00000F
#define WI_0_16 1.00000F
#define WI_0_17 1.00000F
#define WI_0_18 1.00000F
#define WI_0_19 1.00000F
#define WI_0_20 1.00000F
#define WI_0_21 1.00000F

__device__
float COST_0(float * w, float * c)
{
return  (w[2]*w[2])+(w[5]*w[5])+(w[0]*w[0])+(w[4]*w[4])+(w[1]*w[1])+(w[3]*w[3]);
}

__device__
float KKT_0_0(float * w, float * c)
{
return  w[6]+2.0*w[0]-w[7];
}

__device__
float KKT_0_1(float * w, float * c)
{
return -w[9]+2.0*w[1]+w[8];
}

__device__
float KKT_0_2(float * w, float * c)
{
return 2.0*w[2];
}

__device__
float KKT_0_3(float * w, float * c)
{
return 2.0*w[3];
}

__device__
float KKT_0_4(float * w, float * c)
{
return 2.0*w[4];
}

__device__
float KKT_0_5(float * w, float * c)
{
return 2.0*w[5];
}

__device__
float KKT_0_6(float * w, float * c)
{
return  w[0]+(w[14]*w[14])+9.6200390e+00;
}

__device__
float KKT_0_7(float * w, float * c)
{
return -w[0]+(w[15]*w[15])-9.6200390e+00;
}

__device__
float KKT_0_8(float * w, float * c)
{
return  (w[16]*w[16])+w[1]-8.3237314e+00;
}

__device__
float KKT_0_9(float * w, float * c)
{
return  (w[17]*w[17])-w[1]+8.3237314e+00;
}

__device__
float KKT_0_10(float * w, float * c)
{
return (w[18]*w[18]);
}

__device__
float KKT_0_11(float * w, float * c)
{
return (w[19]*w[19]);
}

__device__
float KKT_0_12(float * w, float * c)
{
return (w[20]*w[20]);
}

__device__
float KKT_0_13(float * w, float * c)
{
return (w[21]*w[21]);
}

__device__
float KKT_0_14(float * w, float * c)
{
return 2.0*w[6]*w[14];
}

__device__
float KKT_0_15(float * w, float * c)
{
return 2.0*w[15]*w[7];
}

__device__
float KKT_0_16(float * w, float * c)
{
return 2.0*w[16]*w[8];
}

__device__
float KKT_0_17(float * w, float * c)
{
return 2.0*w[9]*w[17];
}

__device__
float KKT_0_18(float * w, float * c)
{
return 2.0*w[10]*w[18];
}

__device__
float KKT_0_19(float * w, float * c)
{
return 2.0*w[19]*w[11];
}

__device__
float KKT_0_20(float * w, float * c)
{
return 2.0*w[20]*w[12];
}

__device__
float KKT_0_21(float * w, float * c)
{
return 2.0*w[13]*w[21];
}

__device__
float J_0_0_0(float * w, float * c)
{
return 2.0;
}

__device__
float J_0_0_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_0_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_0_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_0_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_0_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_0_6(float * w, float * c)
{
return 1.0;
}

__device__
float J_0_0_7(float * w, float * c)
{
return -1.0;
}

__device__
float J_0_0_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_0_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_0_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_0_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_0_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_0_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_0_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_0_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_0_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_0_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_0_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_0_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_0_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_0_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_1_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_1_1(float * w, float * c)
{
return 2.0;
}

__device__
float J_0_1_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_1_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_1_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_1_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_1_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_1_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_1_8(float * w, float * c)
{
return 1.0;
}

__device__
float J_0_1_9(float * w, float * c)
{
return -1.0;
}

__device__
float J_0_1_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_1_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_1_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_1_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_1_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_1_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_1_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_1_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_1_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_1_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_1_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_1_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_2_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_2_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_2_2(float * w, float * c)
{
return 2.0;
}

__device__
float J_0_2_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_2_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_2_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_2_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_2_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_2_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_2_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_2_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_2_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_2_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_2_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_2_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_2_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_2_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_2_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_2_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_2_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_2_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_2_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_3_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_3_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_3_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_3_3(float * w, float * c)
{
return 2.0;
}

__device__
float J_0_3_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_3_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_3_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_3_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_3_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_3_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_3_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_3_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_3_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_3_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_3_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_3_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_3_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_3_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_3_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_3_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_3_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_3_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_4_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_4_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_4_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_4_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_4_4(float * w, float * c)
{
return 2.0;
}

__device__
float J_0_4_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_4_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_4_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_4_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_4_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_4_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_4_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_4_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_4_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_4_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_4_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_4_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_4_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_4_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_4_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_4_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_4_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_5_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_5_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_5_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_5_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_5_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_5_5(float * w, float * c)
{
return 2.0;
}

__device__
float J_0_5_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_5_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_5_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_5_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_5_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_5_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_5_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_5_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_5_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_5_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_5_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_5_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_5_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_5_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_5_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_5_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_6_0(float * w, float * c)
{
return 1.0;
}

__device__
float J_0_6_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_6_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_6_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_6_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_6_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_6_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_6_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_6_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_6_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_6_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_6_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_6_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_6_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_6_14(float * w, float * c)
{
return 2.0*w[14];
}

__device__
float J_0_6_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_6_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_6_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_6_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_6_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_6_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_6_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_7_0(float * w, float * c)
{
return -1.0;
}

__device__
float J_0_7_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_7_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_7_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_7_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_7_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_7_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_7_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_7_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_7_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_7_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_7_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_7_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_7_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_7_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_7_15(float * w, float * c)
{
return 2.0*w[15];
}

__device__
float J_0_7_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_7_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_7_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_7_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_7_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_7_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_8_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_8_1(float * w, float * c)
{
return 1.0;
}

__device__
float J_0_8_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_8_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_8_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_8_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_8_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_8_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_8_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_8_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_8_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_8_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_8_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_8_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_8_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_8_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_8_16(float * w, float * c)
{
return 2.0*w[16];
}

__device__
float J_0_8_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_8_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_8_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_8_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_8_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_9_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_9_1(float * w, float * c)
{
return -1.0;
}

__device__
float J_0_9_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_9_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_9_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_9_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_9_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_9_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_9_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_9_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_9_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_9_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_9_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_9_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_9_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_9_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_9_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_9_17(float * w, float * c)
{
return 2.0*w[17];
}

__device__
float J_0_9_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_9_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_9_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_9_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_10_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_10_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_10_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_10_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_10_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_10_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_10_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_10_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_10_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_10_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_10_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_10_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_10_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_10_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_10_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_10_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_10_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_10_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_10_18(float * w, float * c)
{
return 2.0*w[18];
}

__device__
float J_0_10_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_10_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_10_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_11_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_11_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_11_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_11_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_11_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_11_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_11_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_11_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_11_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_11_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_11_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_11_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_11_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_11_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_11_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_11_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_11_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_11_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_11_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_11_19(float * w, float * c)
{
return 2.0*w[19];
}

__device__
float J_0_11_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_11_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_12_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_12_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_12_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_12_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_12_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_12_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_12_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_12_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_12_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_12_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_12_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_12_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_12_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_12_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_12_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_12_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_12_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_12_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_12_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_12_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_12_20(float * w, float * c)
{
return 2.0*w[20];
}

__device__
float J_0_12_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_13_21(float * w, float * c)
{
return 2.0*w[21];
}

__device__
float J_0_14_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_14_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_14_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_14_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_14_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_14_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_14_6(float * w, float * c)
{
return 2.0*w[14];
}

__device__
float J_0_14_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_14_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_14_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_14_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_14_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_14_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_14_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_14_14(float * w, float * c)
{
return 2.0*w[6];
}

__device__
float J_0_14_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_14_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_14_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_14_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_14_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_14_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_14_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_15_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_15_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_15_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_15_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_15_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_15_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_15_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_15_7(float * w, float * c)
{
return 2.0*w[15];
}

__device__
float J_0_15_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_15_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_15_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_15_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_15_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_15_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_15_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_15_15(float * w, float * c)
{
return 2.0*w[7];
}

__device__
float J_0_15_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_15_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_15_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_15_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_15_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_15_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_16_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_16_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_16_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_16_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_16_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_16_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_16_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_16_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_16_8(float * w, float * c)
{
return 2.0*w[16];
}

__device__
float J_0_16_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_16_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_16_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_16_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_16_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_16_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_16_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_16_16(float * w, float * c)
{
return 2.0*w[8];
}

__device__
float J_0_16_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_16_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_16_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_16_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_16_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_17_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_17_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_17_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_17_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_17_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_17_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_17_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_17_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_17_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_17_9(float * w, float * c)
{
return 2.0*w[17];
}

__device__
float J_0_17_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_17_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_17_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_17_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_17_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_17_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_17_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_17_17(float * w, float * c)
{
return 2.0*w[9];
}

__device__
float J_0_17_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_17_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_17_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_17_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_18_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_18_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_18_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_18_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_18_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_18_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_18_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_18_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_18_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_18_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_18_10(float * w, float * c)
{
return 2.0*w[18];
}

__device__
float J_0_18_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_18_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_18_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_18_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_18_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_18_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_18_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_18_18(float * w, float * c)
{
return 2.0*w[10];
}

__device__
float J_0_18_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_18_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_18_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_19_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_19_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_19_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_19_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_19_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_19_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_19_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_19_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_19_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_19_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_19_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_19_11(float * w, float * c)
{
return 2.0*w[19];
}

__device__
float J_0_19_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_19_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_19_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_19_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_19_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_19_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_19_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_19_19(float * w, float * c)
{
return 2.0*w[11];
}

__device__
float J_0_19_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_19_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_20_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_20_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_20_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_20_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_20_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_20_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_20_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_20_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_20_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_20_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_20_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_20_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_20_12(float * w, float * c)
{
return 2.0*w[20];
}

__device__
float J_0_20_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_20_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_20_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_20_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_20_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_20_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_20_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_20_20(float * w, float * c)
{
return 2.0*w[12];
}

__device__
float J_0_20_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_21_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_21_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_21_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_21_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_21_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_21_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_21_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_21_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_21_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_21_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_21_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_21_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_21_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_21_13(float * w, float * c)
{
return 2.0*w[21];
}

__device__
float J_0_21_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_21_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_21_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_21_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_21_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_21_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_21_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_0_21_21(float * w, float * c)
{
return 2.0*w[13];
}

#define WI_1_0 1.00000F
#define WI_1_1 1.00000F
#define WI_1_2 1.00000F
#define WI_1_3 1.00000F
#define WI_1_4 1.00000F
#define WI_1_5 1.00000F
#define WI_1_6 1.00000F
#define WI_1_7 1.00000F
#define WI_1_8 1.00000F
#define WI_1_9 1.00000F
#define WI_1_10 1.00000F
#define WI_1_11 1.00000F
#define WI_1_12 1.00000F
#define WI_1_13 1.00000F
#define WI_1_14 1.00000F
#define WI_1_15 1.00000F
#define WI_1_16 1.00000F
#define WI_1_17 1.00000F
#define WI_1_18 1.00000F
#define WI_1_19 1.00000F
#define WI_1_20 1.00000F
#define WI_1_21 1.00000F

__device__
float COST_1(float * w, float * c)
{
return  (w[2]*w[2])+(w[1]*w[1])+(w[4]*w[4])+(w[3]*w[3])+(w[5]*w[5])+(w[0]*w[0]);
}

__device__
float KKT_1_0(float * w, float * c)
{
return  w[6]+2.0*w[0]-w[7];
}

__device__
float KKT_1_1(float * w, float * c)
{
return  2.0*w[1]+w[8]-w[9];
}

__device__
float KKT_1_2(float * w, float * c)
{
return 2.0*w[2];
}

__device__
float KKT_1_3(float * w, float * c)
{
return 2.0*w[3];
}

__device__
float KKT_1_4(float * w, float * c)
{
return 2.0*w[4];
}

__device__
float KKT_1_5(float * w, float * c)
{
return 2.0*w[5];
}

__device__
float KKT_1_6(float * w, float * c)
{
return  w[0]+(w[14]*w[14])+3.1061583e+00;
}

__device__
float KKT_1_7(float * w, float * c)
{
return -w[0]+(w[15]*w[15])-3.1061583e+00;
}

__device__
float KKT_1_8(float * w, float * c)
{
return  (w[16]*w[16])+w[1]+3.5191097e+00;
}

__device__
float KKT_1_9(float * w, float * c)
{
return  (w[17]*w[17])-w[1]-3.5191097e+00;
}

__device__
float KKT_1_10(float * w, float * c)
{
return (w[18]*w[18]);
}

__device__
float KKT_1_11(float * w, float * c)
{
return (w[19]*w[19]);
}

__device__
float KKT_1_12(float * w, float * c)
{
return (w[20]*w[20]);
}

__device__
float KKT_1_13(float * w, float * c)
{
return (w[21]*w[21]);
}

__device__
float KKT_1_14(float * w, float * c)
{
return 2.0*w[6]*w[14];
}

__device__
float KKT_1_15(float * w, float * c)
{
return 2.0*w[7]*w[15];
}

__device__
float KKT_1_16(float * w, float * c)
{
return 2.0*w[8]*w[16];
}

__device__
float KKT_1_17(float * w, float * c)
{
return 2.0*w[17]*w[9];
}

__device__
float KKT_1_18(float * w, float * c)
{
return 2.0*w[10]*w[18];
}

__device__
float KKT_1_19(float * w, float * c)
{
return 2.0*w[11]*w[19];
}

__device__
float KKT_1_20(float * w, float * c)
{
return 2.0*w[20]*w[12];
}

__device__
float KKT_1_21(float * w, float * c)
{
return 2.0*w[13]*w[21];
}

__device__
float J_1_0_0(float * w, float * c)
{
return 2.0;
}

__device__
float J_1_0_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_0_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_0_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_0_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_0_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_0_6(float * w, float * c)
{
return 1.0;
}

__device__
float J_1_0_7(float * w, float * c)
{
return -1.0;
}

__device__
float J_1_0_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_0_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_0_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_0_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_0_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_0_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_0_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_0_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_0_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_0_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_0_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_0_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_0_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_0_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_1_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_1_1(float * w, float * c)
{
return 2.0;
}

__device__
float J_1_1_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_1_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_1_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_1_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_1_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_1_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_1_8(float * w, float * c)
{
return 1.0;
}

__device__
float J_1_1_9(float * w, float * c)
{
return -1.0;
}

__device__
float J_1_1_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_1_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_1_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_1_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_1_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_1_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_1_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_1_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_1_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_1_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_1_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_1_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_2_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_2_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_2_2(float * w, float * c)
{
return 2.0;
}

__device__
float J_1_2_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_2_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_2_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_2_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_2_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_2_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_2_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_2_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_2_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_2_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_2_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_2_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_2_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_2_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_2_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_2_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_2_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_2_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_2_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_3_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_3_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_3_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_3_3(float * w, float * c)
{
return 2.0;
}

__device__
float J_1_3_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_3_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_3_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_3_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_3_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_3_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_3_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_3_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_3_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_3_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_3_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_3_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_3_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_3_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_3_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_3_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_3_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_3_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_4_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_4_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_4_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_4_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_4_4(float * w, float * c)
{
return 2.0;
}

__device__
float J_1_4_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_4_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_4_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_4_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_4_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_4_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_4_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_4_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_4_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_4_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_4_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_4_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_4_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_4_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_4_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_4_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_4_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_5_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_5_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_5_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_5_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_5_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_5_5(float * w, float * c)
{
return 2.0;
}

__device__
float J_1_5_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_5_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_5_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_5_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_5_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_5_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_5_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_5_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_5_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_5_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_5_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_5_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_5_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_5_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_5_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_5_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_6_0(float * w, float * c)
{
return 1.0;
}

__device__
float J_1_6_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_6_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_6_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_6_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_6_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_6_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_6_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_6_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_6_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_6_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_6_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_6_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_6_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_6_14(float * w, float * c)
{
return 2.0*w[14];
}

__device__
float J_1_6_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_6_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_6_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_6_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_6_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_6_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_6_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_7_0(float * w, float * c)
{
return -1.0;
}

__device__
float J_1_7_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_7_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_7_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_7_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_7_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_7_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_7_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_7_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_7_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_7_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_7_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_7_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_7_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_7_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_7_15(float * w, float * c)
{
return 2.0*w[15];
}

__device__
float J_1_7_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_7_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_7_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_7_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_7_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_7_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_8_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_8_1(float * w, float * c)
{
return 1.0;
}

__device__
float J_1_8_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_8_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_8_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_8_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_8_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_8_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_8_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_8_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_8_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_8_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_8_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_8_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_8_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_8_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_8_16(float * w, float * c)
{
return 2.0*w[16];
}

__device__
float J_1_8_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_8_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_8_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_8_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_8_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_9_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_9_1(float * w, float * c)
{
return -1.0;
}

__device__
float J_1_9_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_9_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_9_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_9_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_9_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_9_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_9_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_9_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_9_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_9_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_9_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_9_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_9_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_9_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_9_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_9_17(float * w, float * c)
{
return 2.0*w[17];
}

__device__
float J_1_9_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_9_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_9_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_9_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_10_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_10_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_10_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_10_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_10_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_10_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_10_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_10_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_10_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_10_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_10_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_10_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_10_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_10_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_10_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_10_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_10_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_10_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_10_18(float * w, float * c)
{
return 2.0*w[18];
}

__device__
float J_1_10_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_10_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_10_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_11_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_11_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_11_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_11_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_11_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_11_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_11_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_11_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_11_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_11_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_11_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_11_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_11_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_11_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_11_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_11_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_11_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_11_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_11_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_11_19(float * w, float * c)
{
return 2.0*w[19];
}

__device__
float J_1_11_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_11_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_12_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_12_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_12_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_12_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_12_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_12_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_12_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_12_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_12_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_12_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_12_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_12_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_12_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_12_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_12_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_12_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_12_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_12_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_12_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_12_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_12_20(float * w, float * c)
{
return 2.0*w[20];
}

__device__
float J_1_12_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_13_21(float * w, float * c)
{
return 2.0*w[21];
}

__device__
float J_1_14_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_14_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_14_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_14_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_14_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_14_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_14_6(float * w, float * c)
{
return 2.0*w[14];
}

__device__
float J_1_14_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_14_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_14_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_14_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_14_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_14_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_14_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_14_14(float * w, float * c)
{
return 2.0*w[6];
}

__device__
float J_1_14_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_14_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_14_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_14_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_14_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_14_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_14_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_15_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_15_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_15_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_15_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_15_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_15_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_15_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_15_7(float * w, float * c)
{
return 2.0*w[15];
}

__device__
float J_1_15_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_15_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_15_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_15_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_15_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_15_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_15_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_15_15(float * w, float * c)
{
return 2.0*w[7];
}

__device__
float J_1_15_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_15_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_15_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_15_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_15_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_15_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_16_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_16_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_16_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_16_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_16_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_16_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_16_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_16_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_16_8(float * w, float * c)
{
return 2.0*w[16];
}

__device__
float J_1_16_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_16_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_16_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_16_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_16_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_16_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_16_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_16_16(float * w, float * c)
{
return 2.0*w[8];
}

__device__
float J_1_16_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_16_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_16_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_16_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_16_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_17_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_17_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_17_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_17_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_17_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_17_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_17_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_17_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_17_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_17_9(float * w, float * c)
{
return 2.0*w[17];
}

__device__
float J_1_17_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_17_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_17_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_17_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_17_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_17_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_17_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_17_17(float * w, float * c)
{
return 2.0*w[9];
}

__device__
float J_1_17_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_17_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_17_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_17_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_18_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_18_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_18_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_18_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_18_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_18_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_18_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_18_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_18_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_18_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_18_10(float * w, float * c)
{
return 2.0*w[18];
}

__device__
float J_1_18_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_18_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_18_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_18_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_18_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_18_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_18_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_18_18(float * w, float * c)
{
return 2.0*w[10];
}

__device__
float J_1_18_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_18_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_18_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_19_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_19_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_19_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_19_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_19_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_19_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_19_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_19_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_19_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_19_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_19_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_19_11(float * w, float * c)
{
return 2.0*w[19];
}

__device__
float J_1_19_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_19_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_19_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_19_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_19_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_19_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_19_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_19_19(float * w, float * c)
{
return 2.0*w[11];
}

__device__
float J_1_19_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_19_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_20_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_20_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_20_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_20_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_20_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_20_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_20_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_20_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_20_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_20_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_20_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_20_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_20_12(float * w, float * c)
{
return 2.0*w[20];
}

__device__
float J_1_20_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_20_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_20_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_20_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_20_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_20_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_20_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_20_20(float * w, float * c)
{
return 2.0*w[12];
}

__device__
float J_1_20_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_21_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_21_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_21_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_21_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_21_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_21_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_21_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_21_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_21_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_21_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_21_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_21_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_21_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_21_13(float * w, float * c)
{
return 2.0*w[21];
}

__device__
float J_1_21_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_21_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_21_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_21_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_21_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_21_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_21_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_1_21_21(float * w, float * c)
{
return 2.0*w[13];
}

#define WI_2_0 1.00000F
#define WI_2_1 1.00000F
#define WI_2_2 1.00000F
#define WI_2_3 1.00000F
#define WI_2_4 1.00000F
#define WI_2_5 1.00000F
#define WI_2_6 1.00000F
#define WI_2_7 1.00000F
#define WI_2_8 1.00000F
#define WI_2_9 1.00000F
#define WI_2_10 1.00000F
#define WI_2_11 1.00000F
#define WI_2_12 1.00000F
#define WI_2_13 1.00000F
#define WI_2_14 1.00000F
#define WI_2_15 1.00000F
#define WI_2_16 1.00000F
#define WI_2_17 1.00000F
#define WI_2_18 1.00000F
#define WI_2_19 1.00000F
#define WI_2_20 1.00000F
#define WI_2_21 1.00000F

__device__
float COST_2(float * w, float * c)
{
return  (w[0]*w[0])+(w[3]*w[3])+(w[2]*w[2])+(w[5]*w[5])+(w[4]*w[4])+(w[1]*w[1]);
}

__device__
float KKT_2_0(float * w, float * c)
{
return -w[7]+w[6]+2.0*w[0];
}

__device__
float KKT_2_1(float * w, float * c)
{
return  2.0*w[1]+w[8]-w[9];
}

__device__
float KKT_2_2(float * w, float * c)
{
return 2.0*w[2];
}

__device__
float KKT_2_3(float * w, float * c)
{
return 2.0*w[3];
}

__device__
float KKT_2_4(float * w, float * c)
{
return 2.0*w[4];
}

__device__
float KKT_2_5(float * w, float * c)
{
return 2.0*w[5];
}

__device__
float KKT_2_6(float * w, float * c)
{
return  (w[14]*w[14])+w[0]+2.2708039e+00;
}

__device__
float KKT_2_7(float * w, float * c)
{
return  (w[15]*w[15])-w[0]-2.2708039e+00;
}

__device__
float KKT_2_8(float * w, float * c)
{
return  w[1]+(w[16]*w[16])-3.4868014e+00;
}

__device__
float KKT_2_9(float * w, float * c)
{
return -w[1]+(w[17]*w[17])+3.4868014e+00;
}

__device__
float KKT_2_10(float * w, float * c)
{
return (w[18]*w[18]);
}

__device__
float KKT_2_11(float * w, float * c)
{
return (w[19]*w[19]);
}

__device__
float KKT_2_12(float * w, float * c)
{
return (w[20]*w[20]);
}

__device__
float KKT_2_13(float * w, float * c)
{
return (w[21]*w[21]);
}

__device__
float KKT_2_14(float * w, float * c)
{
return 2.0*w[14]*w[6];
}

__device__
float KKT_2_15(float * w, float * c)
{
return 2.0*w[7]*w[15];
}

__device__
float KKT_2_16(float * w, float * c)
{
return 2.0*w[8]*w[16];
}

__device__
float KKT_2_17(float * w, float * c)
{
return 2.0*w[17]*w[9];
}

__device__
float KKT_2_18(float * w, float * c)
{
return 2.0*w[10]*w[18];
}

__device__
float KKT_2_19(float * w, float * c)
{
return 2.0*w[19]*w[11];
}

__device__
float KKT_2_20(float * w, float * c)
{
return 2.0*w[20]*w[12];
}

__device__
float KKT_2_21(float * w, float * c)
{
return 2.0*w[21]*w[13];
}

__device__
float J_2_0_0(float * w, float * c)
{
return 2.0;
}

__device__
float J_2_0_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_0_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_0_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_0_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_0_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_0_6(float * w, float * c)
{
return 1.0;
}

__device__
float J_2_0_7(float * w, float * c)
{
return -1.0;
}

__device__
float J_2_0_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_0_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_0_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_0_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_0_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_0_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_0_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_0_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_0_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_0_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_0_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_0_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_0_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_0_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_1_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_1_1(float * w, float * c)
{
return 2.0;
}

__device__
float J_2_1_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_1_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_1_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_1_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_1_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_1_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_1_8(float * w, float * c)
{
return 1.0;
}

__device__
float J_2_1_9(float * w, float * c)
{
return -1.0;
}

__device__
float J_2_1_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_1_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_1_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_1_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_1_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_1_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_1_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_1_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_1_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_1_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_1_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_1_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_2_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_2_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_2_2(float * w, float * c)
{
return 2.0;
}

__device__
float J_2_2_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_2_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_2_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_2_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_2_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_2_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_2_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_2_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_2_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_2_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_2_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_2_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_2_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_2_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_2_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_2_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_2_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_2_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_2_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_3_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_3_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_3_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_3_3(float * w, float * c)
{
return 2.0;
}

__device__
float J_2_3_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_3_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_3_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_3_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_3_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_3_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_3_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_3_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_3_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_3_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_3_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_3_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_3_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_3_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_3_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_3_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_3_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_3_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_4_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_4_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_4_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_4_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_4_4(float * w, float * c)
{
return 2.0;
}

__device__
float J_2_4_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_4_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_4_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_4_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_4_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_4_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_4_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_4_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_4_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_4_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_4_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_4_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_4_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_4_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_4_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_4_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_4_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_5_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_5_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_5_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_5_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_5_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_5_5(float * w, float * c)
{
return 2.0;
}

__device__
float J_2_5_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_5_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_5_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_5_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_5_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_5_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_5_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_5_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_5_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_5_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_5_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_5_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_5_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_5_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_5_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_5_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_6_0(float * w, float * c)
{
return 1.0;
}

__device__
float J_2_6_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_6_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_6_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_6_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_6_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_6_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_6_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_6_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_6_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_6_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_6_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_6_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_6_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_6_14(float * w, float * c)
{
return 2.0*w[14];
}

__device__
float J_2_6_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_6_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_6_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_6_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_6_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_6_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_6_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_7_0(float * w, float * c)
{
return -1.0;
}

__device__
float J_2_7_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_7_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_7_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_7_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_7_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_7_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_7_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_7_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_7_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_7_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_7_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_7_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_7_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_7_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_7_15(float * w, float * c)
{
return 2.0*w[15];
}

__device__
float J_2_7_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_7_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_7_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_7_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_7_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_7_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_8_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_8_1(float * w, float * c)
{
return 1.0;
}

__device__
float J_2_8_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_8_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_8_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_8_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_8_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_8_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_8_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_8_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_8_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_8_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_8_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_8_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_8_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_8_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_8_16(float * w, float * c)
{
return 2.0*w[16];
}

__device__
float J_2_8_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_8_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_8_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_8_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_8_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_9_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_9_1(float * w, float * c)
{
return -1.0;
}

__device__
float J_2_9_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_9_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_9_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_9_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_9_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_9_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_9_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_9_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_9_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_9_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_9_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_9_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_9_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_9_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_9_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_9_17(float * w, float * c)
{
return 2.0*w[17];
}

__device__
float J_2_9_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_9_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_9_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_9_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_10_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_10_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_10_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_10_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_10_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_10_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_10_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_10_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_10_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_10_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_10_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_10_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_10_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_10_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_10_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_10_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_10_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_10_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_10_18(float * w, float * c)
{
return 2.0*w[18];
}

__device__
float J_2_10_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_10_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_10_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_11_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_11_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_11_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_11_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_11_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_11_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_11_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_11_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_11_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_11_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_11_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_11_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_11_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_11_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_11_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_11_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_11_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_11_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_11_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_11_19(float * w, float * c)
{
return 2.0*w[19];
}

__device__
float J_2_11_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_11_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_12_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_12_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_12_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_12_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_12_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_12_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_12_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_12_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_12_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_12_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_12_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_12_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_12_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_12_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_12_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_12_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_12_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_12_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_12_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_12_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_12_20(float * w, float * c)
{
return 2.0*w[20];
}

__device__
float J_2_12_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_13_21(float * w, float * c)
{
return 2.0*w[21];
}

__device__
float J_2_14_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_14_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_14_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_14_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_14_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_14_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_14_6(float * w, float * c)
{
return 2.0*w[14];
}

__device__
float J_2_14_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_14_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_14_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_14_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_14_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_14_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_14_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_14_14(float * w, float * c)
{
return 2.0*w[6];
}

__device__
float J_2_14_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_14_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_14_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_14_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_14_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_14_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_14_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_15_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_15_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_15_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_15_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_15_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_15_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_15_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_15_7(float * w, float * c)
{
return 2.0*w[15];
}

__device__
float J_2_15_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_15_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_15_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_15_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_15_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_15_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_15_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_15_15(float * w, float * c)
{
return 2.0*w[7];
}

__device__
float J_2_15_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_15_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_15_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_15_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_15_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_15_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_16_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_16_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_16_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_16_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_16_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_16_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_16_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_16_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_16_8(float * w, float * c)
{
return 2.0*w[16];
}

__device__
float J_2_16_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_16_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_16_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_16_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_16_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_16_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_16_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_16_16(float * w, float * c)
{
return 2.0*w[8];
}

__device__
float J_2_16_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_16_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_16_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_16_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_16_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_17_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_17_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_17_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_17_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_17_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_17_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_17_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_17_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_17_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_17_9(float * w, float * c)
{
return 2.0*w[17];
}

__device__
float J_2_17_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_17_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_17_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_17_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_17_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_17_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_17_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_17_17(float * w, float * c)
{
return 2.0*w[9];
}

__device__
float J_2_17_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_17_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_17_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_17_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_18_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_18_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_18_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_18_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_18_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_18_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_18_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_18_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_18_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_18_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_18_10(float * w, float * c)
{
return 2.0*w[18];
}

__device__
float J_2_18_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_18_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_18_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_18_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_18_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_18_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_18_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_18_18(float * w, float * c)
{
return 2.0*w[10];
}

__device__
float J_2_18_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_18_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_18_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_19_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_19_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_19_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_19_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_19_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_19_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_19_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_19_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_19_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_19_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_19_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_19_11(float * w, float * c)
{
return 2.0*w[19];
}

__device__
float J_2_19_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_19_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_19_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_19_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_19_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_19_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_19_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_19_19(float * w, float * c)
{
return 2.0*w[11];
}

__device__
float J_2_19_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_19_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_20_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_20_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_20_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_20_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_20_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_20_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_20_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_20_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_20_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_20_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_20_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_20_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_20_12(float * w, float * c)
{
return 2.0*w[20];
}

__device__
float J_2_20_13(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_20_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_20_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_20_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_20_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_20_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_20_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_20_20(float * w, float * c)
{
return 2.0*w[12];
}

__device__
float J_2_20_21(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_21_0(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_21_1(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_21_2(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_21_3(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_21_4(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_21_5(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_21_6(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_21_7(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_21_8(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_21_9(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_21_10(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_21_11(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_21_12(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_21_13(float * w, float * c)
{
return 2.0*w[21];
}

__device__
float J_2_21_14(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_21_15(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_21_16(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_21_17(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_21_18(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_21_19(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_21_20(float * w, float * c)
{
return 0.0;
}

__device__
float J_2_21_21(float * w, float * c)
{
return 2.0*w[13];
}

__constant__
float WI_LOOKUP[66] =
{
WI_0_0, WI_0_1, WI_0_2, WI_0_3, WI_0_4, WI_0_5, WI_0_6, WI_0_7, WI_0_8, WI_0_9, WI_0_10, WI_0_11, WI_0_12, WI_0_13, WI_0_14, WI_0_15, WI_0_16, WI_0_17, WI_0_18, WI_0_19, WI_0_20, WI_0_21, 
WI_1_0, WI_1_1, WI_1_2, WI_1_3, WI_1_4, WI_1_5, WI_1_6, WI_1_7, WI_1_8, WI_1_9, WI_1_10, WI_1_11, WI_1_12, WI_1_13, WI_1_14, WI_1_15, WI_1_16, WI_1_17, WI_1_18, WI_1_19, WI_1_20, WI_1_21, 
WI_2_0, WI_2_1, WI_2_2, WI_2_3, WI_2_4, WI_2_5, WI_2_6, WI_2_7, WI_2_8, WI_2_9, WI_2_10, WI_2_11, WI_2_12, WI_2_13, WI_2_14, WI_2_15, WI_2_16, WI_2_17, WI_2_18, WI_2_19, WI_2_20, WI_2_21, 
};

__global__
void WI_EVALUATE(float * w)
{
const int idx = blockDim.x*blockIdx.x + threadIdx.x;
w[idx] = WI_LOOKUP[idx];
};

__constant__
VAL_PTR COST_LOOKUP[3] =
{
COST_0, 
COST_1, 
COST_2, 
};

__constant__
VAL_PTR KKT_LOOKUP[66] =
{
KKT_0_0, KKT_0_1, KKT_0_2, KKT_0_3, KKT_0_4, KKT_0_5, KKT_0_6, KKT_0_7, KKT_0_8, KKT_0_9, KKT_0_10, KKT_0_11, KKT_0_12, KKT_0_13, KKT_0_14, KKT_0_15, KKT_0_16, KKT_0_17, KKT_0_18, KKT_0_19, KKT_0_20, KKT_0_21, 
KKT_1_0, KKT_1_1, KKT_1_2, KKT_1_3, KKT_1_4, KKT_1_5, KKT_1_6, KKT_1_7, KKT_1_8, KKT_1_9, KKT_1_10, KKT_1_11, KKT_1_12, KKT_1_13, KKT_1_14, KKT_1_15, KKT_1_16, KKT_1_17, KKT_1_18, KKT_1_19, KKT_1_20, KKT_1_21, 
KKT_2_0, KKT_2_1, KKT_2_2, KKT_2_3, KKT_2_4, KKT_2_5, KKT_2_6, KKT_2_7, KKT_2_8, KKT_2_9, KKT_2_10, KKT_2_11, KKT_2_12, KKT_2_13, KKT_2_14, KKT_2_15, KKT_2_16, KKT_2_17, KKT_2_18, KKT_2_19, KKT_2_20, KKT_2_21, 
};

__global__
void KKT_EVALUATE(float * KKT, float * w, float * c)
{
const int idx = blockDim.x*blockIdx.x + threadIdx.x;
KKT[idx] = KKT_LOOKUP[idx](w, c);
};

__global__
void KKT_FORMAT(float ** KKT, float * d_KKT)
{
const int idx = blockDim.x*blockIdx.x + threadIdx.x;
KKT[idx] = &d_KKT[NUM_VARIABLES*idx];
};

__constant__
VAL_PTR J_LOOKUP[1452] =
{
J_0_0_0, J_0_0_1, J_0_0_2, J_0_0_3, J_0_0_4, J_0_0_5, J_0_0_6, J_0_0_7, J_0_0_8, J_0_0_9, J_0_0_10, J_0_0_11, J_0_0_12, J_0_0_13, J_0_0_14, J_0_0_15, J_0_0_16, J_0_0_17, J_0_0_18, J_0_0_19, J_0_0_20, J_0_0_21, 
J_0_1_0, J_0_1_1, J_0_1_2, J_0_1_3, J_0_1_4, J_0_1_5, J_0_1_6, J_0_1_7, J_0_1_8, J_0_1_9, J_0_1_10, J_0_1_11, J_0_1_12, J_0_1_13, J_0_1_14, J_0_1_15, J_0_1_16, J_0_1_17, J_0_1_18, J_0_1_19, J_0_1_20, J_0_1_21, 
J_0_2_0, J_0_2_1, J_0_2_2, J_0_2_3, J_0_2_4, J_0_2_5, J_0_2_6, J_0_2_7, J_0_2_8, J_0_2_9, J_0_2_10, J_0_2_11, J_0_2_12, J_0_2_13, J_0_2_14, J_0_2_15, J_0_2_16, J_0_2_17, J_0_2_18, J_0_2_19, J_0_2_20, J_0_2_21, 
J_0_3_0, J_0_3_1, J_0_3_2, J_0_3_3, J_0_3_4, J_0_3_5, J_0_3_6, J_0_3_7, J_0_3_8, J_0_3_9, J_0_3_10, J_0_3_11, J_0_3_12, J_0_3_13, J_0_3_14, J_0_3_15, J_0_3_16, J_0_3_17, J_0_3_18, J_0_3_19, J_0_3_20, J_0_3_21, 
J_0_4_0, J_0_4_1, J_0_4_2, J_0_4_3, J_0_4_4, J_0_4_5, J_0_4_6, J_0_4_7, J_0_4_8, J_0_4_9, J_0_4_10, J_0_4_11, J_0_4_12, J_0_4_13, J_0_4_14, J_0_4_15, J_0_4_16, J_0_4_17, J_0_4_18, J_0_4_19, J_0_4_20, J_0_4_21, 
J_0_5_0, J_0_5_1, J_0_5_2, J_0_5_3, J_0_5_4, J_0_5_5, J_0_5_6, J_0_5_7, J_0_5_8, J_0_5_9, J_0_5_10, J_0_5_11, J_0_5_12, J_0_5_13, J_0_5_14, J_0_5_15, J_0_5_16, J_0_5_17, J_0_5_18, J_0_5_19, J_0_5_20, J_0_5_21, 
J_0_6_0, J_0_6_1, J_0_6_2, J_0_6_3, J_0_6_4, J_0_6_5, J_0_6_6, J_0_6_7, J_0_6_8, J_0_6_9, J_0_6_10, J_0_6_11, J_0_6_12, J_0_6_13, J_0_6_14, J_0_6_15, J_0_6_16, J_0_6_17, J_0_6_18, J_0_6_19, J_0_6_20, J_0_6_21, 
J_0_7_0, J_0_7_1, J_0_7_2, J_0_7_3, J_0_7_4, J_0_7_5, J_0_7_6, J_0_7_7, J_0_7_8, J_0_7_9, J_0_7_10, J_0_7_11, J_0_7_12, J_0_7_13, J_0_7_14, J_0_7_15, J_0_7_16, J_0_7_17, J_0_7_18, J_0_7_19, J_0_7_20, J_0_7_21, 
J_0_8_0, J_0_8_1, J_0_8_2, J_0_8_3, J_0_8_4, J_0_8_5, J_0_8_6, J_0_8_7, J_0_8_8, J_0_8_9, J_0_8_10, J_0_8_11, J_0_8_12, J_0_8_13, J_0_8_14, J_0_8_15, J_0_8_16, J_0_8_17, J_0_8_18, J_0_8_19, J_0_8_20, J_0_8_21, 
J_0_9_0, J_0_9_1, J_0_9_2, J_0_9_3, J_0_9_4, J_0_9_5, J_0_9_6, J_0_9_7, J_0_9_8, J_0_9_9, J_0_9_10, J_0_9_11, J_0_9_12, J_0_9_13, J_0_9_14, J_0_9_15, J_0_9_16, J_0_9_17, J_0_9_18, J_0_9_19, J_0_9_20, J_0_9_21, 
J_0_10_0, J_0_10_1, J_0_10_2, J_0_10_3, J_0_10_4, J_0_10_5, J_0_10_6, J_0_10_7, J_0_10_8, J_0_10_9, J_0_10_10, J_0_10_11, J_0_10_12, J_0_10_13, J_0_10_14, J_0_10_15, J_0_10_16, J_0_10_17, J_0_10_18, J_0_10_19, J_0_10_20, J_0_10_21, 
J_0_11_0, J_0_11_1, J_0_11_2, J_0_11_3, J_0_11_4, J_0_11_5, J_0_11_6, J_0_11_7, J_0_11_8, J_0_11_9, J_0_11_10, J_0_11_11, J_0_11_12, J_0_11_13, J_0_11_14, J_0_11_15, J_0_11_16, J_0_11_17, J_0_11_18, J_0_11_19, J_0_11_20, J_0_11_21, 
J_0_12_0, J_0_12_1, J_0_12_2, J_0_12_3, J_0_12_4, J_0_12_5, J_0_12_6, J_0_12_7, J_0_12_8, J_0_12_9, J_0_12_10, J_0_12_11, J_0_12_12, J_0_12_13, J_0_12_14, J_0_12_15, J_0_12_16, J_0_12_17, J_0_12_18, J_0_12_19, J_0_12_20, J_0_12_21, 
J_0_13_0, J_0_13_1, J_0_13_2, J_0_13_3, J_0_13_4, J_0_13_5, J_0_13_6, J_0_13_7, J_0_13_8, J_0_13_9, J_0_13_10, J_0_13_11, J_0_13_12, J_0_13_13, J_0_13_14, J_0_13_15, J_0_13_16, J_0_13_17, J_0_13_18, J_0_13_19, J_0_13_20, J_0_13_21, 
J_0_14_0, J_0_14_1, J_0_14_2, J_0_14_3, J_0_14_4, J_0_14_5, J_0_14_6, J_0_14_7, J_0_14_8, J_0_14_9, J_0_14_10, J_0_14_11, J_0_14_12, J_0_14_13, J_0_14_14, J_0_14_15, J_0_14_16, J_0_14_17, J_0_14_18, J_0_14_19, J_0_14_20, J_0_14_21, 
J_0_15_0, J_0_15_1, J_0_15_2, J_0_15_3, J_0_15_4, J_0_15_5, J_0_15_6, J_0_15_7, J_0_15_8, J_0_15_9, J_0_15_10, J_0_15_11, J_0_15_12, J_0_15_13, J_0_15_14, J_0_15_15, J_0_15_16, J_0_15_17, J_0_15_18, J_0_15_19, J_0_15_20, J_0_15_21, 
J_0_16_0, J_0_16_1, J_0_16_2, J_0_16_3, J_0_16_4, J_0_16_5, J_0_16_6, J_0_16_7, J_0_16_8, J_0_16_9, J_0_16_10, J_0_16_11, J_0_16_12, J_0_16_13, J_0_16_14, J_0_16_15, J_0_16_16, J_0_16_17, J_0_16_18, J_0_16_19, J_0_16_20, J_0_16_21, 
J_0_17_0, J_0_17_1, J_0_17_2, J_0_17_3, J_0_17_4, J_0_17_5, J_0_17_6, J_0_17_7, J_0_17_8, J_0_17_9, J_0_17_10, J_0_17_11, J_0_17_12, J_0_17_13, J_0_17_14, J_0_17_15, J_0_17_16, J_0_17_17, J_0_17_18, J_0_17_19, J_0_17_20, J_0_17_21, 
J_0_18_0, J_0_18_1, J_0_18_2, J_0_18_3, J_0_18_4, J_0_18_5, J_0_18_6, J_0_18_7, J_0_18_8, J_0_18_9, J_0_18_10, J_0_18_11, J_0_18_12, J_0_18_13, J_0_18_14, J_0_18_15, J_0_18_16, J_0_18_17, J_0_18_18, J_0_18_19, J_0_18_20, J_0_18_21, 
J_0_19_0, J_0_19_1, J_0_19_2, J_0_19_3, J_0_19_4, J_0_19_5, J_0_19_6, J_0_19_7, J_0_19_8, J_0_19_9, J_0_19_10, J_0_19_11, J_0_19_12, J_0_19_13, J_0_19_14, J_0_19_15, J_0_19_16, J_0_19_17, J_0_19_18, J_0_19_19, J_0_19_20, J_0_19_21, 
J_0_20_0, J_0_20_1, J_0_20_2, J_0_20_3, J_0_20_4, J_0_20_5, J_0_20_6, J_0_20_7, J_0_20_8, J_0_20_9, J_0_20_10, J_0_20_11, J_0_20_12, J_0_20_13, J_0_20_14, J_0_20_15, J_0_20_16, J_0_20_17, J_0_20_18, J_0_20_19, J_0_20_20, J_0_20_21, 
J_0_21_0, J_0_21_1, J_0_21_2, J_0_21_3, J_0_21_4, J_0_21_5, J_0_21_6, J_0_21_7, J_0_21_8, J_0_21_9, J_0_21_10, J_0_21_11, J_0_21_12, J_0_21_13, J_0_21_14, J_0_21_15, J_0_21_16, J_0_21_17, J_0_21_18, J_0_21_19, J_0_21_20, J_0_21_21, 
J_1_0_0, J_1_0_1, J_1_0_2, J_1_0_3, J_1_0_4, J_1_0_5, J_1_0_6, J_1_0_7, J_1_0_8, J_1_0_9, J_1_0_10, J_1_0_11, J_1_0_12, J_1_0_13, J_1_0_14, J_1_0_15, J_1_0_16, J_1_0_17, J_1_0_18, J_1_0_19, J_1_0_20, J_1_0_21, 
J_1_1_0, J_1_1_1, J_1_1_2, J_1_1_3, J_1_1_4, J_1_1_5, J_1_1_6, J_1_1_7, J_1_1_8, J_1_1_9, J_1_1_10, J_1_1_11, J_1_1_12, J_1_1_13, J_1_1_14, J_1_1_15, J_1_1_16, J_1_1_17, J_1_1_18, J_1_1_19, J_1_1_20, J_1_1_21, 
J_1_2_0, J_1_2_1, J_1_2_2, J_1_2_3, J_1_2_4, J_1_2_5, J_1_2_6, J_1_2_7, J_1_2_8, J_1_2_9, J_1_2_10, J_1_2_11, J_1_2_12, J_1_2_13, J_1_2_14, J_1_2_15, J_1_2_16, J_1_2_17, J_1_2_18, J_1_2_19, J_1_2_20, J_1_2_21, 
J_1_3_0, J_1_3_1, J_1_3_2, J_1_3_3, J_1_3_4, J_1_3_5, J_1_3_6, J_1_3_7, J_1_3_8, J_1_3_9, J_1_3_10, J_1_3_11, J_1_3_12, J_1_3_13, J_1_3_14, J_1_3_15, J_1_3_16, J_1_3_17, J_1_3_18, J_1_3_19, J_1_3_20, J_1_3_21, 
J_1_4_0, J_1_4_1, J_1_4_2, J_1_4_3, J_1_4_4, J_1_4_5, J_1_4_6, J_1_4_7, J_1_4_8, J_1_4_9, J_1_4_10, J_1_4_11, J_1_4_12, J_1_4_13, J_1_4_14, J_1_4_15, J_1_4_16, J_1_4_17, J_1_4_18, J_1_4_19, J_1_4_20, J_1_4_21, 
J_1_5_0, J_1_5_1, J_1_5_2, J_1_5_3, J_1_5_4, J_1_5_5, J_1_5_6, J_1_5_7, J_1_5_8, J_1_5_9, J_1_5_10, J_1_5_11, J_1_5_12, J_1_5_13, J_1_5_14, J_1_5_15, J_1_5_16, J_1_5_17, J_1_5_18, J_1_5_19, J_1_5_20, J_1_5_21, 
J_1_6_0, J_1_6_1, J_1_6_2, J_1_6_3, J_1_6_4, J_1_6_5, J_1_6_6, J_1_6_7, J_1_6_8, J_1_6_9, J_1_6_10, J_1_6_11, J_1_6_12, J_1_6_13, J_1_6_14, J_1_6_15, J_1_6_16, J_1_6_17, J_1_6_18, J_1_6_19, J_1_6_20, J_1_6_21, 
J_1_7_0, J_1_7_1, J_1_7_2, J_1_7_3, J_1_7_4, J_1_7_5, J_1_7_6, J_1_7_7, J_1_7_8, J_1_7_9, J_1_7_10, J_1_7_11, J_1_7_12, J_1_7_13, J_1_7_14, J_1_7_15, J_1_7_16, J_1_7_17, J_1_7_18, J_1_7_19, J_1_7_20, J_1_7_21, 
J_1_8_0, J_1_8_1, J_1_8_2, J_1_8_3, J_1_8_4, J_1_8_5, J_1_8_6, J_1_8_7, J_1_8_8, J_1_8_9, J_1_8_10, J_1_8_11, J_1_8_12, J_1_8_13, J_1_8_14, J_1_8_15, J_1_8_16, J_1_8_17, J_1_8_18, J_1_8_19, J_1_8_20, J_1_8_21, 
J_1_9_0, J_1_9_1, J_1_9_2, J_1_9_3, J_1_9_4, J_1_9_5, J_1_9_6, J_1_9_7, J_1_9_8, J_1_9_9, J_1_9_10, J_1_9_11, J_1_9_12, J_1_9_13, J_1_9_14, J_1_9_15, J_1_9_16, J_1_9_17, J_1_9_18, J_1_9_19, J_1_9_20, J_1_9_21, 
J_1_10_0, J_1_10_1, J_1_10_2, J_1_10_3, J_1_10_4, J_1_10_5, J_1_10_6, J_1_10_7, J_1_10_8, J_1_10_9, J_1_10_10, J_1_10_11, J_1_10_12, J_1_10_13, J_1_10_14, J_1_10_15, J_1_10_16, J_1_10_17, J_1_10_18, J_1_10_19, J_1_10_20, J_1_10_21, 
J_1_11_0, J_1_11_1, J_1_11_2, J_1_11_3, J_1_11_4, J_1_11_5, J_1_11_6, J_1_11_7, J_1_11_8, J_1_11_9, J_1_11_10, J_1_11_11, J_1_11_12, J_1_11_13, J_1_11_14, J_1_11_15, J_1_11_16, J_1_11_17, J_1_11_18, J_1_11_19, J_1_11_20, J_1_11_21, 
J_1_12_0, J_1_12_1, J_1_12_2, J_1_12_3, J_1_12_4, J_1_12_5, J_1_12_6, J_1_12_7, J_1_12_8, J_1_12_9, J_1_12_10, J_1_12_11, J_1_12_12, J_1_12_13, J_1_12_14, J_1_12_15, J_1_12_16, J_1_12_17, J_1_12_18, J_1_12_19, J_1_12_20, J_1_12_21, 
J_1_13_0, J_1_13_1, J_1_13_2, J_1_13_3, J_1_13_4, J_1_13_5, J_1_13_6, J_1_13_7, J_1_13_8, J_1_13_9, J_1_13_10, J_1_13_11, J_1_13_12, J_1_13_13, J_1_13_14, J_1_13_15, J_1_13_16, J_1_13_17, J_1_13_18, J_1_13_19, J_1_13_20, J_1_13_21, 
J_1_14_0, J_1_14_1, J_1_14_2, J_1_14_3, J_1_14_4, J_1_14_5, J_1_14_6, J_1_14_7, J_1_14_8, J_1_14_9, J_1_14_10, J_1_14_11, J_1_14_12, J_1_14_13, J_1_14_14, J_1_14_15, J_1_14_16, J_1_14_17, J_1_14_18, J_1_14_19, J_1_14_20, J_1_14_21, 
J_1_15_0, J_1_15_1, J_1_15_2, J_1_15_3, J_1_15_4, J_1_15_5, J_1_15_6, J_1_15_7, J_1_15_8, J_1_15_9, J_1_15_10, J_1_15_11, J_1_15_12, J_1_15_13, J_1_15_14, J_1_15_15, J_1_15_16, J_1_15_17, J_1_15_18, J_1_15_19, J_1_15_20, J_1_15_21, 
J_1_16_0, J_1_16_1, J_1_16_2, J_1_16_3, J_1_16_4, J_1_16_5, J_1_16_6, J_1_16_7, J_1_16_8, J_1_16_9, J_1_16_10, J_1_16_11, J_1_16_12, J_1_16_13, J_1_16_14, J_1_16_15, J_1_16_16, J_1_16_17, J_1_16_18, J_1_16_19, J_1_16_20, J_1_16_21, 
J_1_17_0, J_1_17_1, J_1_17_2, J_1_17_3, J_1_17_4, J_1_17_5, J_1_17_6, J_1_17_7, J_1_17_8, J_1_17_9, J_1_17_10, J_1_17_11, J_1_17_12, J_1_17_13, J_1_17_14, J_1_17_15, J_1_17_16, J_1_17_17, J_1_17_18, J_1_17_19, J_1_17_20, J_1_17_21, 
J_1_18_0, J_1_18_1, J_1_18_2, J_1_18_3, J_1_18_4, J_1_18_5, J_1_18_6, J_1_18_7, J_1_18_8, J_1_18_9, J_1_18_10, J_1_18_11, J_1_18_12, J_1_18_13, J_1_18_14, J_1_18_15, J_1_18_16, J_1_18_17, J_1_18_18, J_1_18_19, J_1_18_20, J_1_18_21, 
J_1_19_0, J_1_19_1, J_1_19_2, J_1_19_3, J_1_19_4, J_1_19_5, J_1_19_6, J_1_19_7, J_1_19_8, J_1_19_9, J_1_19_10, J_1_19_11, J_1_19_12, J_1_19_13, J_1_19_14, J_1_19_15, J_1_19_16, J_1_19_17, J_1_19_18, J_1_19_19, J_1_19_20, J_1_19_21, 
J_1_20_0, J_1_20_1, J_1_20_2, J_1_20_3, J_1_20_4, J_1_20_5, J_1_20_6, J_1_20_7, J_1_20_8, J_1_20_9, J_1_20_10, J_1_20_11, J_1_20_12, J_1_20_13, J_1_20_14, J_1_20_15, J_1_20_16, J_1_20_17, J_1_20_18, J_1_20_19, J_1_20_20, J_1_20_21, 
J_1_21_0, J_1_21_1, J_1_21_2, J_1_21_3, J_1_21_4, J_1_21_5, J_1_21_6, J_1_21_7, J_1_21_8, J_1_21_9, J_1_21_10, J_1_21_11, J_1_21_12, J_1_21_13, J_1_21_14, J_1_21_15, J_1_21_16, J_1_21_17, J_1_21_18, J_1_21_19, J_1_21_20, J_1_21_21, 
J_2_0_0, J_2_0_1, J_2_0_2, J_2_0_3, J_2_0_4, J_2_0_5, J_2_0_6, J_2_0_7, J_2_0_8, J_2_0_9, J_2_0_10, J_2_0_11, J_2_0_12, J_2_0_13, J_2_0_14, J_2_0_15, J_2_0_16, J_2_0_17, J_2_0_18, J_2_0_19, J_2_0_20, J_2_0_21, 
J_2_1_0, J_2_1_1, J_2_1_2, J_2_1_3, J_2_1_4, J_2_1_5, J_2_1_6, J_2_1_7, J_2_1_8, J_2_1_9, J_2_1_10, J_2_1_11, J_2_1_12, J_2_1_13, J_2_1_14, J_2_1_15, J_2_1_16, J_2_1_17, J_2_1_18, J_2_1_19, J_2_1_20, J_2_1_21, 
J_2_2_0, J_2_2_1, J_2_2_2, J_2_2_3, J_2_2_4, J_2_2_5, J_2_2_6, J_2_2_7, J_2_2_8, J_2_2_9, J_2_2_10, J_2_2_11, J_2_2_12, J_2_2_13, J_2_2_14, J_2_2_15, J_2_2_16, J_2_2_17, J_2_2_18, J_2_2_19, J_2_2_20, J_2_2_21, 
J_2_3_0, J_2_3_1, J_2_3_2, J_2_3_3, J_2_3_4, J_2_3_5, J_2_3_6, J_2_3_7, J_2_3_8, J_2_3_9, J_2_3_10, J_2_3_11, J_2_3_12, J_2_3_13, J_2_3_14, J_2_3_15, J_2_3_16, J_2_3_17, J_2_3_18, J_2_3_19, J_2_3_20, J_2_3_21, 
J_2_4_0, J_2_4_1, J_2_4_2, J_2_4_3, J_2_4_4, J_2_4_5, J_2_4_6, J_2_4_7, J_2_4_8, J_2_4_9, J_2_4_10, J_2_4_11, J_2_4_12, J_2_4_13, J_2_4_14, J_2_4_15, J_2_4_16, J_2_4_17, J_2_4_18, J_2_4_19, J_2_4_20, J_2_4_21, 
J_2_5_0, J_2_5_1, J_2_5_2, J_2_5_3, J_2_5_4, J_2_5_5, J_2_5_6, J_2_5_7, J_2_5_8, J_2_5_9, J_2_5_10, J_2_5_11, J_2_5_12, J_2_5_13, J_2_5_14, J_2_5_15, J_2_5_16, J_2_5_17, J_2_5_18, J_2_5_19, J_2_5_20, J_2_5_21, 
J_2_6_0, J_2_6_1, J_2_6_2, J_2_6_3, J_2_6_4, J_2_6_5, J_2_6_6, J_2_6_7, J_2_6_8, J_2_6_9, J_2_6_10, J_2_6_11, J_2_6_12, J_2_6_13, J_2_6_14, J_2_6_15, J_2_6_16, J_2_6_17, J_2_6_18, J_2_6_19, J_2_6_20, J_2_6_21, 
J_2_7_0, J_2_7_1, J_2_7_2, J_2_7_3, J_2_7_4, J_2_7_5, J_2_7_6, J_2_7_7, J_2_7_8, J_2_7_9, J_2_7_10, J_2_7_11, J_2_7_12, J_2_7_13, J_2_7_14, J_2_7_15, J_2_7_16, J_2_7_17, J_2_7_18, J_2_7_19, J_2_7_20, J_2_7_21, 
J_2_8_0, J_2_8_1, J_2_8_2, J_2_8_3, J_2_8_4, J_2_8_5, J_2_8_6, J_2_8_7, J_2_8_8, J_2_8_9, J_2_8_10, J_2_8_11, J_2_8_12, J_2_8_13, J_2_8_14, J_2_8_15, J_2_8_16, J_2_8_17, J_2_8_18, J_2_8_19, J_2_8_20, J_2_8_21, 
J_2_9_0, J_2_9_1, J_2_9_2, J_2_9_3, J_2_9_4, J_2_9_5, J_2_9_6, J_2_9_7, J_2_9_8, J_2_9_9, J_2_9_10, J_2_9_11, J_2_9_12, J_2_9_13, J_2_9_14, J_2_9_15, J_2_9_16, J_2_9_17, J_2_9_18, J_2_9_19, J_2_9_20, J_2_9_21, 
J_2_10_0, J_2_10_1, J_2_10_2, J_2_10_3, J_2_10_4, J_2_10_5, J_2_10_6, J_2_10_7, J_2_10_8, J_2_10_9, J_2_10_10, J_2_10_11, J_2_10_12, J_2_10_13, J_2_10_14, J_2_10_15, J_2_10_16, J_2_10_17, J_2_10_18, J_2_10_19, J_2_10_20, J_2_10_21, 
J_2_11_0, J_2_11_1, J_2_11_2, J_2_11_3, J_2_11_4, J_2_11_5, J_2_11_6, J_2_11_7, J_2_11_8, J_2_11_9, J_2_11_10, J_2_11_11, J_2_11_12, J_2_11_13, J_2_11_14, J_2_11_15, J_2_11_16, J_2_11_17, J_2_11_18, J_2_11_19, J_2_11_20, J_2_11_21, 
J_2_12_0, J_2_12_1, J_2_12_2, J_2_12_3, J_2_12_4, J_2_12_5, J_2_12_6, J_2_12_7, J_2_12_8, J_2_12_9, J_2_12_10, J_2_12_11, J_2_12_12, J_2_12_13, J_2_12_14, J_2_12_15, J_2_12_16, J_2_12_17, J_2_12_18, J_2_12_19, J_2_12_20, J_2_12_21, 
J_2_13_0, J_2_13_1, J_2_13_2, J_2_13_3, J_2_13_4, J_2_13_5, J_2_13_6, J_2_13_7, J_2_13_8, J_2_13_9, J_2_13_10, J_2_13_11, J_2_13_12, J_2_13_13, J_2_13_14, J_2_13_15, J_2_13_16, J_2_13_17, J_2_13_18, J_2_13_19, J_2_13_20, J_2_13_21, 
J_2_14_0, J_2_14_1, J_2_14_2, J_2_14_3, J_2_14_4, J_2_14_5, J_2_14_6, J_2_14_7, J_2_14_8, J_2_14_9, J_2_14_10, J_2_14_11, J_2_14_12, J_2_14_13, J_2_14_14, J_2_14_15, J_2_14_16, J_2_14_17, J_2_14_18, J_2_14_19, J_2_14_20, J_2_14_21, 
J_2_15_0, J_2_15_1, J_2_15_2, J_2_15_3, J_2_15_4, J_2_15_5, J_2_15_6, J_2_15_7, J_2_15_8, J_2_15_9, J_2_15_10, J_2_15_11, J_2_15_12, J_2_15_13, J_2_15_14, J_2_15_15, J_2_15_16, J_2_15_17, J_2_15_18, J_2_15_19, J_2_15_20, J_2_15_21, 
J_2_16_0, J_2_16_1, J_2_16_2, J_2_16_3, J_2_16_4, J_2_16_5, J_2_16_6, J_2_16_7, J_2_16_8, J_2_16_9, J_2_16_10, J_2_16_11, J_2_16_12, J_2_16_13, J_2_16_14, J_2_16_15, J_2_16_16, J_2_16_17, J_2_16_18, J_2_16_19, J_2_16_20, J_2_16_21, 
J_2_17_0, J_2_17_1, J_2_17_2, J_2_17_3, J_2_17_4, J_2_17_5, J_2_17_6, J_2_17_7, J_2_17_8, J_2_17_9, J_2_17_10, J_2_17_11, J_2_17_12, J_2_17_13, J_2_17_14, J_2_17_15, J_2_17_16, J_2_17_17, J_2_17_18, J_2_17_19, J_2_17_20, J_2_17_21, 
J_2_18_0, J_2_18_1, J_2_18_2, J_2_18_3, J_2_18_4, J_2_18_5, J_2_18_6, J_2_18_7, J_2_18_8, J_2_18_9, J_2_18_10, J_2_18_11, J_2_18_12, J_2_18_13, J_2_18_14, J_2_18_15, J_2_18_16, J_2_18_17, J_2_18_18, J_2_18_19, J_2_18_20, J_2_18_21, 
J_2_19_0, J_2_19_1, J_2_19_2, J_2_19_3, J_2_19_4, J_2_19_5, J_2_19_6, J_2_19_7, J_2_19_8, J_2_19_9, J_2_19_10, J_2_19_11, J_2_19_12, J_2_19_13, J_2_19_14, J_2_19_15, J_2_19_16, J_2_19_17, J_2_19_18, J_2_19_19, J_2_19_20, J_2_19_21, 
J_2_20_0, J_2_20_1, J_2_20_2, J_2_20_3, J_2_20_4, J_2_20_5, J_2_20_6, J_2_20_7, J_2_20_8, J_2_20_9, J_2_20_10, J_2_20_11, J_2_20_12, J_2_20_13, J_2_20_14, J_2_20_15, J_2_20_16, J_2_20_17, J_2_20_18, J_2_20_19, J_2_20_20, J_2_20_21, 
J_2_21_0, J_2_21_1, J_2_21_2, J_2_21_3, J_2_21_4, J_2_21_5, J_2_21_6, J_2_21_7, J_2_21_8, J_2_21_9, J_2_21_10, J_2_21_11, J_2_21_12, J_2_21_13, J_2_21_14, J_2_21_15, J_2_21_16, J_2_21_17, J_2_21_18, J_2_21_19, J_2_21_20, J_2_21_21, 
};

__global__
void J_EVALUATE(float * J, float * w, float * c)
{
const int idx = blockDim.x*blockIdx.x + threadIdx.x;
J[idx] = J_LOOKUP[idx](w, c);
};

__global__
void J_FORMAT(float ** J, float * d_J)
{
const int idx = blockDim.x*blockIdx.x + threadIdx.x;
J[idx] = &d_J[NUM_VARIABLES*NUM_VARIABLES*idx];
};

#endif