/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

//=====   OPENCL HEADER DECLARATIONS   =====
//------------------------------------------

// MV_JOINT_TYPE <vp9_entropymv.h>
typedef enum {
  MV_JOINT_ZERO = 0,             /* Zero vector */
  MV_JOINT_HNZVZ = 1,            /* Vert zero, hor nonzero */
  MV_JOINT_HZVNZ = 2,            /* Hor zero, vert nonzero */
  MV_JOINT_HNZVNZ = 3,           /* Both components nonzero */
} MV_JOINT_TYPE;

// frame transform mode <vp9_enums.h>
typedef enum {
  ONLY_4X4            = 0,        // only 4x4 transform used
  ALLOW_8X8           = 1,        // allow block transform size up to 8x8
  ALLOW_16X16         = 2,        // allow block transform size up to 16x16
  ALLOW_32X32         = 3,        // allow block transform size up to 32x32
  TX_MODE_SELECT      = 4,        // transform specified for each block
  TX_MODES            = 5,
} TX_MODE;

// Block sizes for which MV computations are done in GPU <vp9_egpu.h>
typedef enum GPU_BLOCK_SIZE {
  GPU_BLOCK_32X32 = 0,
  GPU_BLOCK_64X64 = 1,
  GPU_BLOCK_SIZES,
  GPU_BLOCK_INVALID = GPU_BLOCK_SIZES
} GPU_BLOCK_SIZE;

typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned short uint16_t;
typedef short int16_t;
typedef unsigned int  uint32_t;
typedef long int64_t;
typedef unsigned long uint64_t;

#define INT32_MAX 2147483647
#define INT64_MAX 9223372036854775807LL
#define CL_INT_MAX 2147483647

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define ROUND_POWER_OF_TWO(value, n) (((value) + (1 << ((n) - 1))) >> (n))

// BLOCK_SIZE <vp9_enums.h>
typedef uint8_t BLOCK_SIZE;
#define BLOCK_4X4     0
#define BLOCK_4X8     1
#define BLOCK_8X4     2
#define BLOCK_8X8     3
#define BLOCK_8X16    4
#define BLOCK_16X8    5
#define BLOCK_16X16   6
#define BLOCK_16X32   7
#define BLOCK_32X16   8
#define BLOCK_32X32   9
#define BLOCK_32X64  10
#define BLOCK_64X32  11
#define BLOCK_64X64  12
#define BLOCK_SIZES  13
#define BLOCK_INVALID BLOCK_SIZES

// MODES <vp9_enums.h>
typedef uint8_t PREDICTION_MODE;
#define TM_PRED    9       // True-motion
#define NEARESTMV 10
#define NEARMV    11
#define ZEROMV    12
#define NEWMV     13
#define MB_MODE_COUNT 14

// MV_REFERENCE_FRAME <vp9_blockd.h>
typedef int8_t MV_REFERENCE_FRAME;
#define NONE           -1
#define INTRA_FRAME     0
#define LAST_FRAME      1
#define GOLDEN_FRAME    2
#define ALTREF_FRAME    3
#define MAX_REF_FRAMES  4

// block transform size <vp9_enums.h>
typedef uint8_t TX_SIZE;
#define TX_4X4   ((TX_SIZE)0)   // 4x4 transform
#define TX_8X8   ((TX_SIZE)1)   // 8x8 transform
#define TX_16X16 ((TX_SIZE)2)   // 16x16 transform
#define TX_32X32 ((TX_SIZE)3)   // 32x32 transform
#define TX_SIZES ((TX_SIZE)4)

// interp filter <vp9_filter.h>
typedef uint8_t INTERP_FILTER;
#define EIGHTTAP            0
#define EIGHTTAP_SMOOTH     1
#define EIGHTTAP_SHARP      2
#define SWITCHABLE_FILTERS  3 /* Number of switchable filters */
#define BILINEAR            3
// The codec can operate in four possible inter prediction filter mode:
// 8-tap, 8-tap-smooth, 8-tap-sharp, and switching between the three.
#define SWITCHABLE_FILTER_CONTEXTS (SWITCHABLE_FILTERS + 1)
#define SWITCHABLE 4 /* should be the last one */

#define GPU_INTER_MODES 1
#define GPU_INTER_OFFSET(mode) ((mode) - ZEROMV)

#define INTER_OFFSET(mode) ((mode) - NEARESTMV)

#define VP9_ENC_BORDER_IN_PIXELS    160

#define SKIP_TXFM_NONE 0
#define SKIP_TXFM_AC_DC 1
#define SKIP_TXFM_AC_ONLY 2

typedef struct mv {
  int16_t row;
  int16_t col;
} MV;

typedef union int_mv {
  uint32_t as_int;
  MV as_mv;
} int_mv;

typedef struct GPU_INPUT {
  int do_compute;
}GPU_INPUT;

struct GPU_OUTPUT {
  int64_t dist[GPU_INTER_MODES];
  int rate[GPU_INTER_MODES];
  int_mv mv[GPU_INTER_MODES];
  unsigned int sse_y[GPU_INTER_MODES];
  unsigned int var_y[GPU_INTER_MODES];
  char interp_filter[GPU_INTER_MODES];
  char tx_size[GPU_INTER_MODES];
  char skip_txfm[GPU_INTER_MODES];
  char this_early_term[GPU_INTER_MODES];
  int pred_mv_sad;
  int rv;
} __attribute__ ((aligned(32)));
typedef struct GPU_OUTPUT GPU_OUTPUT;

typedef struct GPU_RD_PARAMETERS {
  TX_MODE tx_mode;
  int dc_dequant;
  int ac_dequant;
} GPU_RD_PARAMETERS;

__constant BLOCK_SIZE txsize_to_bsize[TX_SIZES] = {
    BLOCK_4X4,  // TX_4X4
    BLOCK_8X8,  // TX_8X8
    BLOCK_16X16,  // TX_16X16
    BLOCK_32X32,  // TX_32X32
};

__constant int b_width_log2_lookup[BLOCK_SIZES] =
  {0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4};
__constant int b_height_log2_lookup[BLOCK_SIZES] =
  {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};

__constant int num_pels_log2_lookup[BLOCK_SIZES] =
  {4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12};

__constant TX_SIZE max_txsize_lookup[BLOCK_SIZES] = {
  TX_4X4,   TX_4X4,   TX_4X4,
  TX_8X8,   TX_8X8,   TX_8X8,
  TX_16X16, TX_16X16, TX_16X16,
  TX_32X32, TX_32X32, TX_32X32, TX_32X32
};

__constant TX_SIZE tx_mode_to_biggest_tx_size[TX_MODES] = {
  TX_4X4,  // ONLY_4X4
  TX_8X8,  // ALLOW_8X8
  TX_16X16,  // ALLOW_16X16
  TX_32X32,  // ALLOW_32X32
  TX_32X32,  // TX_MODE_SELECT
};

__constant BLOCK_SIZE ss_size_lookup[BLOCK_SIZES] = {
// ss_x == 1
// ss_y == 1
  BLOCK_INVALID,
  BLOCK_INVALID,
  BLOCK_INVALID,
  BLOCK_4X4,
  BLOCK_4X8,
  BLOCK_8X4,
  BLOCK_8X8,
  BLOCK_8X16,
  BLOCK_16X8,
  BLOCK_16X16,
  BLOCK_16X32,
  BLOCK_32X16,
  BLOCK_32X32,
};

// NOTE: The tables below must be of the same size.

// The functions described below are sampled at the four most significant
// bits of x^2 + 8 / 256.

// Normalized rate:
// This table models the rate for a Laplacian source with given variance
// when quantized with a uniform quantizer with given stepsize. The
// closed form expression is:
// Rn(x) = H(sqrt(r)) + sqrt(r)*[1 + H(r)/(1 - r)],
// where r = exp(-sqrt(2) * x) and x = qpstep / sqrt(variance),
// and H(x) is the binary entropy function.
__constant int rate_tab_q10[] = {
  65536,  6086,  5574,  5275,  5063,  4899,  4764,  4651,
   4553,  4389,  4255,  4142,  4044,  3958,  3881,  3811,
   3748,  3635,  3538,  3453,  3376,  3307,  3244,  3186,
   3133,  3037,  2952,  2877,  2809,  2747,  2690,  2638,
   2589,  2501,  2423,  2353,  2290,  2232,  2179,  2130,
   2084,  2001,  1928,  1862,  1802,  1748,  1698,  1651,
   1608,  1530,  1460,  1398,  1342,  1290,  1243,  1199,
   1159,  1086,  1021,   963,   911,   864,   821,   781,
    745,   680,   623,   574,   530,   490,   455,   424,
    395,   345,   304,   269,   239,   213,   190,   171,
    154,   126,   104,    87,    73,    61,    52,    44,
     38,    28,    21,    16,    12,    10,     8,     6,
      5,     3,     2,     1,     1,     1,     0,     0,
};

// Normalized distortion:
// This table models the normalized distortion for a Laplacian source
// with given variance when quantized with a uniform quantizer
// with given stepsize. The closed form expression is:
// Dn(x) = 1 - 1/sqrt(2) * x / sinh(x/sqrt(2))
// where x = qpstep / sqrt(variance).
// Note the actual distortion is Dn * variance.
__constant int dist_tab_q10[] = {
     0,     0,     1,     1,     1,     2,     2,     2,
     3,     3,     4,     5,     5,     6,     7,     7,
     8,     9,    11,    12,    13,    15,    16,    17,
    18,    21,    24,    26,    29,    31,    34,    36,
    39,    44,    49,    54,    59,    64,    69,    73,
    78,    88,    97,   106,   115,   124,   133,   142,
   151,   167,   184,   200,   215,   231,   245,   260,
   274,   301,   327,   351,   375,   397,   418,   439,
   458,   495,   528,   559,   587,   613,   637,   659,
   680,   717,   749,   777,   801,   823,   842,   859,
   874,   899,   919,   936,   949,   960,   969,   977,
   983,   994,  1001,  1006,  1010,  1013,  1015,  1017,
  1018,  1020,  1022,  1022,  1023,  1023,  1023,  1024,
};

__constant int xsq_iq_q10[] = {
       0,      4,      8,     12,     16,     20,     24,     28,
      32,     40,     48,     56,     64,     72,     80,     88,
      96,    112,    128,    144,    160,    176,    192,    208,
     224,    256,    288,    320,    352,    384,    416,    448,
     480,    544,    608,    672,    736,    800,    864,    928,
     992,   1120,   1248,   1376,   1504,   1632,   1760,   1888,
    2016,   2272,   2528,   2784,   3040,   3296,   3552,   3808,
    4064,   4576,   5088,   5600,   6112,   6624,   7136,   7648,
    8160,   9184,  10208,  11232,  12256,  13280,  14304,  15328,
   16352,  18400,  20448,  22496,  24544,  26592,  28640,  30688,
   32736,  36832,  40928,  45024,  49120,  53216,  57312,  61408,
   65504,  73696,  81888,  90080,  98272, 106464, 114656, 122848,
  131040, 147424, 163808, 180192, 196576, 212960, 229344, 245728,
};


//=====   OPENCL FUNCTION MACROS   =====
//--------------------------------------
#define MODEL_RD_FOR_SB_Y_LARGE                                             \
  if (tx_mode == TX_MODE_SELECT) {                                          \
    if (sse > (var << 2))                                                   \
      tx_size = MIN(max_txsize_lookup[bsize],                               \
                    tx_mode_to_biggest_tx_size[tx_mode]);                   \
    else                                                                    \
      tx_size = TX_8X8;                                                     \
    if (tx_size > TX_16X16)                                                 \
      tx_size = TX_16X16;                                                   \
  } else {                                                                  \
    tx_size = MIN(max_txsize_lookup[bsize],                                 \
                  tx_mode_to_biggest_tx_size[tx_mode]);                     \
  }                                                                         \
  {                                                                         \
    int num8x8 = 1 << (bw + bh - 2);                                        \
    uint32_t sse16x16[16];                                                  \
    int sum16x16[16];                                                       \
    uint32_t var16x16[16];                                                  \
    int num16x16 = num8x8 >> 2;                                             \
                                                                            \
    uint32_t sse32x32[4];                                                   \
    int sum32x32[4];                                                        \
    uint32_t var32x32[4];                                                   \
    int num32x32 = num8x8 >> 4;                                             \
                                                                            \
    int ac_test = 1;                                                        \
    int dc_test = 1;                                                        \
    int k;                                                                  \
    int num = ((tx_size == TX_8X8) ? num8x8 : ((tx_size == TX_16X16) ?      \
        num16x16 : num32x32));                                              \
    uint32_t *sse_tx = ((tx_size == TX_8X8) ? sse8x8 :                      \
        ((tx_size == TX_16X16) ? sse16x16 : sse32x32));                     \
    uint32_t *var_tx = ((tx_size == TX_8X8) ? var8x8 :                      \
        ((tx_size == TX_16X16) ? var16x16 : var32x32));                     \
    if (tx_size >= TX_16X16) {                                              \
      calculate_variance(bw, bh, TX_8X8, sse8x8, sum8x8, var16x16, sse16x16,\
                         sum16x16);                                         \
    }                                                                       \
    if (tx_size == TX_32X32) {                                              \
      calculate_variance(bw, bh, TX_16X16, sse16x16, sum16x16, var32x32,    \
                         sse32x32, sum32x32);                               \
    }                                                                       \
    skip_txfm = SKIP_TXFM_NONE;                                             \
    for (k = 0; k < num; k++) {                                             \
      if (!(var_tx[k] < ac_thr || var == 0)) {                              \
        ac_test = 0;                                                        \
        break;                                                              \
      }                                                                     \
    }                                                                       \
    for (k = 0; k < num; k++) {                                             \
      if (!(sse_tx[k] - var_tx[k] < dc_thr || sse == var)) {                \
        dc_test = 0;                                                        \
        break;                                                              \
      }                                                                     \
    }                                                                       \
    if (ac_test) {                                                          \
      skip_txfm = SKIP_TXFM_AC_ONLY;                                        \
      if (dc_test) {                                                        \
        skip_txfm = SKIP_TXFM_AC_DC;                                        \
      }                                                                     \
    } else if (dc_test) {                                                   \
      skip_dc = 1;                                                          \
    }                                                                       \
  }                                                                         \
  if (skip_txfm == SKIP_TXFM_AC_DC) {                                       \
    actual_rate = 0;                                                        \
    actual_dist = sse << 4;                                                 \
  } else {                                                                  \
    if (!skip_dc) {                                                         \
      vp9_model_rd_from_var_lapndz(sse - var, num_pels_log2_lookup[bsize],  \
                                   dc_quant >> 3, &rate, &dist);            \
    }                                                                       \
    if (!skip_dc) {                                                         \
      actual_rate = rate >> 1;                                              \
      actual_dist = dist << 3;                                              \
    } else {                                                                \
      actual_rate = 0;                                                      \
      actual_dist = (sse - var) << 4;                                       \
    }                                                                       \
    vp9_model_rd_from_var_lapndz(var, num_pels_log2_lookup[bsize],          \
                                 ac_quant >> 3, &rate, &dist);              \
    actual_rate += rate;                                                    \
    actual_dist += dist << 4;                                               \
  }


//=====   OPENCL FUNCTION DEFINITIONS   =====
//-------------------------------------------
inline int get_msb(unsigned int n) {
  return 31 ^ clz(n);
}

void model_rd_norm(int xsq_q10, int *r_q10, int *d_q10) {
  const int tmp = (xsq_q10 >> 2) + 8;
  const int k = get_msb(tmp) - 3;
  const int xq = (k << 3) + ((tmp >> k) & 0x7);
  const int one_q10 = 1 << 10;
  const int a_q10 = ((xsq_q10 - xsq_iq_q10[xq]) << 10) >> (2 + k);
  const int b_q10 = one_q10 - a_q10;
  *r_q10 = (rate_tab_q10[xq] * b_q10 + rate_tab_q10[xq + 1] * a_q10) >> 10;
  *d_q10 = (dist_tab_q10[xq] * b_q10 + dist_tab_q10[xq + 1] * a_q10) >> 10;
}

void vp9_model_rd_from_var_lapndz(unsigned int var, unsigned int n_log2,
                                  unsigned int qstep, int *rate,
                                  int64_t *dist) {
  // This function models the rate and distortion for a Laplacian
  // source with given variance when quantized with a uniform quantizer
  // with given stepsize. The closed form expressions are in:
  // Hang and Chen, "Source Model for transform video coder and its
  // application - Part I: Fundamental Theory", IEEE Trans. Circ.
  // Sys. for Video Tech., April 1997.
  if (var == 0) {
    *rate = 0;
    *dist = 0;
  } else {
    int d_q10, r_q10;
    const uint32_t MAX_XSQ_Q10 = 245727;
    const uint64_t xsq_q10_64 =
        (((uint64_t)qstep * qstep << (n_log2 + 10)) + (var >> 1)) / var;
    const int xsq_q10 = (int)MIN(xsq_q10_64, MAX_XSQ_Q10);
    model_rd_norm(xsq_q10, &r_q10, &d_q10);
    *rate = ((r_q10 << n_log2) + 2) >> 2;
    *dist = (var * (int64_t)d_q10 + 512) >> 10;
  }
}

void calculate_variance(int bw, int bh, TX_SIZE tx_size,
                        uint32_t *sse_i, int *sum_i,
                        uint32_t *var_o, uint32_t *sse_o,
                        int *sum_o) {
  BLOCK_SIZE unit_size = txsize_to_bsize[tx_size];
  int nw = 1 << (bw - b_width_log2_lookup[unit_size]);
  int nh = 1 << (bh - b_height_log2_lookup[unit_size]);
  int i, j, k = 0;

  for (i = 0; i < nh; i += 2) {
    for (j = 0; j < nw; j += 2) {
      sse_o[k] = sse_i[i * nw + j] + sse_i[i * nw + j + 1] +
          sse_i[(i + 1) * nw + j] + sse_i[(i + 1) * nw + j + 1];
      sum_o[k] = sum_i[i * nw + j] + sum_i[i * nw + j + 1] +
          sum_i[(i + 1) * nw + j] + sum_i[(i + 1) * nw + j + 1];
      var_o[k] = sse_o[k] - (((uint32_t)sum_o[k] * sum_o[k]) >>
          (b_width_log2_lookup[unit_size] +
              b_height_log2_lookup[unit_size] + 6));
      k++;
    }
  }
}

void vp9_variance_bxw(__global uchar *ref_frame,
                      __global uchar *cur_frame,
                      int *sum,
                      uint32_t *sse,
                      int stride,
                      int width, int height) {
  uchar8 src_load8, pred_load8;
  short8 src_data8, pred_data8, e_data8, a_data8 = 0;
  int4 b_data4;
  int4 e_data4;
  uint4 ase_data4 = 0;
  int row, col;

  for (row = 0; row < height; row += 1) {
    for (col = 0; col < width; col += 8) {
      src_load8 = vload8(0, cur_frame);
      pred_load8 = vload8(0, ref_frame);

      src_data8 = convert_short8(src_load8);
      pred_data8 = convert_short8(pred_load8);
      e_data8 = src_data8 - pred_data8;

      a_data8 += e_data8;

      e_data4 = convert_int4(e_data8.s0123);
      ase_data4 +=  convert_uint4(e_data4 * e_data4);
      e_data4 = convert_int4(e_data8.s4567);
      ase_data4 +=  convert_uint4(e_data4 * e_data4);

      cur_frame += 8;
      ref_frame += 8;
    }
    cur_frame += stride - width;
    ref_frame += stride - width;
  }

  b_data4.s0123 = convert_int4(a_data8.s0123) + convert_int4(a_data8.s4567);
  b_data4.s01 = b_data4.s01 + b_data4.s23;
  *sum = (int)b_data4.s0 + b_data4.s1;

  ase_data4.s01 = ase_data4.s01 + ase_data4.s23;
  ase_data4.s0 = ase_data4.s0 + ase_data4.s1;
  *sse = ase_data4.s0;
}

void block_variance(__global uchar *ref_frame,
                    __global uchar *cur_frame,
                    int *sum,
                    uint32_t *sse,
                    int stride,
                    unsigned int *sse8x8,
                    int *sum8x8, unsigned int *var8x8) {
  int w, h;
  int i, j, k = 0;

  w = h = BLOCK_SIZE_IN_PIXELS;

  *sse = 0;
  *sum = 0;

  for (i = 0; i < h; i += 8) {
    for (j = 0; j < w; j += 8) {
      vp9_variance_bxw(ref_frame + stride * i + j,
                       cur_frame + stride * i + j,
                       &sum8x8[k], &sse8x8[k], stride, 8, 8);
      *sse += sse8x8[k];
      *sum += sum8x8[k];
      var8x8[k] = sse8x8[k] - (((unsigned int)sum8x8[k] * sum8x8[k]) >> 6);
      k++;
    }
  }
}

TX_SIZE get_uv_tx_size(tx_size, bsize) {
  BLOCK_SIZE plane_bsize = ss_size_lookup[bsize];
  return MIN(tx_size, max_txsize_lookup[plane_bsize]);
}


//=====   OPENCL KERNELS   =====
//------------------------------
__kernel
void vp9_zero_motion_search(__global uchar *ref,
                            __global uchar *cur,
                            int stride,
                            __global GPU_INPUT *gpu_input,
                            __global GPU_OUTPUT *gpu_output,
                            __global GPU_RD_PARAMETERS *rd_parameters,
                            int64_t yplane_size, int64_t uvplane_size) {
  __global uchar *ref_frame = ref;
  __global uchar *cur_frame = cur;
  uint32_t sse8x8[64], var8x8[64];
  int sum8x8[64];
  uint32_t sse, var;
  int sum;
  int bw, bh;
  int frame_offset;
  int global_col = get_global_id( 0 );
  int global_row = get_global_id( 1 );
  int global_stride = get_global_size( 0 );
  BLOCK_SIZE bsize;
  int this_early_term = 0;

  gpu_input += (global_row * global_stride + global_col);
  gpu_output += (global_row * global_stride + global_col);

  if (!gpu_input->do_compute)
    goto exit;

  frame_offset = (VP9_ENC_BORDER_IN_PIXELS * stride) + VP9_ENC_BORDER_IN_PIXELS;
  frame_offset += (global_row * stride * BLOCK_SIZE_IN_PIXELS) + (global_col * BLOCK_SIZE_IN_PIXELS);

  cur += frame_offset;
  ref += frame_offset;

#if BLOCK_SIZE_IN_PIXELS == 64
  bsize = BLOCK_64X64;
#elif BLOCK_SIZE_IN_PIXELS == 32
  bsize = BLOCK_32X32;
#endif

  bw = b_width_log2_lookup[bsize];
  bh = b_height_log2_lookup[bsize];

  block_variance(ref, cur, &sum, &sse, stride, sse8x8, sum8x8, var8x8);
  var = sse - (((int64_t)sum * sum) >> (bw + bh + 4));

  {
    TX_SIZE tx_size;
    TX_MODE tx_mode = rd_parameters->tx_mode;

    int dc_quant = rd_parameters->dc_dequant;
    int ac_quant = rd_parameters->ac_dequant;
    int64_t dc_thr = dc_quant * dc_quant >> 6;
    int64_t ac_thr = ac_quant * ac_quant >> 6;
    int skip_txfm;
    int skip_dc = 0;

    int rate, actual_rate;
    int64_t dist, actual_dist;

    MODEL_RD_FOR_SB_Y_LARGE

    if (skip_txfm == SKIP_TXFM_AC_DC) {
      TX_SIZE uv_tx_size = get_uv_tx_size(tx_size, bsize);
      BLOCK_SIZE unit_size = txsize_to_bsize[uv_tx_size];
      BLOCK_SIZE uv_bsize = ss_size_lookup[bsize];
      int uv_bw = b_width_log2_lookup[uv_bsize];
      int uv_bh = b_height_log2_lookup[uv_bsize];
      int sf = (uv_bw - b_width_log2_lookup[unit_size]) +
          (uv_bh - b_height_log2_lookup[unit_size]);
      uint32_t uv_dc_thr = dc_quant * dc_quant >> (6 - sf);
      uint32_t uv_ac_thr = ac_quant * ac_quant >> (6 - sf);
      int i;

      frame_offset = ((VP9_ENC_BORDER_IN_PIXELS >> 1) * (stride >> 1)) +
          (VP9_ENC_BORDER_IN_PIXELS >> 1);
      frame_offset += (global_row * (stride >> 1) * (BLOCK_SIZE_IN_PIXELS >> 1)) +
          (global_col * (BLOCK_SIZE_IN_PIXELS >> 1));

      for (i = 0; i < 2; i++) {
        __global uchar *ref_uv = ref_frame + yplane_size + i * uvplane_size;
        __global uchar *cur_uv = cur_frame + yplane_size + i * uvplane_size;

        ref_uv += frame_offset;
        cur_uv += frame_offset;

        vp9_variance_bxw(ref_uv,
                         cur_uv,
                         &sum8x8[0], &sse8x8[0],
                         stride >> 1,
                         (1 << (uv_bw + 2)), (1 << (uv_bh + 2)));
        var8x8[0] = sse8x8[0] - (((int64_t)sum8x8[0] * sum8x8[0]) >> (uv_bw + uv_bh + 4));

        if ((var8x8[0] < uv_ac_thr || var8x8[0] == 0) &&
            (sse8x8[0] - var8x8[0] < uv_dc_thr || sse8x8[0] == var8x8[0])) {
          if (i == 1) {
            this_early_term = 1;
            break;
          }
        } else {
          break;
        }
      }
    }

    gpu_output->rate[GPU_INTER_OFFSET(ZEROMV)] = actual_rate;
    gpu_output->dist[GPU_INTER_OFFSET(ZEROMV)] = actual_dist;
    gpu_output->mv[GPU_INTER_OFFSET(ZEROMV)].as_int = 0;
    gpu_output->sse_y[GPU_INTER_OFFSET(ZEROMV)] = sse;
    gpu_output->var_y[GPU_INTER_OFFSET(ZEROMV)] = var;
    gpu_output->interp_filter[GPU_INTER_OFFSET(ZEROMV)] = EIGHTTAP;
    gpu_output->tx_size[GPU_INTER_OFFSET(ZEROMV)] = tx_size;
    gpu_output->skip_txfm[GPU_INTER_OFFSET(ZEROMV)] = skip_txfm;
    gpu_output->this_early_term[GPU_INTER_OFFSET(ZEROMV)] = this_early_term;
  }

  exit:
  return;
}
