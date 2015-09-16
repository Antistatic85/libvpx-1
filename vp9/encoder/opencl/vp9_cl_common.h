/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

//=====   HEADER DECLARATIONS   =====
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

#define NUM_PIXELS_PER_WORKITEM 8
#define LOCAL_STRIDE (BLOCK_SIZE_IN_PIXELS / NUM_PIXELS_PER_WORKITEM)

#define GPU_INTER_MODES 2
#define GPU_INTER_OFFSET(mode) ((mode) - ZEROMV)

#define MAX_PATTERN_SCALES 2
#define MAX_PATTERN_CANDIDATES 8
#define PATTERN_CANDIDATES_REF 3
#define MAX_MVSEARCH_STEPS 11
#define MAX_FULL_PEL_VAL ((1 << (MAX_MVSEARCH_STEPS - 1)) - 1)

#define FILTER_BITS 7
#define SUBPEL_TAPS 8
#define SUBPEL_BITS 3
#define SUBPEL_MASK ((1 << SUBPEL_BITS) - 1)
#define SWITCHABLE_FILTERS  3

#define MV_JOINTS 4
#define MV_CLASSES     11
#define CLASS0_BITS    1
#define MV_MAX_BITS    (MV_CLASSES + CLASS0_BITS + 2)
#define MV_MAX         ((1 << MV_MAX_BITS) - 1)
#define MV_VALS        ((MV_MAX << 1) + 1)
#define MV_IN_USE_BITS 14
#define MV_UPP   ((1 << MV_IN_USE_BITS) - 1)
#define MV_LOW   (-(1 << MV_IN_USE_BITS))
#define MV_COST_WEIGHT      108

#define VP9_INTERP_EXTEND 4
#define VP9_ENC_BORDER_IN_PIXELS    160
#define MI_SIZE 8

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

typedef struct initvalues {
  int mv_row_min;
  int mv_row_max;
  int mv_col_min;
  int mv_col_max;
}INIT;

typedef struct GPU_INPUT {
  int_mv pred_mv;
  char do_compute;
  char seg_id;
} GPU_INPUT;

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

typedef struct GPU_RD_SEG_PARAMETERS {
  int rd_mult;
  int dc_dequant;
  int ac_dequant;

  int sad_per_bit;

  int error_per_bit;
} GPU_RD_SEG_PARAMETERS;

typedef struct GPU_RD_PARAMETERS {
  TX_MODE tx_mode;
  int rd_div;
  unsigned int inter_mode_cost[GPU_INTER_MODES];
  int switchable_interp_costs[SWITCHABLE_FILTERS];

  int nmvjointcost[MV_JOINTS];
  int nmvsadcost[2][MV_VALS];

  int mvcost[2][MV_VALS];

  // Currently supporting only 2 segments in GPU
  GPU_RD_SEG_PARAMETERS seg_rd_param[2];
} GPU_RD_PARAMETERS;


//=====   GLOBAL DEFINITIONS   =====
//--------------------------------------
__constant int num_8x8_blocks_wide_lookup[BLOCK_SIZES] =
  {1, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8};
__constant int num_8x8_blocks_high_lookup[BLOCK_SIZES] =
  {1, 1, 1, 1, 2, 1, 2, 4, 2, 4, 8, 4, 8};

//=====   FUNCTION MACROS   =====
//--------------------------------------
#define RDCOST(RM, DM, R, D) (((128 + ((int64_t)R) * (RM)) >> 8) + (D << DM))

//=====   FUNCTION DEFINITIONS   =====
//-------------------------------------------
inline MV_JOINT_TYPE vp9_get_mv_joint(const MV *mv) {
  if (mv->row == 0) {
    return mv->col == 0 ? MV_JOINT_ZERO : MV_JOINT_HNZVZ;
  } else {
    return mv->col == 0 ? MV_JOINT_HZVNZ : MV_JOINT_HNZVNZ;
  }
}

inline int mv_cost(MV *mv, __global int *joint_cost,
                   __global int *comp_cost_0, __global int *comp_cost_1) {
  return joint_cost[vp9_get_mv_joint(mv)] +
      comp_cost_0[mv->row] + comp_cost_1[mv->col];
}

void vp9_gpu_set_mv_search_range(INIT *x, int mi_row, int mi_col, int mi_rows,
                                 int mi_cols, int bsize) {

  int mi_width  = num_8x8_blocks_wide_lookup[bsize];
  int mi_height = num_8x8_blocks_high_lookup[bsize];

  // Set up limit values for MV components.
  // MV beyond the range do not produce new / different prediction block.
  x->mv_row_min = -(((mi_row + mi_height) * MI_SIZE) + VP9_INTERP_EXTEND);
  x->mv_col_min = -(((mi_col + mi_width) * MI_SIZE) + VP9_INTERP_EXTEND);
  x->mv_row_max = (mi_rows - mi_row) * MI_SIZE + VP9_INTERP_EXTEND;
  x->mv_col_max = (mi_cols - mi_col) * MI_SIZE + VP9_INTERP_EXTEND;
}
