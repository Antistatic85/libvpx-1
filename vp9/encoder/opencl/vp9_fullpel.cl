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
//--------------------------------------
#include "vp9_cl_common.h"

#define DIAMOND_NUM_CANDIDATES  8
//=====   GLOBAL DEFINITIONS   =====
//--------------------------------------
__constant int nmvjointsadcost[MV_JOINTS] = {600, 300, 300, 300};

__constant MV diamond_8_points[DIAMOND_NUM_CANDIDATES] = {
    {-1, -1}, {0, -2}, {1, -1}, {2, 0}, {1, 1}, {0, 2}, {-1, 1}, {-2, 0}
  };

__constant MV diamond_4_points[4] = {
    {0, -1}, {1, 0}, { 0, 1}, {-1, 0}
  };

//=====   FUNCTION MACROS   =====
//-------------------------------------
#define CHECK_BETTER(i, offset)                              \
  {                                                          \
    thissad = intermediate_sad[i];                           \
    if (thissad < bestsad) {                                 \
      this_mv = best_mv + offset[i];                         \
      thissad += mvsad_err_cost(&this_mv, &zero_mv,          \
                    nmvsadcost_0, nmvsadcost_1, sad_per_bit);\
      if (thissad < bestsad) {                               \
        bestsad = thissad;                                   \
        best_site = i;                                       \
      }                                                      \
    }                                                        \
  }

#define CHECK_BETTER_NO_MVCOST(i)                            \
  {                                                          \
    thissad = intermediate_sad[i];                           \
    if (thissad < bestsad) {                                 \
      bestsad = thissad;                                     \
      best_site = i;                                         \
    }                                                        \
  }

//=====   FUNCTION DEFINITIONS   =====
//-------------------------------------------
inline int mv_cost_constant(MV *mv,
                            __constant int *joint_cost,
                            __global int *comp_cost_0,
                            __global int *comp_cost_1) {
  return joint_cost[vp9_get_mv_joint(mv)] + comp_cost_0[mv->row] +
      comp_cost_1[mv->col];
}

int mvsad_err_cost(MV *mv,
                   MV *ref,
                   __global int *nmvsadcost_0,
                   __global int *nmvsadcost_1,
                   int sad_per_bit) {
  MV diff;

  diff.row = mv->row - ref->row;
  diff.col = mv->col - ref->col;

  return ROUND_POWER_OF_TWO(mv_cost_constant(&diff, nmvjointsadcost,
                                             nmvsadcost_0,
                                             nmvsadcost_1) * sad_per_bit, 8);
}

int clamp_it(int value, int low, int high) {
  return value < low ? low : (value > high ? high : value);
}

void clamp_gpu_mv(MV *mv, int min_col, int max_col, int min_row, int max_row) {
  mv->col = clamp_it(mv->col, min_col, max_col);
  mv->row = clamp_it(mv->row, min_row, max_row);
}

inline int is_mv_in(INIT x, const MV mv) {
  return (mv.col >= x.mv_col_min) && (mv.col <= x.mv_col_max) &&
         (mv.row >= x.mv_row_min) && (mv.row <= x.mv_row_max);
}

//=====   KERNELS   =====
//------------------------------
__kernel
__attribute__((reqd_work_group_size((BLOCK_SIZE_IN_PIXELS / NUM_PIXELS_PER_WORKITEM) * 4,
                                    BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM,
                                    1)))
void vp9_full_pixel_search(__global uchar *ref,
                           __global uchar *cur,
                           int stride,
                           __global GPU_INPUT *gpu_input,
                           __global GPU_OUTPUT_ME *gpu_output_me,
                           __global GPU_RD_PARAMETERS *rd_parameters,
                           int mi_rows,
                           int mi_cols) {
  __local int intermediate_sad[9];
  __local short best_site_shared;
  int group_col = get_group_id(0);
  int global_row = get_global_id(1);
  int group_stride = get_num_groups(0);
  int local_col = get_local_id(0);
  int local_row = get_local_id(1);
  int global_offset = (global_row * PIXEL_ROWS_PER_WORKITEM * stride) +
                      (group_col * BLOCK_SIZE_IN_PIXELS) +
                      ((local_col >> 2) * NUM_PIXELS_PER_WORKITEM);
  global_offset += (VP9_ENC_BORDER_IN_PIXELS * stride) + VP9_ENC_BORDER_IN_PIXELS;
#if BLOCK_SIZE_IN_PIXELS == 64
  GPU_BLOCK_SIZE gpu_bsize = GPU_BLOCK_64X64;
  int group_offset = (global_row / (BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM) *
      group_stride * 4 + group_col * 2);
#else
  GPU_BLOCK_SIZE gpu_bsize = GPU_BLOCK_32X32;
  int group_offset = (global_row / (BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM) *
      group_stride + group_col);
#endif
  MV best_mv;
  int pred_mv_sad;

  gpu_input += group_offset;
  gpu_output_me += group_offset;

  if (gpu_input->do_compute != gpu_bsize)
    goto exit;

  __global GPU_RD_SEG_PARAMETERS *seg_rd_params =
      &rd_parameters->seg_rd_param[gpu_input->seg_id];

  int sad_per_bit = seg_rd_params->sad_per_bit;

  __global int *nmvsadcost_0 = rd_parameters->nmvsadcost[0] + MV_MAX;
  __global int *nmvsadcost_1 = rd_parameters->nmvsadcost[1] + MV_MAX;
  MV zero_mv = 0;
  MV pred_mv;
  INIT x;
  int global_col = get_global_id(0);
  int mi_row = (global_row * PIXEL_ROWS_PER_WORKITEM) / MI_SIZE;
  int mi_col = global_col / 4;
  int bsize;
  int bestsad;

#if BLOCK_SIZE_IN_PIXELS == 64
  mi_row = (mi_row >> 3) << 3;
  mi_col = (mi_col >> 3) << 3;
  bsize = BLOCK_64X64;
#elif BLOCK_SIZE_IN_PIXELS == 32
  mi_row = (mi_row >> 2) << 2;
  mi_col = (mi_col >> 2) << 2;
  bsize = BLOCK_32X32;
#endif

  vp9_gpu_set_mv_search_range(&x, mi_row, mi_col, mi_rows, mi_cols, bsize);

  pred_mv = gpu_input->pred_mv.as_mv;

  best_mv.col = pred_mv.col >> 3;
  best_mv.row = pred_mv.row >> 3;

  clamp_gpu_mv(&best_mv, x.mv_col_min, x.mv_col_max, x.mv_row_min , x.mv_row_max);

  int idx = local_col & 3;
  int local_offset = idx * (PIXEL_ROWS_PER_WORKITEM / 4) * stride;

  __global uchar *cur_frame = cur + (global_offset + local_offset);
  __global uchar *ref_frame = ref + (global_offset + local_offset);

  if (local_col == 0 && local_row == 0) {
    vstore8(0, 0, intermediate_sad);
    intermediate_sad[8] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Get the base SAD
  uint sad = calculate_sad_rows(&best_mv, ref_frame, cur_frame, stride,
                               PIXEL_ROWS_PER_WORKITEM / 4);
  atomic_add(intermediate_sad + 8, sad);
  barrier(CLK_LOCAL_MEM_FENCE);

  bestsad = intermediate_sad[8];

  bestsad += mvsad_err_cost(&best_mv, &zero_mv, nmvsadcost_0, nmvsadcost_1,
      sad_per_bit);

  // Get the SAD of 8-point diamond
  idx = (local_col & 3) + (local_row & 1) * 4;

  MV this_mv = best_mv + diamond_8_points[idx];

  if (is_mv_in(x, this_mv)) {
    local_offset = (local_row & 1) * PIXEL_ROWS_PER_WORKITEM * stride;
    cur_frame = cur + (global_offset - local_offset);
    ref_frame = ref + (global_offset - local_offset);

    sad = calculate_sad_rows(&this_mv, ref_frame, cur_frame, stride,
                             PIXEL_ROWS_PER_WORKITEM * 2);
    atomic_add(intermediate_sad + idx, sad);
  } else {
    intermediate_sad[idx] = INT32_MAX;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  int best_site = -1;
  int thissad;
  if (local_col == 0 && local_row == 0) {
    CHECK_BETTER(0, diamond_8_points);
    CHECK_BETTER(1, diamond_8_points);
    CHECK_BETTER(2, diamond_8_points);
    CHECK_BETTER(3, diamond_8_points);
    CHECK_BETTER(4, diamond_8_points);
    CHECK_BETTER(5, diamond_8_points);
    CHECK_BETTER(6, diamond_8_points);
    CHECK_BETTER(7, diamond_8_points);
    vstore4(0, 0, intermediate_sad);
    best_site_shared = best_site;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  best_site = best_site_shared;

  // Looping till the 8-point diamond converges
  cur_frame = cur + global_offset;
  ref_frame = ref + global_offset;
  idx = (local_col & 3);
  while (best_site != -1) {
    best_mv += diamond_8_points[best_site];
    int new_idx = best_site + idx - 1;
    new_idx = new_idx & (DIAMOND_NUM_CANDIDATES - 1);
    this_mv = best_mv + diamond_8_points[new_idx];
    if (is_mv_in(x, this_mv)) {
      sad = calculate_sad(&this_mv, ref_frame, cur_frame, stride);
      if (local_row == 0 && local_col < 4) {
        sad += mvsad_err_cost(&this_mv, &zero_mv, nmvsadcost_0, nmvsadcost_1,
                              sad_per_bit);
      }
      atomic_add(intermediate_sad + idx, sad);
    } else {
      intermediate_sad[idx] = INT32_MAX;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    short old_best_site = best_site;
    best_site = -1;
    if (local_col == 0 && local_row == 0) {
      CHECK_BETTER_NO_MVCOST(0);
      CHECK_BETTER_NO_MVCOST(1);
      CHECK_BETTER_NO_MVCOST(2);
      CHECK_BETTER_NO_MVCOST(3);
      vstore4(0, 0, intermediate_sad);
      if (best_site != -1)
        best_site = (old_best_site + best_site - 1) & (DIAMOND_NUM_CANDIDATES - 1);
      best_site_shared = best_site;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    best_site = best_site_shared;
  }

  // Get the SAD of smaller 4-point diamond
  this_mv = best_mv + diamond_4_points[idx];
  if (is_mv_in(x, this_mv)) {
    sad = calculate_sad(&this_mv, ref_frame, cur_frame, stride);
    atomic_add(intermediate_sad + idx, sad);
  } else {
    intermediate_sad[idx] = INT32_MAX;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (local_col == 0 && local_row == 0) {
    CHECK_BETTER(0, diamond_4_points);
    CHECK_BETTER(1, diamond_4_points);
    CHECK_BETTER(2, diamond_4_points);
    CHECK_BETTER(3, diamond_4_points);

    gpu_output_me->pred_mv_sad = bestsad;

    if (best_site != -1)
      best_mv += diamond_4_points[best_site];

    best_mv.row = best_mv.row * 8;
    best_mv.col = best_mv.col * 8;

    gpu_output_me->mv.as_mv = best_mv;
    gpu_output_me->rv = 0;
  }

exit:
  return;
}
