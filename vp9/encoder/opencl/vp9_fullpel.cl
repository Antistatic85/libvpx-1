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

//=====   GLOBAL DEFINITIONS   =====
//--------------------------------------
__constant int nmvjointsadcost[MV_JOINTS] = {600, 300, 300, 300};

__constant int hex_num_candidates[MAX_PATTERN_SCALES] = {8, 6};

__constant MV hex_candidates[MAX_PATTERN_SCALES][MAX_PATTERN_CANDIDATES] = {
    {{-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, { 0, 1}, { -1, 1}, {-1, 0}},
    {{-1, -2}, {1, -2}, {2, 0}, {1, 2}, {-1, 2}, { -2, 0}},
  };

//=====   FUNCTION MACROS   =====
//-------------------------------------
#define CHECK_BETTER                                         \
  {                                                          \
    if (thissad < bestsad) {                                 \
     thissad += mvsad_err_cost(&this_mv,&fcenter_mv,         \
                    nmvsadcost_0,nmvsadcost_1,sad_per_bit);  \
      if (thissad < bestsad) {                               \
        bestsad = thissad;                                   \
        best_site = i;                                       \
      }                                                      \
    }                                                        \
  }

//=====   FUNCTION DEFINITIONS   =====
//-------------------------------------------
ushort calculate_sad(MV *currentmv,
                     __global uchar *ref_frame,
                     __global uchar *cur_frame,
                     int stride) {
  __global uchar *tmp_ref, *tmp_cur;
  uchar8 ref, cur;
  ushort8 sad = 0;
  int buffer_offset;
  int row;

  buffer_offset = (currentmv->row * stride) + currentmv->col;
  tmp_ref = ref_frame + buffer_offset;
  tmp_cur = cur_frame;

  for (row = 0; row < PIXEL_ROWS_PER_WORKITEM; row++) {
    ref = vload8(0, tmp_ref);
    cur = vload8(0, tmp_cur);

    sad += abs_diff(convert_ushort8(ref), convert_ushort8(cur));

    tmp_ref += stride;
    tmp_cur += stride;
  }

  ushort4 final_sad = convert_ushort4(sad.s0123) + convert_ushort4(sad.s4567);
  final_sad.s01 = final_sad.s01 + final_sad.s23;

  return (final_sad.s0 + final_sad.s1);
}

int get_sad(__global uchar *ref_frame, __global uchar *cur_frame,
            int stride, __local int* intermediate_sad, MV this_mv) {
  int local_col = get_local_id(0);
  int local_row = get_local_id(1);
  int sad;

  barrier(CLK_LOCAL_MEM_FENCE);
  intermediate_sad[0] = 0;

  sad = calculate_sad(&this_mv, ref_frame, cur_frame, stride);

  barrier(CLK_LOCAL_MEM_FENCE);
  atomic_add(intermediate_sad, sad);

  barrier(CLK_LOCAL_MEM_FENCE);
  return intermediate_sad[0];
}

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

inline int gpu_check_bounds(INIT *x, int row, int col, int range) {
  return ((row - range) >= x->mv_row_min) &
      ((row + range) <= x->mv_row_max) &
      ((col - range) >= x->mv_col_min) &
      ((col + range) <= x->mv_col_max);
}

int is_mv_in(INIT *x, const MV *mv) {
  return (mv->col >= x->mv_col_min) &&
      (mv->col <= x->mv_col_max) &&
      (mv->row >= x->mv_row_min) &&
      (mv->row <= x->mv_row_max);
}

MV inline full_pixel_pattern_search(__global uchar *ref_frame,
                                    __global uchar *cur_frame,
                                    int stride,
                                    __local int* intermediate_sad,
                                    MV best_mv,
                                    MV fcenter_mv,
                                    __global int *nmvsadcost_0,
                                    __global int *nmvsadcost_1,
                                    INIT *x,
                                    int sad_per_bit,
                                    int *pbestsad,
                                    int pattern) {
  MV this_mv;
  int best_site = -1;
  short br, bc;
  int thissad, bestsad;
  int i, k;
  int next_chkpts_indices[PATTERN_CANDIDATES_REF];

  br = best_mv.row;
  bc = best_mv.col;
  bestsad = *pbestsad;
  best_site = -1;
  if (gpu_check_bounds(x, br, bc, 1 << pattern)) {
    for (i = 0; i < hex_num_candidates[pattern]; i++) {
      this_mv.row = br + hex_candidates[pattern][i].row;
      this_mv.col = bc + hex_candidates[pattern][i].col;

      thissad = get_sad(ref_frame, cur_frame, stride, intermediate_sad, this_mv);

      CHECK_BETTER
    }
  } else {
    for (i = 0; i < hex_num_candidates[pattern]; i++) {
      this_mv.row = br + hex_candidates[pattern][i].row;
      this_mv.col = bc + hex_candidates[pattern][i].col;

      if (!is_mv_in(x, &this_mv)) {
        continue;
      }

      thissad = get_sad(ref_frame, cur_frame, stride, intermediate_sad, this_mv);
      CHECK_BETTER
    }
  }

  if (best_site == -1) {
    goto exit;
  } else {
    br += hex_candidates[pattern][best_site].row;
    bc += hex_candidates[pattern][best_site].col;
    k = best_site;
  }

  do {
    best_site = -1;
    next_chkpts_indices[0] = (k == 0) ? hex_num_candidates[pattern] - 1 : k - 1;
    next_chkpts_indices[1] = k;
    next_chkpts_indices[2] = (k == hex_num_candidates[pattern] - 1) ? 0 : k + 1;

    if (gpu_check_bounds(x, br, bc, 1 << pattern)) {
      for (i = 0; i < PATTERN_CANDIDATES_REF; i++) {
        this_mv.row = br + hex_candidates[pattern][next_chkpts_indices[i]].row;
        this_mv.col = bc + hex_candidates[pattern][next_chkpts_indices[i]].col;

        thissad = get_sad(ref_frame, cur_frame, stride, intermediate_sad, this_mv);
        CHECK_BETTER
      }
    } else {
      for (i = 0; i < PATTERN_CANDIDATES_REF; i++) {
        this_mv.row = br + hex_candidates[pattern][next_chkpts_indices[i]].row;
        this_mv.col = bc + hex_candidates[pattern][next_chkpts_indices[i]].col;

        if (!is_mv_in(x, &this_mv)) {
          continue;
        }

        thissad = get_sad(ref_frame, cur_frame, stride, intermediate_sad, this_mv);
        CHECK_BETTER
      }
    }

    if (best_site != -1) {
      k = next_chkpts_indices[best_site];
      br += hex_candidates[pattern][k].row;
      bc += hex_candidates[pattern][k].col;
    }
  }while(best_site != -1);
exit:
  *pbestsad = bestsad;
  best_mv.row = br;
  best_mv.col = bc;

  return best_mv;
}

int clamp_it(int value, int low, int high) {
  return value < low ? low : (value > high ? high : value);
}

void clamp_gpu_mv(MV *mv, int min_col, int max_col, int min_row, int max_row) {
  mv->col = clamp_it(mv->col, min_col, max_col);
  mv->row = clamp_it(mv->row, min_row, max_row);
}

MV full_pixel_search(__global uchar *ref_frame,
                     __global uchar *cur_frame,
                     int stride,
                     __local int* intermediate_sad,
                     MV best_mv,
                     MV fcenter_mv,
                     __global int *nmvsadcost_0,
                     __global int *nmvsadcost_1,
                     INIT *x,
                     int sad_per_bit) {
  int thissad, bestsad;

  clamp_gpu_mv(&best_mv, x->mv_col_min, x->mv_col_max, x->mv_row_min , x->mv_row_max);

  bestsad = get_sad(ref_frame, cur_frame, stride, intermediate_sad, best_mv);

  bestsad += mvsad_err_cost(&best_mv, &fcenter_mv, nmvsadcost_0, nmvsadcost_1,
      sad_per_bit);

  // Search with pattern = 1
  best_mv = full_pixel_pattern_search(ref_frame, cur_frame, stride,
                                      intermediate_sad, best_mv, fcenter_mv,
                                      nmvsadcost_0, nmvsadcost_1,
                                      x, sad_per_bit,
                                      &bestsad, 1);
  // Search with pattern = 0
  best_mv = full_pixel_pattern_search(ref_frame, cur_frame, stride,
                                      intermediate_sad, best_mv, fcenter_mv,
                                      nmvsadcost_0, nmvsadcost_1,
                                      x, sad_per_bit,
                                      &bestsad, 0);

  best_mv.row = best_mv.row * 8;
  best_mv.col = best_mv.col * 8;

  return best_mv;
}

void vp9_set_mv_search_range_step2(INIT *x, const MV *mv) {
  int col_min = (mv->col >> 3) - MAX_FULL_PEL_VAL + (mv->col & 7 ? 1 : 0);
  int row_min = (mv->row >> 3) - MAX_FULL_PEL_VAL + (mv->row & 7 ? 1 : 0);
  int col_max = (mv->col >> 3) + MAX_FULL_PEL_VAL;
  int row_max = (mv->row >> 3) + MAX_FULL_PEL_VAL;

  col_min = MAX(col_min, (MV_LOW >> 3) + 1);
  row_min = MAX(row_min, (MV_LOW >> 3) + 1);
  col_max = MIN(col_max, (MV_UPP >> 3) - 1);
  row_max = MIN(row_max, (MV_UPP >> 3) - 1);

  // Get intersection of UMV window and valid MV window to reduce # of checks
  // in diamond search.
  if (x->mv_col_min < col_min)
    x->mv_col_min = col_min;
  if (x->mv_col_max > col_max)
    x->mv_col_max = col_max;
  if (x->mv_row_min < row_min)
    x->mv_row_min = row_min;
  if (x->mv_row_max > row_max)
    x->mv_row_max = row_max;
}

inline MV get_best_mv(__global uchar *ref_frame, __global uchar *cur_frame,
                      int stride, __local int* intermediate_sad,
                      MV candidate_a_mv, MV candidate_b_mv, int *pred_mv_sad) {
  MV this_mv, best_mv;
  int thissad, bestsad = CL_INT_MAX;

#if 0
  this_mv.row = (candidate_a_mv.row + 3 + (candidate_a_mv.row >= 0)) >> 3;
  this_mv.col = (candidate_a_mv.col + 3 + (candidate_a_mv.col >= 0)) >> 3;

  thissad = get_sad(ref_frame, cur_frame, stride, intermediate_sad, this_mv);

  if (thissad < bestsad) {
    bestsad = thissad;
    best_mv.col = candidate_a_mv.col >> 3;
    best_mv.row = candidate_a_mv.row >> 3;
  }
#endif

  this_mv.row = (candidate_b_mv.row + 3 + (candidate_b_mv.row >= 0)) >> 3;
  this_mv.col = (candidate_b_mv.col + 3 + (candidate_b_mv.col >= 0)) >> 3;

  thissad = get_sad(ref_frame, cur_frame, stride, intermediate_sad, this_mv);

  if (thissad < bestsad) {
    bestsad = thissad;
    best_mv.col = candidate_b_mv.col >> 3;
    best_mv.row = candidate_b_mv.row >> 3;
  }

  *pred_mv_sad = bestsad;

  return best_mv;
}

MV combined_motion_search(__global uchar *ref_frame,
                          __global uchar *cur_frame,
                          __global GPU_INPUT *gpu_input,
                          __global GPU_RD_PARAMETERS *rd_parameters,
                          int sad_per_bit,
                          int stride,
                          int mi_rows,
                          int mi_cols,
                          int *pred_mv_sad,
                          __local int *intermediate_int) {
  __global int *nmvsadcost_0 = rd_parameters->nmvsadcost[0] + MV_MAX;
  __global int *nmvsadcost_1 = rd_parameters->nmvsadcost[1] + MV_MAX;
  int_mv zero_mv;
  int_mv pred_mv;
  MV best_mv;
  INIT x;
  int local_col = get_local_id(0);
  int local_row = get_local_id(1);
  int global_col = get_global_id(0);
  int global_row = get_global_id(1);
  int mi_row = (global_row * PIXEL_ROWS_PER_WORKITEM) / MI_SIZE;
  int mi_col = global_col;
  int bsize;

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

  pred_mv = gpu_input->pred_mv;
  zero_mv.as_int = 0;

  vp9_set_mv_search_range_step2(&x, &zero_mv.as_mv);

  best_mv = get_best_mv(ref_frame, cur_frame, stride, intermediate_int,
                        zero_mv.as_mv, pred_mv.as_mv, pred_mv_sad);

  best_mv = full_pixel_search(ref_frame, cur_frame, stride,
                              intermediate_int, best_mv, zero_mv.as_mv,
                              nmvsadcost_0, nmvsadcost_1,
                              &x, sad_per_bit);

  return best_mv;
}

//=====   KERNELS   =====
//------------------------------
__kernel
__attribute__((reqd_work_group_size(BLOCK_SIZE_IN_PIXELS / NUM_PIXELS_PER_WORKITEM,
                                    BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM,
                                    1)))
void vp9_full_pixel_search(__global uchar *ref_frame,
                           __global uchar *cur_frame,
                           int stride,
                           __global GPU_INPUT *gpu_input,
                           __global GPU_OUTPUT *gpu_output,
                           __global GPU_RD_PARAMETERS *rd_parameters,
                           int mi_rows,
                           int mi_cols) {
  __local int intermediate_int[1];
  int group_col = get_group_id(0);
  int global_row = get_global_id(1);
  int group_stride = get_num_groups(0);
  int local_col = get_local_id(0);
  int local_row = get_local_id(1);
  int global_offset = (global_row * PIXEL_ROWS_PER_WORKITEM * stride) +
                      (group_col * BLOCK_SIZE_IN_PIXELS) +
                      (local_col * NUM_PIXELS_PER_WORKITEM);
  global_offset += (VP9_ENC_BORDER_IN_PIXELS * stride) + VP9_ENC_BORDER_IN_PIXELS;
  int group_offset = (global_row / (BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM) *
      group_stride + group_col);
  MV best_mv;
  int pred_mv_sad;

  gpu_input += group_offset;
  gpu_output += group_offset;

  cur_frame += global_offset;
  ref_frame += global_offset;

  if (!gpu_input->do_compute) {
    goto exit;
  }

  if (gpu_output->this_early_term[GPU_INTER_OFFSET(ZEROMV)]) {
    gpu_output->rv = 1;
    goto exit;
  }

  __global GPU_RD_SEG_PARAMETERS *seg_rd_params =
      &rd_parameters->seg_rd_param[gpu_input->seg_id];

  best_mv = combined_motion_search(ref_frame, cur_frame,
                                   gpu_input, rd_parameters,
                                   seg_rd_params->sad_per_bit,
                                   stride,
                                   mi_rows, mi_cols, &pred_mv_sad,
                                   intermediate_int);

  gpu_output->mv[GPU_INTER_OFFSET(NEWMV)].as_mv = best_mv;
  gpu_output->rv = 0;

  if (local_col == 0 && local_row == 0) {
    int rate_mv = ROUND_POWER_OF_TWO(mv_cost(&best_mv, rd_parameters->nmvjointcost,
            rd_parameters->mvcost[0] + MV_MAX,
            rd_parameters->mvcost[1] + MV_MAX) * MV_COST_WEIGHT, 7);

    int rate_mode = rd_parameters->inter_mode_cost[GPU_INTER_OFFSET(NEWMV)];

    int64_t best_rd_so_far = RDCOST(seg_rd_params->rd_mult,
        rd_parameters->rd_div,
        gpu_output->rate[GPU_INTER_OFFSET(ZEROMV)],
        gpu_output->dist[GPU_INTER_OFFSET(ZEROMV)]);

    gpu_output->rv = RDCOST(seg_rd_params->rd_mult, rd_parameters->rd_div,
        (rate_mv + rate_mode), 0) > best_rd_so_far;
  }
exit:
  return;
}