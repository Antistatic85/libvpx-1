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
__constant ushort2 vp9_bilinear_filters[16] = {
  {128,   0},
  {120,   8},
  {112,  16},
  {104,  24},
  { 96,  32},
  { 88,  40},
  { 80,  48},
  { 72,  56},
  { 64,  64},
  { 56,  72},
  { 48,  80},
  { 40,  88},
  { 32,  96},
  { 24, 104},
  { 16, 112},
  {  8, 120}
};

//=====   FUNCTION MACROS   =====
//--------------------------------------
///* estimated cost of a motion vector (r,c) */
#define MVC(v, r, c)                                             \
     (((nmvjointcost[((r) != refmv.row) * 2 + ((c) != refmv.col)]\
          + nmvcost_0[((r) - refmv.row)]                         \
                 + nmvcost_1[((c) - refmv.col)])                 \
                      * error_per_bit + 4096) >> 13)

// The VP9_BILINEAR_FILTERS_2TAP macro returns a pointer to the bilinear
// filter kernel as a 2 tap filter.
#define BILINEAR_FILTERS_2TAP(x)  (vp9_bilinear_filters[(x)])

//=====   FUNCTION DEFINITIONS   =====
//-------------------------------------------
// convert motion vector component to offset for svf calc
inline int sp(int x) {
  return (x & 7) << 1;
}

void var_filter_block2d_bil_both(__global uchar *ref_data,
                                 __global uchar *cur_data,
                                 int stride,
                                 ushort2 horz_filter,
                                 ushort2 vert_filter,
                                 unsigned int *sse,
                                 int *sum) {
  uchar8 output;
  uchar16 src;
  ushort8 round_factor = 1 << (FILTER_BITS - 1);
  ushort8 filter_shift = FILTER_BITS;
  short8 diff;
  short8 vsum = 0;
  uint4 vsse = 0;
  int row;
  uchar8 tmp_out1, tmp_out2;
  uchar8 cur;

  src = vload16(0, ref_data);
  ref_data += stride;

  tmp_out1 = convert_uchar8((convert_ushort8(src.s01234567) * horz_filter.s0 +
      convert_ushort8(src.s12345678) * horz_filter.s1 + round_factor) >> filter_shift);

  for (row = 0; row < PIXEL_ROWS_PER_WORKITEM; row += 2) {

    // Iteration 1
    src = vload16(0, ref_data);
    ref_data += stride;

    tmp_out2 = convert_uchar8((convert_ushort8(src.s01234567) * horz_filter.s0 +
        convert_ushort8(src.s12345678) * horz_filter.s1 + round_factor) >> filter_shift);

    output = convert_uchar8((convert_ushort8(tmp_out1) * vert_filter.s0 +
        convert_ushort8(tmp_out2) * vert_filter.s1 + round_factor) >> filter_shift);

    cur = vload8(0, cur_data);
    cur_data += stride;

    diff = convert_short8(output) - convert_short8(cur);
    vsum += diff;
    vsse += convert_uint4(convert_int4(diff.s0123) * convert_int4(diff.s0123));
    vsse += convert_uint4(convert_int4(diff.s4567) * convert_int4(diff.s4567));

    // Iteration 2
    src = vload16(0, ref_data);
    ref_data += stride;

    tmp_out1 = convert_uchar8((convert_ushort8(src.s01234567) * horz_filter.s0 +
        convert_ushort8(src.s12345678) * horz_filter.s1 + round_factor) >> filter_shift);

    output = convert_uchar8((convert_ushort8(tmp_out2) * vert_filter.s0 +
        convert_ushort8(tmp_out1) * vert_filter.s1 + round_factor) >> filter_shift);

    cur = vload8(0, cur_data);
    cur_data += stride;

    diff = convert_short8(output) - convert_short8(cur);
    vsum += diff;
    vsse += convert_uint4(convert_int4(diff.s0123) * convert_int4(diff.s0123));
    vsse += convert_uint4(convert_int4(diff.s4567) * convert_int4(diff.s4567));

  }
  vsum.s0123 = vsum.s0123 + vsum.s4567;
  vsum.s01 = vsum.s01 + vsum.s23;
  *sum = vsum.s0 + vsum.s1;

  vsse.s01 = vsse.s01 + vsse.s23;
  *sse = vsse.s0 + vsse.s1;

  return;
}

void var_filter_block2d_bil_horizontal(__global uchar *ref_frame,
                                       ushort2 vp9_filter,
                                       __global uchar *cur_frame,
                                       unsigned int *sse,
                                       int *sum,
                                       int stride) {
  uchar8 output;
  uchar8 src_0;
  uchar8 src_1;
  ushort8 round_factor = 1 << (FILTER_BITS - 1);
  ushort8 filter_shift = FILTER_BITS;
  short8 diff;
  short8 vsum = 0;
  uint4 vsse = 0;
  int row;

  for (row = 0; row < PIXEL_ROWS_PER_WORKITEM; row++) {
    src_0 = vload8(0, ref_frame);
    src_1 = vload8(0, ref_frame + 1);
    output = convert_uchar8((convert_ushort8(src_0) * vp9_filter.s0 +
        convert_ushort8(src_1) * vp9_filter.s1 + round_factor) >> filter_shift);

    uchar8 cur = vload8(0, cur_frame);

    diff = convert_short8(output) - convert_short8(cur);
    vsum += diff;
    vsse += convert_uint4(convert_int4(diff.s0123) * convert_int4(diff.s0123));
    vsse += convert_uint4(convert_int4(diff.s4567) * convert_int4(diff.s4567));

    cur_frame += stride;
    ref_frame += stride;
  }
  vsum.s0123 = vsum.s0123 + vsum.s4567;
  vsum.s01 = vsum.s01 + vsum.s23;
  *sum = vsum.s0 + vsum.s1;

  vsse.s01 = vsse.s01 + vsse.s23;
  *sse = vsse.s0 + vsse.s1;

  return;
}

void var_filter_block2d_bil_vertical(__global uchar *ref_frame,
                                     ushort2 vp9_filter,
                                     __global uchar *cur_frame,
                                     unsigned int *sse,
                                     int *sum,
                                     int stride) {
  uchar8 output;
  uchar8 src_0;
  uchar8 src_1;
  ushort8 round_factor = 1 << (FILTER_BITS - 1);
  ushort8 filter_shift = FILTER_BITS;
  short8 diff;
  short8 vsum = 0;
  uint4 vsse = 0;
  int row;
  int stride_by_8 = stride / 8;

  for (row = 0; row < PIXEL_ROWS_PER_WORKITEM; row++) {
    src_0 = vload8(0, ref_frame);
    src_1 = vload8(stride_by_8, ref_frame);
    output = convert_uchar8((convert_ushort8(src_0) * vp9_filter.s0 +
        convert_ushort8(src_1) * vp9_filter.s1 + round_factor) >> filter_shift);

    uchar8 cur = vload8(0, cur_frame);

    diff = convert_short8(output) - convert_short8(cur);
    vsum += diff;
    vsse += convert_uint4(convert_int4(diff.s0123) * convert_int4(diff.s0123));
    vsse += convert_uint4(convert_int4(diff.s4567) * convert_int4(diff.s4567));

    cur_frame += stride;
    ref_frame += stride;

  }
  vsum.s0123 = vsum.s0123 + vsum.s4567;
  vsum.s01 = vsum.s01 + vsum.s23;
  *sum = vsum.s0 + vsum.s1;

  vsse.s01 = vsse.s01 + vsse.s23;
  *sse = vsse.s0 + vsse.s1;

  return;
}

void calculate_subpel_variance(__global uchar *ref_frame,
                               __global uchar *cur_frame,
                               int stride,
                               int xoffset,
                               int yoffset,
                               int row,
                               int col,
                               unsigned int *sse,
                               int *sum) {
  int buffer_offset;
  __global uchar *tmp_ref,*tmp_cur;

  buffer_offset = ((row >> 3) * stride) + (col >> 3);

  tmp_ref = ref_frame + buffer_offset;
  tmp_cur = cur_frame;

// Enabling this piece of code causes a crash in Intel HD graphics. But it works
// fine in Mali GPU and AMD GPU. Must be an issue with Intel's driver
#if !INTEL_HD_GRAPHICS
  if(!yoffset) {
    var_filter_block2d_bil_horizontal(tmp_ref,
                                      BILINEAR_FILTERS_2TAP(xoffset),
                                      tmp_cur, sse, sum, stride);
  } else if(!xoffset) {
    var_filter_block2d_bil_vertical(tmp_ref,
                                    BILINEAR_FILTERS_2TAP(yoffset),
                                    tmp_cur, sse, sum, stride);

  } else
#endif
  {
    var_filter_block2d_bil_both(tmp_ref, tmp_cur, stride,
                                BILINEAR_FILTERS_2TAP(xoffset),
                                BILINEAR_FILTERS_2TAP(yoffset),
                                sse, sum);
  }
}

int mv_err_cost(MV *mv, MV *ref,
                __global int *nmvcost_0,
                __global int *nmvcost_1,
                __global int *nmvjointcost,
                int error_per_bit) {
  MV diff;

  diff.row = mv->row - ref->row;
  diff.col = mv->col - ref->col;

  return ROUND_POWER_OF_TWO(mv_cost(&diff, nmvjointcost, nmvcost_0,
                                    nmvcost_1) * error_per_bit, 13);
}

MV check_better_subpel(__global uchar *ref_frame,
                       __global uchar *cur_frame,
                       __global int *nmvcost_0,
                       __global int *nmvcost_1,
                       __global int *nmvjointcost,
                       int stride,
                       unsigned int *v,
                       int r,
                       int c,
                       MV best_mv,
                       MV refmv,
                       MV minmv,
                       MV maxmv,
                       unsigned int *pbesterr,
                       int error_per_bit,
                       __local int *intermediate_int) {

  int sum, thismse;
  unsigned int sse;
  int distortion;

  if (c >= minmv.col && c <= maxmv.col && r >= minmv.row && r <= maxmv.row) {
    calculate_subpel_variance(ref_frame, cur_frame, stride,
                              sp(c), sp(r), r, c, &sse, &sum);

    barrier(CLK_LOCAL_MEM_FENCE);
    intermediate_int[0] = 0;
    intermediate_int[1] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    atomic_add(intermediate_int, sum);
    atomic_add(intermediate_int + 1, sse);

    barrier(CLK_LOCAL_MEM_FENCE);
    sum = intermediate_int[0];
    sse = intermediate_int[1];

    thismse = sse - (((long int)sum * sum)
        / (BLOCK_SIZE_IN_PIXELS * BLOCK_SIZE_IN_PIXELS));

    if ((*v = MVC(*v, r, c) + thismse) < *pbesterr) {
      *pbesterr = *v;
      best_mv.row = r;
      best_mv.col = c;
      distortion = thismse;
    }
  } else {
    *v = CL_INT_MAX;
  }

  return best_mv;
}

MV first_level_checks(__global uchar *ref_frame,
                      __global uchar *cur_frame,
                      __global int *nmvcost_0,
                      __global int *nmvcost_1,
                      __global int *nmvjointcost,
                      int stride,
                      int hstep,
                      MV best_mv,
                      MV refmv,
                      MV minmv,
                      MV maxmv,
                      unsigned int *pbesterr,
                      int error_per_bit,
                      __local int *intermediate_int) {
  unsigned int left, right, up, down, diag, whichdir;
  int distortion;
  int sum, thismse, tr, tc;
  unsigned int besterr, sse;

  tr = best_mv.row;
  tc = best_mv.col;

  besterr = *pbesterr;

  best_mv = check_better_subpel(ref_frame, cur_frame,
                                nmvcost_0, nmvcost_1, nmvjointcost, stride,
                                &left, tr, (tc - hstep),
                                best_mv, refmv, minmv, maxmv,
                                &besterr, error_per_bit,
                                intermediate_int);

  best_mv = check_better_subpel(ref_frame, cur_frame,
                                nmvcost_0, nmvcost_1, nmvjointcost, stride,
                                &right, tr, (tc + hstep),
                                best_mv, refmv, minmv, maxmv,
                                &besterr, error_per_bit,
                                intermediate_int);

  best_mv = check_better_subpel(ref_frame, cur_frame,
                                nmvcost_0, nmvcost_1, nmvjointcost, stride,
                                &up, (tr - hstep), tc,
                                best_mv, refmv, minmv, maxmv,
                                &besterr, error_per_bit,
                                intermediate_int);

  best_mv = check_better_subpel(ref_frame, cur_frame,
                                nmvcost_0, nmvcost_1, nmvjointcost, stride,
                                &down, (tr + hstep), tc,
                                best_mv, refmv, minmv, maxmv,
                                &besterr, error_per_bit,
                                intermediate_int);

  whichdir = (left <= right ? 0 : 1) + (up <= down ? 0 : 2);

  switch (whichdir) {
    case 0:
      tr = tr - hstep;
      tc = tc - hstep;
      break;
    case 1:
      tr = tr - hstep;
      tc = tc + hstep;
      break;
    case 2:
      tr = tr + hstep;
      tc = tc - hstep;
      break;
    case 3:
      tr = tr + hstep;
      tc = tc + hstep;
      break;
  }

  best_mv = check_better_subpel(ref_frame, cur_frame,
                                nmvcost_0, nmvcost_1, nmvjointcost, stride,
                                &diag, tr, tc,
                                best_mv, refmv, minmv, maxmv,
                                &besterr, error_per_bit,
                                intermediate_int);

  *pbesterr = besterr;

  return best_mv;
}

void calculate_fullpel_variance(__global uchar *ref_frame,
                                __global uchar *cur_frame,
                                int stride,
                                unsigned int *sse,
                                int *sum,
                                MV *submv) {
  uchar8 ref,cur;
  int buffer_offset;
  __global uchar *tmp_ref,*tmp_cur;
  short8 diff;
  short8 vsum = 0;
  uint4 vsse = 0;
  int row;

  buffer_offset = ((submv->row >> 3) * stride) + (submv->col >> 3);
  *sum = 0;
  *sse = 0;

  tmp_ref = ref_frame + buffer_offset;
  tmp_cur = cur_frame;

  for(row = 0; row < PIXEL_ROWS_PER_WORKITEM; row++) {
    ref = vload8(0,tmp_ref);
    cur = vload8(0,tmp_cur);

    diff = convert_short8(ref) - convert_short8(cur);
    vsum += diff;
    vsse += convert_uint4(convert_int4(diff.s0123) * convert_int4(diff.s0123));
    vsse += convert_uint4(convert_int4(diff.s4567) * convert_int4(diff.s4567));

    tmp_ref += stride;
    tmp_cur += stride;
  }
  vsum.s0123 = vsum.s0123 + vsum.s4567;
  vsum.s01 = vsum.s01 + vsum.s23;
  *sum = vsum.s0 + vsum.s1;

  vsse.s01 = vsse.s01 + vsse.s23;
  *sse = vsse.s0 + vsse.s1;
}

MV vp9_find_best_sub_pixel_tree(__global uchar *ref_frame,
                                __global uchar *cur_frame,
                                __global int *nmvcost_0,
                                __global int *nmvcost_1,
                                __global int *nmvjointcost,
                                int stride,
                                MV best_mv,
                                MV nearest_mv,
                                MV fcenter_mv,
                                INIT *x,
                                int error_per_bit,
                                __local int *intermediate_int) {
  int sum, thismse;
  int hstep;
  unsigned int sse, besterr;
  MV minmv, maxmv;

  hstep = 4;
  besterr = CL_INT_MAX;

  calculate_fullpel_variance(ref_frame, cur_frame, stride,
                             &sse, &sum, &best_mv);

  barrier(CLK_LOCAL_MEM_FENCE);
  intermediate_int[0] = 0;
  intermediate_int[1] = 0;

  barrier(CLK_LOCAL_MEM_FENCE);
  atomic_add(intermediate_int, sum);
  atomic_add(intermediate_int + 1, sse);
  barrier(CLK_LOCAL_MEM_FENCE);
  sum = intermediate_int[0];
  sse = intermediate_int[1];

  besterr = sse - (((long int)sum * sum)
      / (BLOCK_SIZE_IN_PIXELS * BLOCK_SIZE_IN_PIXELS));

  besterr += mv_err_cost(&best_mv, &nearest_mv,
                         nmvcost_0, nmvcost_1, nmvjointcost,
                         error_per_bit);

  minmv.col = MAX(x->mv_col_min * 8, fcenter_mv.col - MV_MAX);
  maxmv.col = MIN(x->mv_col_max * 8, fcenter_mv.col + MV_MAX);
  minmv.row = MAX(x->mv_row_min * 8, fcenter_mv.row - MV_MAX);
  maxmv.row = MIN(x->mv_row_max * 8, fcenter_mv.row + MV_MAX);

  best_mv = first_level_checks(ref_frame, cur_frame,
                               nmvcost_0, nmvcost_1, nmvjointcost,
                               stride, hstep,
                               best_mv, nearest_mv, minmv, maxmv,
                               &besterr, error_per_bit,
                               intermediate_int);

  hstep >>= 1;
  best_mv = first_level_checks(ref_frame, cur_frame,
                               nmvcost_0, nmvcost_1, nmvjointcost,
                               stride, hstep,
                               best_mv, nearest_mv, minmv, maxmv,
                               &besterr, error_per_bit,
                               intermediate_int);

  return best_mv;
}

__kernel
__attribute__((reqd_work_group_size(BLOCK_SIZE_IN_PIXELS / NUM_PIXELS_PER_WORKITEM,
                                    BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM,
                                    1)))
void vp9_sub_pixel_search(__global uchar *ref_frame,
                          __global uchar *cur_frame,
                          int stride,
                          __global GPU_INPUT *gpu_input,
                          __global GPU_OUTPUT *gpu_output,
                          __global GPU_RD_PARAMETERS *rd_parameters,
                          int mi_rows,
                          int mi_cols) {
  __local int intermediate_int[2];
  int global_col = get_global_id(0);
  int global_row = get_global_id(1);
  int global_stride = get_global_size(0);
  int group_col = get_group_id(0);
  int group_row = get_group_id(1);
  int group_stride = get_num_groups(0);
  int local_col  = get_local_id(0);
  int local_row  = get_local_id(1);
  int global_offset = (global_row * PIXEL_ROWS_PER_WORKITEM * stride) +
                      (global_col * NUM_PIXELS_PER_WORKITEM);

  global_offset += (VP9_ENC_BORDER_IN_PIXELS * stride) + VP9_ENC_BORDER_IN_PIXELS;

  int group_offset = (global_row / (BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM) *
        group_stride + group_col);

  gpu_input += group_offset;
  gpu_output += group_offset;

  cur_frame += global_offset;
  ref_frame += global_offset;

  if (!gpu_input->do_compute) {
    goto exit;
  }

  if (gpu_output->rv) {
    goto exit;
  }

  int mi_row = (global_row * PIXEL_ROWS_PER_WORKITEM) / MI_SIZE;
  int mi_col = global_col;
  BLOCK_SIZE bsize;

#if BLOCK_SIZE_IN_PIXELS == 64
  mi_row = (mi_row >> 3) << 3;
  mi_col = (mi_col >> 3) << 3;
  bsize = BLOCK_64X64;
#elif BLOCK_SIZE_IN_PIXELS == 32
  mi_row = (mi_row >> 2) << 2;
  mi_col = (mi_col >> 2) << 2;
  bsize = BLOCK_32X32;
#endif

  MV best_mv = gpu_output->mv[GPU_INTER_OFFSET(NEWMV)].as_mv;
  MV nearest_mv = {0, 0};
  MV fcenter_mv;
  INIT x;
  __global int *nmvcost_0 = rd_parameters->mvcost[0] + MV_MAX;
  __global int *nmvcost_1 = rd_parameters->mvcost[1] + MV_MAX;
  __global int *nmvjointcost = rd_parameters->nmvjointcost;
  int error_per_bit = rd_parameters->error_per_bit;

  fcenter_mv.row = nearest_mv.row >> 3;
  fcenter_mv.col = nearest_mv.col >> 3;

  vp9_gpu_set_mv_search_range(&x, mi_row, mi_col, mi_rows, mi_cols, bsize);

  gpu_output->mv[GPU_INTER_OFFSET(NEWMV)].as_mv  =
      vp9_find_best_sub_pixel_tree(ref_frame, cur_frame,
                                   nmvcost_0, nmvcost_1, nmvjointcost,
                                   stride,
                                   best_mv, nearest_mv, fcenter_mv,
                                   &x, error_per_bit,
                                   intermediate_int);
exit:
  return;
}