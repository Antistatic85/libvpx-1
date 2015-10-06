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

#define CR_SEGMENT_ID_BASE    0

typedef struct {
  short sum8x8[64];
}SUM8X8;

struct GPU_OUTPUT_PRO_ME {
  SUM8X8 sum8x8;
  int_mv pred_mv;
  int pred_mv_sad;
  char color_sensitivity;
} __attribute__ ((aligned(32)));
typedef struct GPU_OUTPUT_PRO_ME GPU_OUTPUT_PRO_ME;

//=====   GLOBAL DEFINITIONS   =====
//--------------------------------------
__constant MV search_pos[4] = {
    {-1, 0}, {0, -1}, {0, 1}, {1, 0},
};

//=====   FUNCTION DEFINITIONS   =====
//-------------------------------------------
int vp9_minmax_8x8(__global uchar *s, __global uchar *d, int stride,
                   uchar *max, uchar *min) {
  int i, j;
  uchar8 src, pred;
  uchar8 diff;

  *min = 255;
  *max = 0;
  for (i = 0; i < 8; ++i, s += stride, d += stride) {
    src = vload8(0, s);
    pred = vload8(0, d);
    diff = abs_diff(src, pred);

    *max = MAX(*max, diff.s0);
    *min = MIN(*min, diff.s0);
    *max = MAX(*max, diff.s1);
    *min = MIN(*min, diff.s1);
    *max = MAX(*max, diff.s2);
    *min = MIN(*min, diff.s2);
    *max = MAX(*max, diff.s3);
    *min = MIN(*min, diff.s3);
    *max = MAX(*max, diff.s4);
    *min = MIN(*min, diff.s4);
    *max = MAX(*max, diff.s5);
    *min = MIN(*min, diff.s5);
    *max = MAX(*max, diff.s6);
    *min = MIN(*min, diff.s6);
    *max = MAX(*max, diff.s7);
    *min = MIN(*min, diff.s7);
  }
}

// Compute the minmax over the 8x8 subblocks.
int compute_minmax_8x8(__global uchar *src, __global uchar *ref,
                       int stride) {
  int k;
  int minmax_max = 0;
  int minmax_min = 255;
  // Loop over the 4 8x8 subblocks.
  for (k = 0; k < 4; k++) {
    int x8_idx = ((k & 1) << 3);
    int y8_idx = ((k >> 1) << 3);
    uchar min;
    uchar max;

    vp9_minmax_8x8(src + y8_idx * stride + x8_idx,
                   ref + y8_idx * stride + x8_idx, stride,
                   &max, &min);
    if ((max - min) > minmax_max)
      minmax_max = (max - min);
    if ((max - min) < minmax_min)
      minmax_min = (max - min);
  }
  return (minmax_max - minmax_min);
}

int get_variance(uint32_t sum_square_error, int sum_error, int log2_count) {
  return ((int)(256 * (sum_square_error -
      ((sum_error * sum_error) >> log2_count)) >> log2_count));
}

uint32_t vp9_avg_8x8(__global uchar *buffer, int stride) {
  int index_y;
  uchar8 cur;
  ushort8 sum = 0;
  ushort4 final_sum;
  for (index_y = 0; index_y < 8; index_y++) {
    cur = vload8(0, buffer);
    sum += convert_ushort8(cur);
    buffer += stride;
  }
  final_sum = convert_ushort4(sum.s0123) + convert_ushort4(sum.s4567);
  final_sum.s01 = final_sum.s01 + final_sum.s23;
  return (final_sum.s0 + final_sum.s1 + 32) >> 6;
}

void var_filter_block2d_bil_both(__global uchar *ref_data,
                                 __global uchar *cur_data,
                                 int stride,
                                 ushort2 horz_filter,
                                 ushort2 vert_filter,
                                 int *sum) {
  uchar8 output;
  uchar16 src;
  ushort8 round_factor = 1 << (FILTER_BITS - 1);
  ushort8 filter_shift = FILTER_BITS;
  ushort8 diff;
  ushort8 vsum = 0;
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

    diff = abs_diff(convert_short8(output), convert_short8(cur));
    vsum += diff;

    // Iteration 2
    src = vload16(0, ref_data);
    ref_data += stride;

    tmp_out1 = convert_uchar8((convert_ushort8(src.s01234567) * horz_filter.s0 +
        convert_ushort8(src.s12345678) * horz_filter.s1 + round_factor) >> filter_shift);

    output = convert_uchar8((convert_ushort8(tmp_out2) * vert_filter.s0 +
        convert_ushort8(tmp_out1) * vert_filter.s1 + round_factor) >> filter_shift);

    cur = vload8(0, cur_data);
    cur_data += stride;

    diff = abs_diff(convert_short8(output), convert_short8(cur));
    vsum += diff;

  }
  vsum.s0123 = vsum.s0123 + vsum.s4567;
  vsum.s01 = vsum.s01 + vsum.s23;
  *sum = vsum.s0 + vsum.s1;

  return;
}

int get_uv_filtered_sad(__global uchar *ref_frame,
                        __global uchar *cur_frame,
                        int stride,
                        __local int* atomic_sad,
                        MV this_mv) {
  int local_col = get_local_id(0);
  int local_row = get_local_id(1);
  int intermediate_sad;

  ref_frame += ((this_mv.row >> 4) * stride + (this_mv.col >> 4));

  barrier(CLK_LOCAL_MEM_FENCE);
  atomic_sad[0] = 0;

  var_filter_block2d_bil_both(ref_frame, cur_frame, stride,
                              BILINEAR_FILTERS_2TAP(sp(this_mv.col >> 1)),
                              BILINEAR_FILTERS_2TAP(sp(this_mv.row >> 1)),
                              &intermediate_sad);

  barrier(CLK_LOCAL_MEM_FENCE);
  atomic_add(atomic_sad, intermediate_sad);

  barrier(CLK_LOCAL_MEM_FENCE);
  return atomic_sad[0];
}

void calculate_sad_4d(MV *currentmv,
                      __global uchar *ref_frame,
                      __global uchar *cur_frame,
                      int stride,
                      int *sad_array) {
  __global uchar *tmp_cur;
  __global uchar *tmp_ref_l, *tmp_ref_r, *tmp_ref_t, *tmp_ref_b;
  uchar8 ref, cur;
  ushort8 sad[4] = {0};
  ushort4 final_sad[4];
  int row;

  tmp_ref_l = ref_frame + (currentmv->row * stride) + currentmv->col - 1;
  tmp_ref_r = ref_frame + (currentmv->row * stride) + currentmv->col + 1;
  tmp_ref_t = ref_frame + ((currentmv->row - 1) * stride) + currentmv->col;
  tmp_ref_b = ref_frame + ((currentmv->row + 1) * stride) + currentmv->col;
  tmp_cur = cur_frame;

  for (row = 0; row < PIXEL_ROWS_PER_WORKITEM; row++) {
    cur = vload8(0, tmp_cur);

    ref = vload8(0, tmp_ref_l);
    sad[0] += abs_diff(convert_ushort8(ref), convert_ushort8(cur));

    ref = vload8(0, tmp_ref_r);
    sad[1] += abs_diff(convert_ushort8(ref), convert_ushort8(cur));

    ref = vload8(0, tmp_ref_t);
    sad[2] += abs_diff(convert_ushort8(ref), convert_ushort8(cur));

    ref = vload8(0, tmp_ref_b);
    sad[3] += abs_diff(convert_ushort8(ref), convert_ushort8(cur));

    tmp_ref_l += stride;
    tmp_ref_r += stride;
    tmp_ref_t += stride;
    tmp_ref_b += stride;
    tmp_cur += stride;
  }

  final_sad[0] = convert_ushort4(sad[0].s0123) + convert_ushort4(sad[0].s4567);
  final_sad[1] = convert_ushort4(sad[1].s0123) + convert_ushort4(sad[1].s4567);
  final_sad[2] = convert_ushort4(sad[2].s0123) + convert_ushort4(sad[2].s4567);
  final_sad[3] = convert_ushort4(sad[3].s0123) + convert_ushort4(sad[3].s4567);

  final_sad[0].s01 = final_sad[0].s01 + final_sad[0].s23;
  final_sad[1].s01 = final_sad[1].s01 + final_sad[1].s23;
  final_sad[2].s01 = final_sad[2].s01 + final_sad[2].s23;
  final_sad[3].s01 = final_sad[3].s01 + final_sad[3].s23;

  sad_array[0] = final_sad[2].s0 + final_sad[2].s1;
  sad_array[1] = final_sad[0].s0 + final_sad[0].s1;
  sad_array[2] = final_sad[1].s0 + final_sad[1].s1;
  sad_array[3] = final_sad[3].s0 + final_sad[3].s1;
}

void get_sad_4d(__global uchar *ref_frame,
                __global uchar *cur_frame,
                int stride,
                int* sad_array,
                MV this_mv,
                __local int* atomic_sad) {
  int intermediate_sad[4];
  int local_col = get_local_id(0);
  int local_row = get_local_id(1);

  barrier(CLK_LOCAL_MEM_FENCE);

  atomic_sad[0] = 0;
  atomic_sad[1] = 0;
  atomic_sad[2] = 0;
  atomic_sad[3] = 0;

  calculate_sad_4d(&this_mv, ref_frame, cur_frame, stride, intermediate_sad);

  barrier(CLK_LOCAL_MEM_FENCE);

  atomic_add(&atomic_sad[0], intermediate_sad[0]);
  atomic_add(&atomic_sad[1], intermediate_sad[1]);
  atomic_add(&atomic_sad[2], intermediate_sad[2]);
  atomic_add(&atomic_sad[3], intermediate_sad[3]);

  barrier(CLK_LOCAL_MEM_FENCE);

  sad_array[0] = atomic_sad[0];
  sad_array[1] = atomic_sad[1];
  sad_array[2] = atomic_sad[2];
  sad_array[3] = atomic_sad[3];
}

void calculate_vector_var(__global ushort *ref_vector,
                          __global ushort *src_vector,
                          int *sum,
                          int *sse) {
  ushort8 ref, src;
  short8 diff;
  short8 vsum = 0;
  uint4 vsse = 0;

  // load vectors
  ref = vload8(0, ref_vector);
  src = vload8(0, src_vector);

  // vector diff
  vsum = convert_short8(ref) - convert_short8(src);

  // vector sse
  vsse += convert_uint4(convert_int4(vsum.s0123) * convert_int4(vsum.s0123));
  vsse += convert_uint4(convert_int4(vsum.s4567) * convert_int4(vsum.s4567));

  // sum
  vsum.s0123 = vsum.s0123 + vsum.s4567;
  vsum.s01 = vsum.s01 + vsum.s23;
  *sum = vsum.s0 + vsum.s1;

  // sse
  vsse.s01 = vsse.s01 + vsse.s23;
  *sse = vsse.s0 + vsse.s1;
}

int get_vector_var(__global ushort *ref_vector,
                   __global ushort *src_vector,
                   int bwl,
                   __local int* intermediate_sad) {
  int sum, sse;

  barrier(CLK_LOCAL_MEM_FENCE);

  intermediate_sad[0] = 0;
  intermediate_sad[1] = 0;

  calculate_vector_var(ref_vector, src_vector, &sum, &sse);

  barrier(CLK_LOCAL_MEM_FENCE);

  atomic_add(&intermediate_sad[0], sum);
  atomic_add(&intermediate_sad[1], sse);

  barrier(CLK_LOCAL_MEM_FENCE);

  return intermediate_sad[1] - ((intermediate_sad[0] * intermediate_sad[0]) >> (bwl + 2));
}

int vector_match(__global ushort *proj_ref,
                 __global ushort *proj_src,
                 __local int* intermediate_int) {
  int best_sad = INT_MAX;
  int this_sad;
  int d;
  int center, offset = 0;
  int bw = 64;
  int bwl = 4;

  for (d = 0; d <= bw; d += 16) {
    this_sad = get_vector_var(proj_ref + d, proj_src, bwl, intermediate_int);
    if (this_sad < best_sad) {
      best_sad = this_sad;
      offset = d;
    }
  }
  center = offset;

  for (d = -8; d <= 8; d += 16) {
    int this_pos = offset + d;
    // check limit
    if (this_pos < 0 || this_pos > bw)
      continue;
    this_sad = get_vector_var(proj_ref + this_pos, proj_src, bwl,
                              intermediate_int);
    if (this_sad < best_sad) {
      best_sad = this_sad;
      center = this_pos;
    }
  }
  offset = center;

  for (d = -4; d <= 4; d += 8) {
    int this_pos = offset + d;
    // check limit
    if (this_pos < 0 || this_pos > bw)
      continue;
    this_sad = get_vector_var(proj_ref + this_pos, proj_src, bwl,
                              intermediate_int);
    if (this_sad < best_sad) {
      best_sad = this_sad;
      center = this_pos;
    }
  }
  offset = center;

  for (d = -2; d <= 2; d += 4) {
    int this_pos = offset + d;
    // check limit
    if (this_pos < 0 || this_pos > bw)
      continue;
    this_sad = get_vector_var(proj_ref + this_pos, proj_src, bwl,
                              intermediate_int);
    if (this_sad < best_sad) {
      best_sad = this_sad;
      center = this_pos;
    }
  }
  offset = center;

  for (d = -1; d <= 1; d += 2) {
    int this_pos = offset + d;
    // check limit
    if (this_pos < 0 || this_pos > bw)
      continue;
    this_sad = get_vector_var(proj_ref + this_pos, proj_src, bwl,
                              intermediate_int);
    if (this_sad < best_sad) {
      best_sad = this_sad;
      center = this_pos;
    }
  }

  return (center - (bw >> 1));
}

ushort8 row_project(__global uchar *buff, int stride, int height) {
  int idx;
  uchar8 ref;
  ushort8 sum = 0;

  for (idx = 0; idx < height; idx += 1) {
    ref = vload8(0, buff);
    sum += convert_ushort8(ref);
    buff += stride;
  }
  return sum;
}

ushort column_project(__global uchar *buff, int width) {
  int idx;
  uchar8 ref;
  ushort8 sum = 0;

  for (idx = 0; idx < width; idx += 8) {
    ref = vload8(0, buff);
    sum += convert_ushort8(ref);
    buff += 8;
  }
  ushort4 final_sad = convert_ushort4(sum.s0123) + convert_ushort4(sum.s4567);
  final_sad.s01 = final_sad.s01 + final_sad.s23;

  return ((final_sad.s0 + final_sad.s1) >> (short)5);
}

//=====   KERNELS   =====
//------------------------------
__kernel
void vp9_col_row_projection(__global uchar *src_frame,
                        __global uchar *ref_frame,
                        int in_stride,
                        __global ushort *src_proj_r,
                        __global ushort *ref_proj_r,
                        __global ushort *src_proj_c,
                        __global ushort *ref_proj_c) {
  int global_row = get_global_id(1);
  int local_col = get_local_id(0);
  int group_col = get_group_id(0);
  int global_offset = (global_row * in_stride * BLOCK_SIZE_IN_PIXELS) +
      (local_col * in_stride) + (group_col * BLOCK_SIZE_IN_PIXELS);
  int out_stride = get_num_groups(1) * BLOCK_SIZE_IN_PIXELS;
  __local ushort8 intermediate_sum[8 * 4 * 2];
  __global uchar *src, *ref;

  // Column projection
  global_offset += (VP9_ENC_BORDER_IN_PIXELS * in_stride) + VP9_ENC_BORDER_IN_PIXELS;

  src = src_frame + global_offset;
  src_proj_c[group_col * out_stride + (global_row * BLOCK_SIZE_IN_PIXELS) + local_col] =
      column_project(src, BLOCK_SIZE_IN_PIXELS);

  ref = ref_frame + global_offset - (32 * in_stride);
  ref_proj_c[group_col * out_stride + (global_row * BLOCK_SIZE_IN_PIXELS) + local_col] =
      column_project(ref, BLOCK_SIZE_IN_PIXELS);

  // Row projection
  if (local_col >= 32)
    goto barrier_sync;

  int col_offset = (group_col * BLOCK_SIZE_IN_PIXELS) +
      ((local_col / 4) * NUM_PIXELS_PER_WORKITEM);
  int row_offset = ((global_row * BLOCK_SIZE_IN_PIXELS) +
      ((local_col & 3) * (BLOCK_SIZE_IN_PIXELS / 4))) * in_stride;
  global_offset = row_offset + col_offset;
  global_offset += VP9_ENC_BORDER_IN_PIXELS * in_stride + VP9_ENC_BORDER_IN_PIXELS;
  out_stride = get_num_groups(0) * BLOCK_SIZE_IN_PIXELS;

  src = src_frame + global_offset;
  intermediate_sum[2 * local_col] =
      row_project(src, in_stride, BLOCK_SIZE_IN_PIXELS / 4);

  ref = ref_frame + global_offset - 32;
  intermediate_sum[2 * local_col + 1] =
      row_project(ref, in_stride, BLOCK_SIZE_IN_PIXELS / 4);

barrier_sync:
  barrier(CLK_LOCAL_MEM_FENCE);

  if (((local_col & 3) == 0) && local_col < 32) {
    int local_offset = 2 * local_col;
    ushort8 sum = intermediate_sum[local_offset] + intermediate_sum[local_offset + 2] +
        intermediate_sum[local_offset + 4] + intermediate_sum[local_offset + 6];
    sum >>= (ushort)5;
    vstore8(sum, 0, &src_proj_r[global_row * out_stride + col_offset]);

    local_offset += 1;
    sum = intermediate_sum[local_offset] + intermediate_sum[local_offset + 2] +
        intermediate_sum[local_offset + 4] + intermediate_sum[local_offset + 6];
    sum >>= (ushort)5;
    vstore8(sum, 0, &ref_proj_r[global_row * out_stride + col_offset]);
  }
}

__kernel
void vp9_vector_match(__global ushort *proj_src_h,
                      __global ushort *proj_ref_h,
                      __global ushort *proj_src_v,
                      __global ushort *proj_ref_v,
                      __global GPU_OUTPUT_PRO_ME *gpu_output_pro_me) {
  __local int intermediate_int[2];
  int group_col = get_group_id(0);
  int group_stride = get_num_groups(0);
  int global_row = get_global_id(1);
  int local_col = get_local_id(0);
  int local_row = get_local_id(1);
  MV thismv;

  gpu_output_pro_me += global_row * group_stride + group_col;

  {
    int stride_h = (get_num_groups(0) + 1) * BLOCK_SIZE_IN_PIXELS;
    int offset_h = (global_row * stride_h) + (group_col * BLOCK_SIZE_IN_PIXELS) +
        (local_col * NUM_PIXELS_PER_WORKITEM);

    proj_src_h += offset_h;
    proj_ref_h += offset_h;

    thismv.col = vector_match(proj_ref_h, proj_src_h, intermediate_int);
  }

  {
    int stride_v = (get_num_groups(1) + 1) * BLOCK_SIZE_IN_PIXELS;
    int offset_v = (group_col * stride_v) + (global_row * BLOCK_SIZE_IN_PIXELS) +
        (local_col * NUM_PIXELS_PER_WORKITEM);

    proj_src_v += offset_v;
    proj_ref_v += offset_v;

    thismv.row = vector_match(proj_ref_v, proj_src_v, intermediate_int);
  }
  gpu_output_pro_me->pred_mv.as_mv = thismv;
}

__kernel
void vp9_pro_motion_estimation(__global uchar *src,
                               __global uchar *ref,
                               int stride,
                               __global GPU_OUTPUT_PRO_ME *gpu_output_pro_me) {
  __local int intermediate_int[4];
  int group_col = get_group_id(0);
  int group_row = get_group_id(1);
  int group_stride = get_num_groups(0);
  int global_row = get_global_id(1);
  int local_col = get_local_id(0);
  int local_row = get_local_id(1);
  MV thismv, bestmv;
  int i;
  int thissad[5], bestsad = INT_MAX;
  MV zeromv = 0;
  int bestsad_g = INT_MAX;
  int global_offset = (global_row * PIXEL_ROWS_PER_WORKITEM * stride) +
                      (group_col * BLOCK_SIZE_IN_PIXELS) +
                      (local_col * NUM_PIXELS_PER_WORKITEM);
  global_offset += (VP9_ENC_BORDER_IN_PIXELS * stride) + VP9_ENC_BORDER_IN_PIXELS;

  src += global_offset;
  ref += global_offset;

  gpu_output_pro_me += (global_row / (BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM)) *
      group_stride + group_col;

  // best sad at init mv
  bestmv = thismv = gpu_output_pro_me->pred_mv.as_mv;
  bestsad = get_sad(ref, src, stride, intermediate_int, thismv);

  // diamond search at 1 pixel distance around init
  get_sad_4d(ref, src, stride, thissad, thismv, intermediate_int);
  for (i = 0; i < 4; i++) {
    if (thissad[i] < bestsad) {
      bestsad = thissad[i];
      bestmv = gpu_output_pro_me->pred_mv.as_mv + search_pos[i];
    }
  }

  // best sad at diagonal
  if (thissad[0] < thissad[3]) {
    thismv.row = gpu_output_pro_me->pred_mv.as_mv.row - 1;
  } else {
    thismv.row = gpu_output_pro_me->pred_mv.as_mv.row + 1;
  }
  if (thissad[1] < thissad[2]) {
    thismv.col = gpu_output_pro_me->pred_mv.as_mv.col - 1;
  } else {
    thismv.col = gpu_output_pro_me->pred_mv.as_mv.col + 1;
  }
  thissad[4] = get_sad(ref, src, stride, intermediate_int, thismv);
  if (thissad[4] < bestsad) {
    bestsad = thissad[4];
    bestmv = thismv;
  }

  thismv = 0;

  thissad[4] = get_sad(ref, src, stride, intermediate_int, thismv);
  if (thissad[4] < bestsad) {
    bestsad = thissad[4];
    bestmv = thismv;
  }

  // best mv & best sad
  gpu_output_pro_me->pred_mv.as_mv.col = bestmv.col << 3;
  gpu_output_pro_me->pred_mv.as_mv.row = bestmv.row << 3;
  gpu_output_pro_me->pred_mv_sad = bestsad;
}

__kernel
void vp9_color_sensitivity(__global uchar *src,
                           __global uchar *ref,
                           int stride,
                           __global GPU_OUTPUT_PRO_ME *gpu_output_pro_me,
                           int64_t yplane_size,
                           int64_t uvplane_size) {
  __local int intermediate_int[1];
  int group_col = get_group_id(0);
  int group_row = get_group_id(1);
  int group_stride = get_num_groups(0);
  int global_row = get_global_id(1);
  int local_col = get_local_id(0);
  int local_row = get_local_id(1);
  MV thismv;
  int uv_sad, y_sad;
  uint8_t color_sensitivity = 0;
  int i;
  int global_offset = (global_row * PIXEL_ROWS_PER_WORKITEM * (stride >> 1)) +
                      (group_col * (BLOCK_SIZE_IN_PIXELS >> 1)) +
                      (local_col * NUM_PIXELS_PER_WORKITEM);
  global_offset += ((VP9_ENC_BORDER_IN_PIXELS >> 1) * (stride >> 1)) +
      (VP9_ENC_BORDER_IN_PIXELS >> 1);

  gpu_output_pro_me += (global_row / ((BLOCK_SIZE_IN_PIXELS >> 1) / PIXEL_ROWS_PER_WORKITEM)) *
      group_stride + group_col;

  thismv = gpu_output_pro_me->pred_mv.as_mv;
  y_sad = gpu_output_pro_me->pred_mv_sad;

  if (((thismv.col & 15) == 0) && ((thismv.row & 15) == 0)) {
    thismv.col = thismv.col >> 4;
    thismv.row = thismv.row >> 4;

    for (i = 0; i < 2; i++) {
      __global uchar *ref_uv = ref + yplane_size + i * uvplane_size;
      __global uchar *src_uv = src + yplane_size + i * uvplane_size;

      src_uv += global_offset;
      ref_uv += global_offset;

      uv_sad = get_sad(ref_uv, src_uv, stride >> 1, intermediate_int, thismv);

      color_sensitivity |= ((uv_sad > (y_sad >> 2)) << i);
    }
    gpu_output_pro_me->color_sensitivity = color_sensitivity;
  } else {

    for (i = 0; i < 2; i++) {
      __global uchar *ref_uv = ref + yplane_size + i * uvplane_size;
      __global uchar *src_uv = src + yplane_size + i * uvplane_size;

      src_uv += global_offset;
      ref_uv += global_offset;

      uv_sad = get_uv_filtered_sad(ref_uv, src_uv, stride >> 1,
                                   intermediate_int, thismv);

      color_sensitivity |= ((uv_sad > (y_sad >> 2)) << i);
    }
    gpu_output_pro_me->color_sensitivity = color_sensitivity;
  }
}

__kernel
void vp9_choose_partitions(__global uchar *src,
                           __global uchar *ref,
                           int stride,
                           __global GPU_OUTPUT_PRO_ME *gpu_output_pro_me,
                           __global GPU_RD_PARAMETERS *rd_parameters,
                           __global GPU_INPUT *gpu_input,
                           int gpu_input_stride) {
  __local int sum[64 + 16 + 4 + 1];
  __local uint32_t sse[64 + 16 + 4 + 1];
  __local int *sum_array[4];
  __local uint32_t *sse_array[4];
  __local int force_split[21];
  uint32_t s_avg, d_avg;
  int variance;
  int segment_id;
  int group_col = get_group_id(0);
  int group_row = get_group_id(1);
  int group_stride = get_num_groups(0);
  int global_row = get_global_id(1);
  int local_col = get_local_id(0);
  int local_row = get_local_id(1);
  int global_offset = (global_row * 8 * stride) +
      (group_col * BLOCK_SIZE_IN_PIXELS) + (local_col * NUM_PIXELS_PER_WORKITEM);
  global_offset += (VP9_ENC_BORDER_IN_PIXELS * stride) + VP9_ENC_BORDER_IN_PIXELS;

  gpu_output_pro_me += (global_row / (BLOCK_SIZE_IN_PIXELS / 8)) * group_stride + group_col;

  gpu_input += (global_row / (BLOCK_SIZE_IN_PIXELS / 8)) * (gpu_input_stride * 2) + (group_col * 2);

  src += global_offset;
  ref += global_offset;

  ref += ((gpu_output_pro_me->pred_mv.as_mv.row >> 3) * stride) +
      (gpu_output_pro_me->pred_mv.as_mv.col >> 3);

  segment_id = gpu_input->seg_id;

  if (segment_id == CR_SEGMENT_ID_BASE &&
      gpu_output_pro_me->pred_mv_sad < rd_parameters->vbp_threshold_sad) {
    gpu_input[0].do_compute = GPU_BLOCK_64X64;
    gpu_input[0].pred_mv.as_int = gpu_output_pro_me->pred_mv.as_int;
    gpu_input[1].do_compute = GPU_BLOCK_64X64;
    gpu_input[1].pred_mv.as_int = gpu_output_pro_me->pred_mv.as_int;
    gpu_input[gpu_input_stride].do_compute = GPU_BLOCK_64X64;
    gpu_input[gpu_input_stride].pred_mv.as_int = gpu_output_pro_me->pred_mv.as_int;
    gpu_input[gpu_input_stride + 1].do_compute = GPU_BLOCK_64X64;
    gpu_input[gpu_input_stride + 1].pred_mv.as_int = gpu_output_pro_me->pred_mv.as_int;

    return;
  }

  sum_array[0] = sum;
  sum_array[1] = sum + 64;
  sum_array[2] = sum + 64 + 16;
  sum_array[3] = sum + 64 + 16 + 4;

  sse_array[0] = sse;
  sse_array[1] = sse + 64;
  sse_array[2] = sse + 64 + 16;
  sse_array[3] = sse + 64 + 16 + 4;

  s_avg = vp9_avg_8x8(src, stride);
  d_avg = vp9_avg_8x8(ref, stride);

  gpu_output_pro_me->sum8x8.sum8x8[local_row * 8 + local_col] = s_avg - d_avg;
  sum_array[0][local_row * 8 + local_col] = s_avg - d_avg;
  sse_array[0][local_row * 8 + local_col] = (s_avg - d_avg) * (s_avg - d_avg);

  if (local_col == 0 && local_row == 0)
    force_split[0] = 0;

  if (local_col < 2 && local_row < 2)
    force_split[1 + local_row * 2 + local_col] = 0;

  if (local_col < 4 && local_row < 4)
    force_split[5 + local_row * 4 + local_col] = 0;

  int i = 2;
  int log2_count = 2;
  int blk_stride = 8;
  int blk_index = 0;
  while (i < 8) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_col % i == 0 && local_row % i == 0) {
      int j, k;
      int sum = 0;
      uint32_t sse = 0;
      for (j = 0; j < 2; j++) {
        for (k = 0; k < 2; k++) {
          sum += sum_array[blk_index][(j + local_row / (i >> 1)) * blk_stride +
                                      (k + local_col / (i >> 1))];
          sse += sse_array[blk_index][(j + local_row / (i >> 1)) * blk_stride +
                                      (k + local_col / (i >> 1))];
        }
      }
      sum_array[blk_index + 1][(local_row / i) * (blk_stride >> 1) +
                               (local_col / i)] = sum;
      sse_array[blk_index + 1][(local_row / i) * (blk_stride >> 1) +
                               (local_col / i)] = sse;
      variance = get_variance(sse, sum, log2_count);
      if (variance >= rd_parameters->seg_rd_param[segment_id].vbp_thresholds[i >> 2]) {
        if (i < 4)
          force_split[5 + ((local_row / 2) * 4 + (local_col / 2))] = 1; // 16X16
        force_split[1 + ((local_row / 4) * 2 + (local_col / 4))] = 1; // 32X32
        force_split[0] = 1; // 64X64
      } else if (i == 2 && segment_id == 0 &&
          variance >= rd_parameters->seg_rd_param[segment_id].vbp_thresholds[1]) {
        int minmax = compute_minmax_8x8(src, ref, stride);

        if (minmax > rd_parameters->vbp_threshold_minmax) {
          force_split[5 + ((local_row / 2) * 4 + (local_col / 2))] = 1;
          force_split[1 + ((local_row / 4) * 2 + (local_col / 4))] = 1;
          force_split[0] = 1;
        }
      }
    }
    blk_index += 1;
    blk_stride >>= 1;
    log2_count += 2;
    i <<= 1;
  }

select_partitions:
  barrier(CLK_LOCAL_MEM_FENCE);
  if (local_col == 0 && local_row == 0) {
    if (force_split[0] == 0) {
      int sum_1, sum_2;
      uint32_t sse_1, sse_2;

      // check for 64X64 partitions
      sse_1 = sse_array[2][0] + sse_array[2][2];
      sum_1 = sum_array[2][0] + sum_array[2][2];
      sse_2 = sse_array[2][1] + sse_array[2][3];
      sum_2 = sum_array[2][1] + sum_array[2][3];
      variance = get_variance(sse_1 + sse_2, sum_1 + sum_2, 6);
      if (variance < rd_parameters->seg_rd_param[segment_id].vbp_thresholds[2]) {
        gpu_input[0].do_compute = GPU_BLOCK_64X64;
        gpu_input[0].pred_mv.as_int = gpu_output_pro_me->pred_mv.as_int;
        gpu_input[1].do_compute = GPU_BLOCK_64X64;
        gpu_input[1].pred_mv.as_int = gpu_output_pro_me->pred_mv.as_int;
        gpu_input[gpu_input_stride].do_compute = GPU_BLOCK_64X64;
        gpu_input[gpu_input_stride].pred_mv.as_int = gpu_output_pro_me->pred_mv.as_int;
        gpu_input[gpu_input_stride + 1].do_compute = GPU_BLOCK_64X64;
        gpu_input[gpu_input_stride + 1].pred_mv.as_int = gpu_output_pro_me->pred_mv.as_int;
      }
      else {
        int var1, var2;

        force_split[0] = 1;
        var1 = get_variance(sse_1, sum_1, 5);
        var2 = get_variance(sse_2, sum_2, 5);
        if (var1 < rd_parameters->seg_rd_param[segment_id].vbp_thresholds[2] &&
            var2 < rd_parameters->seg_rd_param[segment_id].vbp_thresholds[2]) {
          gpu_input[0].do_compute = GPU_BLOCK_INVALID;
          gpu_input[1].do_compute = GPU_BLOCK_INVALID;
          gpu_input[gpu_input_stride].do_compute = GPU_BLOCK_INVALID;
          gpu_input[gpu_input_stride + 1].do_compute = GPU_BLOCK_INVALID;
          force_split[0] = 0;
        } else {
          sse_1 = sse_array[2][0] + sse_array[2][1];
          sum_1 = sum_array[2][0] + sum_array[2][1];
          sse_2 = sse_array[2][2] + sse_array[2][3];
          sum_2 = sum_array[2][2] + sum_array[2][3];
          var1 = get_variance(sse_1, sum_1, 5);
          var2 = get_variance(sse_2, sum_2, 5);
          if (var1 < rd_parameters->seg_rd_param[segment_id].vbp_thresholds[2] &&
              var2 < rd_parameters->seg_rd_param[segment_id].vbp_thresholds[2]) {
            gpu_input[0].do_compute = GPU_BLOCK_INVALID;
            gpu_input[1].do_compute = GPU_BLOCK_INVALID;
            gpu_input[gpu_input_stride].do_compute = GPU_BLOCK_INVALID;
            gpu_input[gpu_input_stride + 1].do_compute = GPU_BLOCK_INVALID;
            force_split[0] = 0;
          }
        }
      }
    }

    if (force_split[0] == 1) {
      // check for 32X32 partitions
      if (force_split[1] == 0) {
        gpu_input[0].do_compute = GPU_BLOCK_32X32;
        gpu_input[0].pred_mv.as_int = gpu_output_pro_me->pred_mv.as_int;
      } else {
        gpu_input[0].do_compute = GPU_BLOCK_INVALID;
      }

      if (force_split[2] == 0) {
        gpu_input[1].do_compute = GPU_BLOCK_32X32;
        gpu_input[1].pred_mv.as_int = gpu_output_pro_me->pred_mv.as_int;
      } else {
        gpu_input[1].do_compute = GPU_BLOCK_INVALID;
      }

      if (force_split[3] == 0) {
        gpu_input[gpu_input_stride].do_compute = GPU_BLOCK_32X32;
        gpu_input[gpu_input_stride].pred_mv.as_int = gpu_output_pro_me->pred_mv.as_int;
      } else {
        gpu_input[gpu_input_stride].do_compute = GPU_BLOCK_INVALID;
      }

      if (force_split[4] == 0) {
        gpu_input[gpu_input_stride + 1].do_compute = GPU_BLOCK_32X32;
        gpu_input[gpu_input_stride + 1].pred_mv.as_int = gpu_output_pro_me->pred_mv.as_int;
      } else {
        gpu_input[gpu_input_stride + 1].do_compute = GPU_BLOCK_INVALID;
      }
    }
  }
}
