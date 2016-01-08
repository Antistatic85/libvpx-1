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

struct GPU_OUTPUT_PRO_ME {
  char block_type[64];
  int_mv pred_mv;
  int pred_mv_sad;
  char color_sensitivity;
} __attribute__ ((aligned(32)));
typedef struct GPU_OUTPUT_PRO_ME GPU_OUTPUT_PRO_ME;

//=====   GLOBAL DEFINITIONS   =====
//--------------------------------------
__constant MV search_pos[4] = {
  {-1,  0},  // Top
  { 1,  0},  // Bottom
  { 0, -1},  // Left
  { 0,  1}   // Right
};

//=====   FUNCTION MACROS   =====
//--------------------------------------
#define CHECK_BETTER(sad, idx)                                    \
      sad = intermediate_sad[idx];                                \
                                                                  \
      if (sad < bestsad) {                                        \
        bestsad = sad;                                            \
        best_mv = this_mv + search_pos[idx];                      \
      }

//=====   FUNCTION DEFINITIONS   =====
//-------------------------------------------
void vp9_minmax_8x8(__global uchar *s, __global uchar *d, int stride,
                    uchar *max_val, uchar *min_val) {
  int i, j;
  uchar8 src, pred;
  uchar8 diff;

  *min_val = 255;
  *max_val = 0;
  for (i = 0; i < 8; ++i, s += stride, d += stride) {
    src = vload8(0, s);
    pred = vload8(0, d);
    diff = abs_diff(src, pred);

    *max_val = max(*max_val, diff.s0);
    *min_val = min(*min_val, diff.s0);
    *max_val = max(*max_val, diff.s1);
    *min_val = min(*min_val, diff.s1);
    *max_val = max(*max_val, diff.s2);
    *min_val = min(*min_val, diff.s2);
    *max_val = max(*max_val, diff.s3);
    *min_val = min(*min_val, diff.s3);
    *max_val = max(*max_val, diff.s4);
    *min_val = min(*min_val, diff.s4);
    *max_val = max(*max_val, diff.s5);
    *min_val = min(*min_val, diff.s5);
    *max_val = max(*max_val, diff.s6);
    *min_val = min(*min_val, diff.s6);
    *max_val = max(*max_val, diff.s7);
    *min_val = min(*min_val, diff.s7);
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
    uchar min_val;
    uchar max_val;

    vp9_minmax_8x8(src + y8_idx * stride + x8_idx,
                   ref + y8_idx * stride + x8_idx, stride,
                   &max_val, &min_val);
    if ((max_val - min_val) > minmax_max)
      minmax_max = (max_val - min_val);
    if ((max_val - min_val) < minmax_min)
      minmax_min = (max_val - min_val);
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

int set_vt_partitioning(__local int *var,
                        __local int *sum_array,
                        __local uint32_t *sse_array,
                        int stride, int log2_count,
                        BLOCK_SIZE bsize,
                        __global char *block_sz,
                        int threshold,
                        BLOCK_SIZE bsize_min,
                        int force_split,
                        int mi_row, int mi_col,
                        int mi_rows, int mi_cols) {
  int block_width = num_8x8_blocks_wide_lookup[bsize];
  int block_height = num_8x8_blocks_high_lookup[bsize];
  int var1, var2;
  int sum_1, sum_2;
  uint32_t sse_1, sse_2;

  if (force_split == 1)
    return 0;

  if (bsize == bsize_min) {
    if (mi_col + block_width / 2 < mi_cols &&
        mi_row + block_height / 2 < mi_rows &&
        *var < threshold) {
      *block_sz = bsize;
      return 1;
    }
    return 0;
  } else if (bsize > bsize_min) {
    if (mi_col + block_width / 2 < mi_cols &&
        mi_row + block_height / 2 < mi_rows &&
        *var < threshold) {
      *block_sz = bsize;
      return 1;
    }

    // Check vertical split.
    if (mi_row + block_height / 2 < mi_rows) {
      sse_1 = sse_array[0] + sse_array[stride];
      sum_1 = sum_array[0] + sum_array[stride];
      sse_2 = sse_array[1] + sse_array[stride + 1];
      sum_2 = sum_array[1] + sum_array[stride + 1];
      var1 = get_variance(sse_1, sum_1, log2_count);
      var2 = get_variance(sse_2, sum_2, log2_count);
      if (var1 < threshold && var2 < threshold &&
          ss_size_lookup[bsize - 2] < BLOCK_INVALID) {
        block_width >>= 1;
        *block_sz = bsize - 2;
        *(block_sz + block_width * block_width) = bsize - 2;
        return 1;
      }
    }

    // Check horizontal split.
    if (mi_col + block_width / 2 < mi_cols) {
      sse_1 = sse_array[0] + sse_array[1];
      sum_1 = sum_array[0] + sum_array[1];
      sse_2 = sse_array[stride] + sse_array[stride + 1];
      sum_2 = sum_array[stride] + sum_array[stride + 1];
      var1 = get_variance(sse_1, sum_1, log2_count);
      var2 = get_variance(sse_2, sum_2, log2_count);
      if (var1 < threshold && var2 < threshold &&
          ss_size_lookup[bsize - 1] < BLOCK_INVALID) {
        block_width >>= 1;
        *block_sz = bsize - 1;
        *(block_sz + block_width * block_width * 2) = bsize - 1;
        return 1;
      }
    }

    return 0;
  }
  return 0;
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
                   __global int* intermediate_sad) {
  int sum, sse;

  barrier(CLK_GLOBAL_MEM_FENCE);

  intermediate_sad[0] = 0;
  intermediate_sad[1] = 0;

  calculate_vector_var(ref_vector, src_vector, &sum, &sse);

  barrier(CLK_GLOBAL_MEM_FENCE);

  atomic_add(&intermediate_sad[0], sum);
  atomic_add(&intermediate_sad[1], sse);

  barrier(CLK_GLOBAL_MEM_FENCE);

  return intermediate_sad[1] - ((intermediate_sad[0] * intermediate_sad[0]) >> (bwl + 2));
}

int vector_match(__global ushort *proj_ref,
                 __global ushort *proj_src,
                 __global int* intermediate_int) {
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
                            __global ushort *ref_proj_c,
                            int padding_offset) {
  int global_row = get_global_id(1);
  int local_col = get_local_id(0);
  int group_col = get_group_id(0);
  int global_offset = (global_row * in_stride * BLOCK_SIZE_IN_PIXELS) +
      (local_col * in_stride) + (group_col * BLOCK_SIZE_IN_PIXELS);
  int out_stride = get_num_groups(1) * BLOCK_SIZE_IN_PIXELS;
  __local ushort8 intermediate_sum[8 * 4 * 2];
  __global uchar *src, *ref;

  // Column projection
  global_offset += padding_offset;

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
  global_offset += padding_offset;
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
  int group_col = get_group_id(0);
  int group_stride = get_num_groups(0);
  int global_row = get_global_id(1);
  int local_col = get_local_id(0);
  MV thismv;

  gpu_output_pro_me += global_row * group_stride + group_col;
  __global int *intermediate_int = (__global int *)&gpu_output_pro_me->block_type;

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
  barrier(CLK_GLOBAL_MEM_FENCE);
  intermediate_int[local_col] = 0;
  gpu_output_pro_me->pred_mv.as_mv = thismv;
}

__kernel
void vp9_pro_motion_estimation(__global uchar *cur,
                               __global uchar *ref,
                               int stride,
                               __global GPU_OUTPUT_PRO_ME *gpu_output_pro_me,
                               int padding_offset) {
  __global int *intermediate_sad;
  int global_row = get_global_id(1);
  int group_col = get_group_id(0);
  int group_stride = get_num_groups(0);
  int local_col = get_local_id(0);
  int group_offset = (global_row / (BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM) *
        group_stride + (group_col));
  int sad;
  int top, bottom;
  int left, right;
  MV best_mv, this_mv, next_mv;
  int buffer_offset;
  unsigned int bestsad;
  int global_offset = (global_row * PIXEL_ROWS_PER_WORKITEM * stride) +
      (group_col * BLOCK_SIZE_IN_PIXELS) +
      ((local_col >> 2) * NUM_PIXELS_PER_WORKITEM);
  global_offset += padding_offset;

  gpu_output_pro_me += group_offset;

  intermediate_sad = (__global int *)&gpu_output_pro_me->block_type;

  best_mv = this_mv = gpu_output_pro_me->pred_mv.as_mv;

  // Compute sad for pred MV and zero MV
  int idx = (local_col & 3);
  int tmp_idx = (local_col & 1);
  int local_offset = (idx >> 1) * (PIXEL_ROWS_PER_WORKITEM / 2) * stride;
  __global uchar *cur_frame = cur + (global_offset + local_offset);
  __global uchar *ref_frame = ref + (global_offset + local_offset);
  if (tmp_idx) {
    this_mv = 0;
  }

  sad = calculate_sad_rows(&this_mv, ref_frame, cur_frame, stride,
                           PIXEL_ROWS_PER_WORKITEM / 2);
  atomic_add(intermediate_sad + tmp_idx, sad);
  barrier(CLK_GLOBAL_MEM_FENCE);
  bestsad = intermediate_sad[0];
  if (bestsad >= intermediate_sad[1]) {
    best_mv = 0;
    bestsad = intermediate_sad[1];
  }
  intermediate_sad += 2;

  // Compute sad for 4 diamond points
  cur_frame = cur + global_offset;
  ref_frame = ref + global_offset;

  this_mv = best_mv + search_pos[idx];

  sad = calculate_sad(&this_mv, ref_frame, cur_frame, stride);
  atomic_add(intermediate_sad + idx, sad);
  barrier(CLK_GLOBAL_MEM_FENCE);

  // Check which among the 4 diamond points are best
  this_mv = best_mv;
  next_mv = best_mv;

  CHECK_BETTER(top, 0);
  CHECK_BETTER(bottom, 1);
  if (top < bottom) {
    next_mv.row -= 1;
  } else {
    next_mv.row += 1;
  }

  CHECK_BETTER(left, 2);
  CHECK_BETTER(right, 3);
  if (left < right) {
    next_mv.col -= 1;
  } else {
    next_mv.col += 1;
  }

  // Compute SAD for diagonal point
  local_offset = idx * (PIXEL_ROWS_PER_WORKITEM / 4) * stride;
  cur_frame = cur + global_offset + local_offset;
  ref_frame = ref + global_offset + local_offset;
  sad = calculate_sad_rows(&next_mv, ref_frame, cur_frame, stride, PIXEL_ROWS_PER_WORKITEM / 4);
  atomic_add(intermediate_sad + 4, sad);
  barrier(CLK_GLOBAL_MEM_FENCE);

  if (intermediate_sad[4] < bestsad) {
    best_mv = next_mv;
    bestsad = intermediate_sad[4];
  }

  if (get_local_id(0) == 0 && get_local_id(1) == 0) {
    gpu_output_pro_me->pred_mv.as_mv.col = best_mv.col << 3;
    gpu_output_pro_me->pred_mv.as_mv.row = best_mv.row << 3;
    gpu_output_pro_me->pred_mv_sad = bestsad;
  }
}

__kernel
void vp9_color_sensitivity(__global uchar *src,
                           __global uchar *ref,
                           int stride,
                           __global GPU_OUTPUT_PRO_ME *gpu_output_pro_me,
                           int64_t yplane_size,
                           int64_t uvplane_size,
                           int padding_offset) {
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
  global_offset += padding_offset;

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
                           __global GPU_RD_PARAMS_DYNAMIC *rd_parameters,
                           __global GPU_INPUT *gpu_input,
                           int gpu_input_stride,
                           int padding_offset,
                           int mi_rows, int mi_cols) {
  __local int sum[64 + 16 + 4 + 1];
  __local uint32_t sse[64 + 16 + 4 + 1];
  __local int *sum_array[4];
  __local uint32_t *sse_array[4];
  __local int force_split[21];
  __local int var[16 + 4 + 1];
  __local int *var_array[3];
  uint32_t s_avg, d_avg;
  int variance;
  int segment_id;
  int group_col = get_group_id(0);
  int group_row = get_group_id(1);
  int group_stride = get_num_groups(0);
  int global_row = get_global_id(1);
  int local_col = get_local_id(0);
  int local_row = get_local_id(1);
  int mi_row = (global_row >> 3) << 3;
  int mi_col = group_col << 3;
  int global_offset = (global_row * 8 * stride) +
      (group_col * BLOCK_SIZE_IN_PIXELS) + (local_col * NUM_PIXELS_PER_WORKITEM);
  global_offset += padding_offset;

  gpu_output_pro_me += (global_row / (BLOCK_SIZE_IN_PIXELS / 8)) * group_stride + group_col;

  gpu_input += (global_row / (BLOCK_SIZE_IN_PIXELS / 8)) * (gpu_input_stride * 2) + (group_col * 2);

  src += global_offset;
  ref += global_offset;

  ref += ((gpu_output_pro_me->pred_mv.as_mv.row >> 3) * stride) +
      (gpu_output_pro_me->pred_mv.as_mv.col >> 3);

  segment_id = gpu_input->seg_id;

  if (segment_id == CR_SEGMENT_ID_BASE &&
      gpu_output_pro_me->pred_mv_sad < rd_parameters->vbp_threshold_sad &&
      mi_col + 4 < mi_cols && mi_row + 4 < mi_rows) {
    if (local_col == 0 && local_row == 0) {
      gpu_output_pro_me->block_type[0] = BLOCK_64X64;
      gpu_input[0].do_compute = GPU_BLOCK_64X64;
      gpu_input[0].pred_mv.as_int = gpu_output_pro_me->pred_mv.as_int;
      gpu_input[1].do_compute = GPU_BLOCK_64X64;
      gpu_input[gpu_input_stride].do_compute = GPU_BLOCK_64X64;
      gpu_input[gpu_input_stride + 1].do_compute = GPU_BLOCK_64X64;
    }
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

  var_array[0] = var;
  var_array[1] = var + 16;
  var_array[2] = var + 16 + 4;

  s_avg = vp9_avg_8x8(src, stride);
  d_avg = vp9_avg_8x8(ref, stride);

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
  while (i <= 8) {
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
      // couldve used continue statement here but intel opencl compiler has
      // a problem with continue statments
      if (i == 4 && force_split[1 + ((local_row / 4) * 2 + (local_col / 4))])
        goto SKIP_VAR_CALC;
      if (i == 8 && force_split[0])
        goto SKIP_VAR_CALC;
      variance = get_variance(sse, sum, log2_count);
      var_array[blk_index][(local_row / i) * (blk_stride >> 1) +
                           (local_col / i)] = variance;
      if (variance >= rd_parameters->seg_rd_param[segment_id].vbp_thresholds[i >> 2]) {
        if (i < 4)
          force_split[5 + ((local_row / 2) * 4 + (local_col / 2))] = 1; // 16X16
        if (i < 8) {
          force_split[1 + ((local_row / 4) * 2 + (local_col / 4))] = 1; // 32X32
          force_split[0] = 1; // 64X64
        }
      } else if (i == 2 && segment_id == 0 &&
          variance >= rd_parameters->seg_rd_param[segment_id].vbp_thresholds[1]) {
        int minmax = compute_minmax_8x8(src, ref, stride);

        if (minmax > rd_parameters->vbp_threshold_minmax) {
          force_split[5 + ((local_row / 2) * 4 + (local_col / 2))] = 1;
          force_split[1 + ((local_row / 4) * 2 + (local_col / 4))] = 1;
          force_split[0] = 1;
        }
      }
      blk_index += 1;
      blk_stride >>= 1;
      log2_count += 2;
    }
SKIP_VAR_CALC:
    i <<= 1;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  if (local_col == 0 && local_row == 0) {
    // Now go through the entire structure, splitting every block size until
    // we get to one that's got a variance lower than our threshold.
    if (mi_col + 8 > mi_cols || mi_row + 8 > mi_rows ||
        !set_vt_partitioning(var_array[2], sum_array[2], sse_array[2],
                             2, 5, BLOCK_64X64, gpu_output_pro_me->block_type,
                             rd_parameters->seg_rd_param[segment_id].vbp_thresholds[2],
                             BLOCK_16X16,
                             force_split[0], mi_row, mi_col, mi_rows, mi_cols)) {
      for (i = 0; i < 4; ++i) {
        int x32_idx = (i & 1);
        int y32_idx = (i >> 1);
        int j;
        if (!set_vt_partitioning(&var_array[1][i],
                                 &sum_array[1][y32_idx * 8 + x32_idx * 2],
                                 &sse_array[1][y32_idx * 8 + x32_idx * 2],
                                 4, 3, BLOCK_32X32,
                                 &gpu_output_pro_me->block_type[i * 16],
                                 rd_parameters->seg_rd_param[segment_id].vbp_thresholds[1],
                                 BLOCK_16X16,
                                 force_split[i + 1],
                                 (mi_row + (y32_idx << 2)), (mi_col + (x32_idx << 2)), mi_rows, mi_cols)) {
          for (j = 0; j < 4; ++j) {
            int x16_idx = (j & 1);
            int y16_idx = (j >> 1);
            int k;
            if (!set_vt_partitioning(&var_array[0][y32_idx * 8 + x32_idx * 2 + y16_idx * 4 + x16_idx],
                                     &sum_array[0][y32_idx * 32 + x32_idx * 4 + y16_idx * 16 + x16_idx * 2],
                                     &sse_array[0][y32_idx * 32 + x32_idx * 4 + y16_idx * 16 + x16_idx * 2],
                                     8, 1, BLOCK_16X16,
                                     &gpu_output_pro_me->block_type[i * 16 + j * 4],
                                     rd_parameters->seg_rd_param[segment_id].vbp_thresholds[0],
                                     BLOCK_16X16,
                                     force_split[5 + y32_idx * 8 + x32_idx * 2 + y16_idx * 4 + x16_idx],
                                     (mi_row + (y32_idx << 2) + (y16_idx << 1)),
                                     (mi_col + (x32_idx << 2) + (x16_idx << 1)),
                                     mi_rows, mi_cols)) {
              for (k = 0; k < 4; ++k) {
                gpu_output_pro_me->block_type[i * 16 + j * 4 + k] = BLOCK_8X8;
              }
            }
          }
        }
      }
    }

    if (gpu_output_pro_me->block_type[0] == BLOCK_64X64) {
      gpu_input[0].do_compute = GPU_BLOCK_64X64;
      gpu_input[0].pred_mv.as_int = gpu_output_pro_me->pred_mv.as_int;
      gpu_input[1].do_compute = GPU_BLOCK_64X64;
      gpu_input[gpu_input_stride].do_compute = GPU_BLOCK_64X64;
      gpu_input[gpu_input_stride + 1].do_compute = GPU_BLOCK_64X64;
    } else if (gpu_output_pro_me->block_type[0] == BLOCK_32X64) {
      gpu_input[0].do_compute = GPU_BLOCK_INVALID;
      gpu_input[1].do_compute = GPU_BLOCK_INVALID;
      gpu_input[gpu_input_stride].do_compute = GPU_BLOCK_INVALID;
      gpu_input[gpu_input_stride + 1].do_compute = GPU_BLOCK_INVALID;
    } else {
      if (gpu_output_pro_me->block_type[0] == BLOCK_32X32) {
        gpu_input[0].do_compute = GPU_BLOCK_32X32;
        gpu_input[0].pred_mv.as_int = gpu_output_pro_me->pred_mv.as_int;
      } else {
        gpu_input[0].do_compute = GPU_BLOCK_INVALID;
      }
      if (gpu_output_pro_me->block_type[16] == BLOCK_32X32) {
        gpu_input[1].do_compute = GPU_BLOCK_32X32;
        gpu_input[1].pred_mv.as_int = gpu_output_pro_me->pred_mv.as_int;
      } else {
        gpu_input[1].do_compute = GPU_BLOCK_INVALID;
      }
      if (gpu_output_pro_me->block_type[32] == BLOCK_32X32) {
        gpu_input[gpu_input_stride].do_compute = GPU_BLOCK_32X32;
        gpu_input[gpu_input_stride].pred_mv.as_int = gpu_output_pro_me->pred_mv.as_int;
      } else {
        gpu_input[gpu_input_stride].do_compute = GPU_BLOCK_INVALID;
      }
      if (gpu_output_pro_me->block_type[48] == BLOCK_32X32) {
        gpu_input[gpu_input_stride + 1].do_compute = GPU_BLOCK_32X32;
        gpu_input[gpu_input_stride + 1].pred_mv.as_int = gpu_output_pro_me->pred_mv.as_int;
      } else {
        gpu_input[gpu_input_stride + 1].do_compute = GPU_BLOCK_INVALID;
      }
    }
  }
}
