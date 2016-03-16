//
//  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
//
//  Use of this source code is governed by a BSD-style license
//  that can be found in the LICENSE file in the root of the source
//  tree. An additional intellectual property rights grant can be found
//  in the file PATENTS.  All contributing project authors may
//  be found in the AUTHORS file in the root of the source tree.
//

typedef struct {
  SUM_SSE sum_sse[9];
  char dummy[952];
} subpel_sum_sse;

//=====   GLOBAL DEFINITIONS   =====
//--------------------------------------
__constant MV hpel_offset[9] =
    {{0, -4}, {0, 4}, {-4, 0}, {4, 0}, {-4, -4}, {-4, 4}, {4, -4}, {4, 4}, {0, 0}};

__constant MV qpel_offset[8] =
    {{0, -2}, {0, 2}, {-2, 0}, {2, 0}, {-2, -2}, {-2, 2}, {2, -2}, {2, 2}};

//=====   FUNCTION MACROS   =====
//--------------------------------------

#define CHECK_BETTER_SUBPEL(offset, idx)                              \
      sum = intermediate_sum_sse[2 * idx];                            \
      sse = intermediate_sum_sse[2 * idx + 1];                        \
                                                                      \
      thiserr  = (sse - (((long int)sum * sum)                        \
              / (BLOCK_SIZE_IN_PIXELS * BLOCK_SIZE_IN_PIXELS)));      \
                                                                      \
      if (thiserr < besterr) {                                        \
        besterr = thiserr;                                            \
        best_mv.row = this_mv.row + offset[idx].row;                  \
        best_mv.col = this_mv.col + offset[idx].col;                  \
      }


//=====   FUNCTION DEFINITIONS   =====
//-------------------------------------------
void calculate_fullpel_variance(__global uchar *ref_frame,
                                __global uchar *cur_frame,
                                unsigned int *sse,
                                int *sum,
                                int stride) {
  uchar8 output;
  short8 diff;
  short8 vsum = 0;
  uint4 vsse = 0;
  short row;

  *sse = 0;
  *sum = 0;

  for(row = 0; row < PIXEL_ROWS_PER_WORKITEM; row++) {

    output = vload8(0, ref_frame);
    ref_frame += stride;

    uchar8 cur = vload8(0, cur_frame);
    cur_frame += stride;

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

__kernel
__attribute__((reqd_work_group_size((BLOCK_SIZE_IN_PIXELS * 8) / NUM_PIXELS_PER_WORKITEM,
                                    BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM,
                                    1)))
void vp9_sub_pixel_search_halfpel_filtering(__global uchar *ref_frame,
    __global uchar *cur_frame,
    int stride,
    __global GPU_INPUT *gpu_input,
    __global GPU_OUTPUT_ME *gpu_output_me,
    __global subpel_sum_sse *gpu_scratch,
    int padding_offset) {
  short global_row = get_global_id(1);

  short group_col = get_group_id(0);
  int group_stride = get_num_groups(0);

  int local_col = get_local_id(0);
  int global_offset = (global_row * PIXEL_ROWS_PER_WORKITEM * stride) +
                      ((group_col) * BLOCK_SIZE_IN_PIXELS) +
                      ((local_col >> 3) * NUM_PIXELS_PER_WORKITEM);
  global_offset += padding_offset;

#if BLOCK_SIZE_IN_PIXELS == 64
  GPU_BLOCK_SIZE gpu_bsize = GPU_BLOCK_64X64;
  int group_offset = (global_row / (BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM) *
      group_stride * 4 + (group_col) * 2);
#else
  GPU_BLOCK_SIZE gpu_bsize = GPU_BLOCK_32X32;
  int group_offset = (global_row / (BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM) *
      group_stride + (group_col));
#endif
  gpu_input += group_offset;
  gpu_scratch += group_offset;
  gpu_output_me += group_offset;

  if (gpu_input->do_compute != gpu_bsize)
    goto exit;

  if (gpu_output_me->rv)
    goto exit;

  cur_frame += global_offset;
  ref_frame += global_offset;

  int sum;
  unsigned int sse;

  MV best_mv = gpu_output_me->mv.as_mv;
  int buffer_offset;
  int local_offset;

  int idx = (local_col & 7);

  __global int *intermediate_sum_sse = (__global int *)gpu_scratch;

  // Half pel
  best_mv = best_mv + hpel_offset[idx];

  idx *= 2;
  buffer_offset = ((best_mv.row >> 3) * stride) + (best_mv.col >> 3);
  ref_frame += buffer_offset;

  if (!(local_col >> 3) && !get_local_id(1)) {
    vstore2(0, 0, intermediate_sum_sse + idx);
    if (idx == 2) {
      vstore2(0, 0, intermediate_sum_sse + 16);
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  if (idx == 2) {
    calculate_fullpel_variance(ref_frame, cur_frame, &sse, &sum, stride);
    atomic_add(intermediate_sum_sse + 16, sum);
    atomic_add(intermediate_sum_sse + 16 + 1, sse);
  }

  var_filter_block2d_bil_both(ref_frame, cur_frame, stride,
                              BILINEAR_FILTERS_2TAP(sp(best_mv.col)),
                              BILINEAR_FILTERS_2TAP(sp(best_mv.row)),
                              &sse, &sum);

  atomic_add(intermediate_sum_sse + idx, sum);
  atomic_add(intermediate_sum_sse + idx + 1, sse);

  barrier(CLK_GLOBAL_MEM_FENCE);

  if (!get_local_id(0) && !get_local_id(1)) {
    int sum, tr, tc;
    unsigned int besterr, sse, thiserr;
    MV best_mv = gpu_output_me->mv.as_mv;
    MV this_mv = best_mv;
    besterr = INT32_MAX;
    // Part 1
    tr = best_mv.row;
    tc = best_mv.col;

    CHECK_BETTER_SUBPEL(hpel_offset, 8);
    CHECK_BETTER_SUBPEL(hpel_offset, 0);
    CHECK_BETTER_SUBPEL(hpel_offset, 1);
    CHECK_BETTER_SUBPEL(hpel_offset, 2);
    CHECK_BETTER_SUBPEL(hpel_offset, 3);
    CHECK_BETTER_SUBPEL(hpel_offset, 4);
    CHECK_BETTER_SUBPEL(hpel_offset, 5);
    CHECK_BETTER_SUBPEL(hpel_offset, 6);
    CHECK_BETTER_SUBPEL(hpel_offset, 7);

    intermediate_sum_sse[16] = besterr;
    gpu_output_me->mv.as_mv = best_mv;
    vstore16(0, 0, intermediate_sum_sse);
  }

exit:
  return;
}

__kernel
__attribute__((reqd_work_group_size((BLOCK_SIZE_IN_PIXELS * 8) / NUM_PIXELS_PER_WORKITEM,
                                    BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM,
                                    1)))
void vp9_sub_pixel_search_quarterpel_filtering(__global uchar *ref_frame,
    __global uchar *cur_frame,
    int stride,
    __global GPU_INPUT *gpu_input,
    __global GPU_OUTPUT_ME *gpu_output_me,
    __global subpel_sum_sse *gpu_scratch,
    int padding_offset) {
  short global_row = get_global_id(1);

  short group_col = get_group_id(0);
  int group_stride = get_num_groups(0);

  int local_col = get_local_id(0);
  int global_offset = (global_row * PIXEL_ROWS_PER_WORKITEM * stride) +
                      ((group_col) * BLOCK_SIZE_IN_PIXELS) +
                      ((local_col >> 3) * NUM_PIXELS_PER_WORKITEM);
  global_offset += padding_offset;

#if BLOCK_SIZE_IN_PIXELS == 64
  GPU_BLOCK_SIZE gpu_bsize = GPU_BLOCK_64X64;
  int group_offset = (global_row / (BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM) *
      group_stride * 4 + group_col * 2);
#else
  GPU_BLOCK_SIZE gpu_bsize = GPU_BLOCK_32X32;
  int group_offset = (global_row / (BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM) *
      group_stride + group_col);
#endif

  gpu_input += group_offset;
  gpu_scratch += group_offset;
  gpu_output_me += group_offset;

  if (gpu_input->do_compute != gpu_bsize)
    goto exit;

  if (gpu_output_me->rv)
    goto exit;

  cur_frame += global_offset;
  ref_frame += global_offset;

  int sum;
  unsigned int sse;

  MV best_mv = gpu_output_me->mv.as_mv;
  int buffer_offset;

  int idx = (local_col & 7);

  __global int *intermediate_sum_sse = (__global int *)gpu_scratch;

  // Quarter pel
  best_mv = best_mv + qpel_offset[idx];

  idx *= 2;

  buffer_offset = ((best_mv.row >> 3) * stride) + (best_mv.col >> 3);
  ref_frame += buffer_offset;

  var_filter_block2d_bil_both(ref_frame, cur_frame, stride,
                              BILINEAR_FILTERS_2TAP(sp(best_mv.col)),
                              BILINEAR_FILTERS_2TAP(sp(best_mv.row)),
                              &sse, &sum);

  atomic_add(intermediate_sum_sse + idx, sum);
  atomic_add(intermediate_sum_sse + idx + 1, sse);

  barrier(CLK_GLOBAL_MEM_FENCE);

  if (!get_local_id(0) && !get_local_id(1)) {
    int sum, tr, tc;
    unsigned int besterr, sse, thiserr;

    MV best_mv = gpu_output_me->mv.as_mv;
    MV this_mv = best_mv;
    besterr = intermediate_sum_sse[16];

    // Part 2
    tr = best_mv.row;
    tc = best_mv.col;

    CHECK_BETTER_SUBPEL(qpel_offset, 0);
    CHECK_BETTER_SUBPEL(qpel_offset, 1);
    CHECK_BETTER_SUBPEL(qpel_offset, 2);
    CHECK_BETTER_SUBPEL(qpel_offset, 3);
    CHECK_BETTER_SUBPEL(qpel_offset, 4);
    CHECK_BETTER_SUBPEL(qpel_offset, 5);
    CHECK_BETTER_SUBPEL(qpel_offset, 6);
    CHECK_BETTER_SUBPEL(qpel_offset, 7);

    gpu_output_me->mv.as_mv = best_mv;
  }

exit:
  return;
}