/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VP9_ENCODER_VP9_EOPENCL_H_
#define VP9_ENCODER_VP9_EOPENCL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "vp9/common/opencl/CL/cl.h"
#include "vp9/encoder/vp9_encoder.h"

#define NUM_PIXELS_PER_WORKITEM 8

typedef struct {
  int sum;
  unsigned int sse;
} SUM_SSE;

typedef struct {
  SUM_SSE sum_sse[EIGHTTAP_SMOOTH + 1][64];
}GPU_SCRATCH;

typedef struct VP9_EOPENCL {
  VP9_OPENCL *opencl;

  // gpu me interface buffers
  opencl_buffer gpu_input;
  cl_mem gpu_output_me;
  opencl_buffer gpu_output_me_sub_buf[MAX_SUB_FRAMES];
  opencl_buffer rdopt_params_dyn[NUM_PING_PONG_BUFFERS];
  opencl_buffer rdopt_params_static;

  cl_mem gpu_scratch;

  // gpu me kernels
  cl_kernel rd_calculation_zeromv[GPU_BLOCK_SIZES];
  cl_kernel full_pixel_search[GPU_BLOCK_SIZES];
  cl_kernel hpel_search[GPU_BLOCK_SIZES];
  cl_kernel qpel_search[GPU_BLOCK_SIZES];
  cl_kernel inter_prediction_and_sse[GPU_BLOCK_SIZES];
  cl_kernel rd_calculation_newmv[GPU_BLOCK_SIZES];

  // gpu choose partitioning interface buffers
  cl_mem pred_1d_set[2];
  cl_mem src_1d_set[2];

  // buffer pair to be used alternately across frames (ping-pong)
  cl_mem gpu_output_pro_me[NUM_PING_PONG_BUFFERS];
  opencl_buffer gpu_output_pro_me_sub_buf[NUM_PING_PONG_BUFFERS][MAX_SUB_FRAMES];

  // gpu choose partitioning kernels
  cl_kernel col_row_projection;
  cl_kernel vector_match;
  cl_kernel pro_motion_estimation;
  cl_kernel color_sensitivity;
  cl_kernel choose_partitions;

  // gpu cpu sync handlers
  cl_event event[NUM_PING_PONG_BUFFERS][MAX_SUB_FRAMES];

  // gpu profiling code
#if OPENCL_PROFILING
  cl_event event_prome[NUM_PING_PONG_BUFFERS][4];
  cl_event event_me[NUM_PING_PONG_BUFFERS][MAX_SUB_FRAMES][11];
  cl_ulong total_time_taken_pro_me[4];
  cl_ulong total_time_taken_me[MAX_SUB_FRAMES];
#endif
} VP9_EOPENCL;

int vp9_eopencl_remove(VP9_COMP *cpi);

int vp9_eopencl_init(VP9_COMP *cpi);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* VP9_ENCODER_VP9_EOPENCL_H_ */
