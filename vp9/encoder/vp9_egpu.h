/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VP9_ENCODER_VP9_EGPU_H_
#define VP9_ENCODER_VP9_EGPU_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "vp9/common/vp9_enums.h"
#include "vp9/common/vp9_mv.h"

#define GPU_INTER_MODES 2 // ZEROMV and NEWMV

#define MAX_SUB_FRAMES 3

// Number of frames to wait before launching me/pro-me kernels asynchronously.
// NOTE: minimum value is 1
#define ASYNC_FRAME_COUNT_WAIT 30

#define GPU_INTER_OFFSET(mode) ((mode) - ZEROMV)

// Block sizes for which MV computations are done in GPU
typedef enum GPU_BLOCK_SIZE {
  GPU_BLOCK_32X32 = 0,
  GPU_BLOCK_64X64 = 1,
  GPU_BLOCK_SIZES,
  GPU_BLOCK_INVALID = GPU_BLOCK_SIZES
} GPU_BLOCK_SIZE;

struct VP9_COMP;
struct macroblockd;
struct ThreadData;
struct thread_context_gpu;

typedef struct GPU_INPUT {
  int_mv pred_mv;
  char do_compute;
  char seg_id;
} GPU_INPUT;

struct GPU_OUTPUT_ME {
  int64_t dist[GPU_INTER_MODES];
  int rate[GPU_INTER_MODES];
  unsigned int sse_y[GPU_INTER_MODES];
  unsigned int var_y[GPU_INTER_MODES];
  char interp_filter[GPU_INTER_MODES];
  char tx_size[GPU_INTER_MODES];
  char skip_txfm[GPU_INTER_MODES];
  char this_early_term[GPU_INTER_MODES];
  int_mv mv;
  int pred_mv_sad;
  int rv;
} __attribute__ ((aligned(32)));
typedef struct GPU_OUTPUT_ME GPU_OUTPUT_ME;

struct GPU_OUTPUT_PRO_ME {
  char block_type[64];
  int_mv pred_mv;
  int pred_mv_sad;
  char color_sensitivity;
  char pad[32];
} __attribute__ ((aligned(32)));
typedef struct GPU_OUTPUT_PRO_ME GPU_OUTPUT_PRO_ME;

typedef struct GPU_RD_PARAMS_STATIC {
  int rd_div;
  unsigned int inter_mode_cost[GPU_INTER_MODES];
  int switchable_interp_costs[SWITCHABLE_FILTERS];

  int nmvjointcost[MV_JOINTS];
  int nmvsadcost[2][MV_VALS];
} GPU_RD_PARAMS_STATIC;

typedef struct GPU_RD_SEG_PARAMETERS {
  int rd_mult;
  int dc_dequant;
  int ac_dequant;

  int sad_per_bit;

  int vbp_thresholds[3];
} GPU_RD_SEG_PARAMETERS;

typedef struct GPU_RD_PARAMS_DYNAMIC {
  int vbp_threshold_sad;
  int vbp_threshold_minmax;

  // Currently supporting only 2 segments in GPU
  GPU_RD_SEG_PARAMETERS seg_rd_param[2];
} GPU_RD_PARAMS_DYNAMIC;

typedef struct SubFrameInfo {
  int mi_row_start, mi_row_end;
} SubFrameInfo;

typedef struct VP9_EGPU {
  void *compute_framework;
  int (*alloc_buffers)(struct VP9_COMP *cpi);
  int (*free_buffers)(struct VP9_COMP *cpi);
  void (*acquire_input_buffer)(struct VP9_COMP *cpi, void **host_ptr);
  void (*acquire_output_me_buffer)(struct VP9_COMP *cpi, void **host_ptr,
      int sub_frame_idx);
  void (*acquire_output_pro_me_buffer)(struct VP9_COMP *cpi, void **host_ptr,
      int sub_frame_idx);
  void (*acquire_rd_param_buffer_static)(struct VP9_COMP *cpi, void **host_ptr);
  void (*acquire_rd_param_buffer_dynamic)(struct VP9_COMP *cpi, void **host_ptr,
      int index);
  void (*enc_sync_read)(struct VP9_COMP *cpi, int sub_frame_idx);
  void (*execute)(struct VP9_COMP *cpi, int sub_frame_idx, int async);
  void (*execute_prologue)(struct VP9_COMP *cpi);
  int (*remove)(struct VP9_COMP *cpi);
  void (*enc_profile)(struct VP9_COMP *cpi, int sub_frame_idx);
} VP9_EGPU;

extern const BLOCK_SIZE vp9_actual_block_size_lookup[GPU_BLOCK_SIZES];
extern const BLOCK_SIZE vp9_gpu_block_size_lookup[BLOCK_SIZES];

static INLINE BLOCK_SIZE get_actual_block_size(GPU_BLOCK_SIZE bsize) {
  return vp9_actual_block_size_lookup[bsize];
}

static INLINE GPU_BLOCK_SIZE get_gpu_block_size(BLOCK_SIZE bsize) {
  return vp9_gpu_block_size_lookup[bsize];
}

static INLINE int mi_width_log2(BLOCK_SIZE bsize) {
  return mi_width_log2_lookup[bsize];
}
static INLINE int mi_height_log2(BLOCK_SIZE bsize) {
  return mi_height_log2_lookup[bsize];
}

static INLINE int is_gpu_inter_mode(PREDICTION_MODE mode) {
  return (mode == ZEROMV || mode == NEWMV);
}

int vp9_get_gpu_buffer_index(struct VP9_COMP *const cpi, int mi_row, int mi_col);

void vp9_gpu_set_mvinfo_offsets(struct VP9_COMP *const cpi,
                                struct macroblock *const x,
                                int mi_row, int mi_col);

void vp9_subframe_init(SubFrameInfo *subframe, const VP9_COMMON *cm, int row);

int vp9_get_subframe_index(const VP9_COMMON *cm, int mi_row);

void vp9_alloc_gpu_interface_buffers(struct VP9_COMP *cpi);

void vp9_free_gpu_interface_buffers(struct VP9_COMP *cpi);

#if CONFIG_GPU_COMPUTE

void vp9_egpu_remove(struct VP9_COMP *cpi);

int vp9_egpu_init(struct VP9_COMP *cpi);

void vp9_gpu_mv_compute(struct thread_context_gpu *const egpu_thread_ctxt,
                        void* data2);

void vp9_gpu_mv_compute_async(struct thread_context_gpu *const egpu_thread_ctxt,
                              void* data2);

void vp9_enc_sync_gpu(struct VP9_COMP *cpi, struct ThreadData *td, int mi_row);

void vp9_enc_enqueue_jobs_gpu(struct VP9_COMP *cpi, struct ThreadData *td,
                              int mi_row);

#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VP9_ENCODER_VP9_EGPU_H_
