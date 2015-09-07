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

#if CONFIG_GPU_COMPUTE
#define GPU_INTER_MODES 1 // ZEROMV
#else
#define GPU_INTER_MODES 2 // ZEROMV and NEWMV
#endif

#define MAX_SUB_FRAMES 1
#define CPU_SUB_FRAMES 0

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

typedef struct GPU_INPUT {
  int_mv nearest_mv;
  int_mv near_mv;
  int filter_type;
  int do_compute;
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

typedef struct GPU_RD_PARAMETERS {
  TX_MODE tx_mode;
  int dc_dequant;
  int ac_dequant;
} GPU_RD_PARAMETERS;

typedef struct SubFrameInfo {
  int mi_row_start, mi_row_end;
} SubFrameInfo;

typedef struct VP9_EGPU {
  void *compute_framework;
  GPU_INPUT *gpu_input[GPU_BLOCK_SIZES];
  void (*alloc_buffers)(struct VP9_COMP *cpi);
  void (*free_buffers)(struct VP9_COMP *cpi);
  void (*acquire_input_buffer)(struct VP9_COMP *cpi, GPU_BLOCK_SIZE gpu_bsize,
      void **host_ptr);
  void (*acquire_output_buffer)(struct VP9_COMP *cpi, GPU_BLOCK_SIZE gpu_bsize,
      void **host_ptr, int sub_frame_idx);
  void (*acquire_rd_param_buffer)(struct VP9_COMP *cpi, void **host_ptr);
  void (*enc_sync_read)(struct VP9_COMP *cpi, int event_id);
  void (*execute)(struct VP9_COMP *cpi, GPU_BLOCK_SIZE gpu_bsize,
      int sub_frame_idx);
  void (*remove)(struct VP9_COMP *cpi);
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
  // this function will be removed once the newmv opencl computation kernels
  // are added
#if CONFIG_GPU_COMPUTE
  return (mode == ZEROMV);
#else
  return (mode == ZEROMV || mode == NEWMV);
#endif
}

int vp9_get_gpu_buffer_index(struct VP9_COMP *const cpi, int mi_row, int mi_col,
                             GPU_BLOCK_SIZE gpu_bsize);

void vp9_gpu_set_mvinfo_offsets(struct VP9_COMP *const cpi,
                                struct macroblock *const x,
                                int mi_row, int mi_col, BLOCK_SIZE bsize);

void vp9_find_mv_refs_dp(const VP9_COMMON *cm, const MACROBLOCKD *xd,
                         MODE_INFO *mi, MV_REFERENCE_FRAME ref_frame,
                         int_mv *mv_ref_list,
                         int mi_row, int mi_col,
                         uint8_t *mode_context);

void vp9_subframe_init(SubFrameInfo *subframe, const VP9_COMMON *cm, int row);

int vp9_get_subframe_index(const VP9_COMMON *cm, int mi_row);

void vp9_alloc_gpu_interface_buffers(struct VP9_COMP *cpi);

void vp9_free_gpu_interface_buffers(struct VP9_COMP *cpi);

#if CONFIG_GPU_COMPUTE

void vp9_egpu_remove(struct VP9_COMP *cpi);

int vp9_egpu_init(struct VP9_COMP *cpi);

void vp9_fill_mv_reference_partition(struct VP9_COMP *cpi,
                                     const TileInfo *const tile);

void vp9_gpu_mv_compute(struct VP9_COMP *cpi, struct macroblock *const x);

#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VP9_ENCODER_VP9_EGPU_H_
