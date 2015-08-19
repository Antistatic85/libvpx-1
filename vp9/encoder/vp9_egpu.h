/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
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

struct GPU_OUTPUT {
  int pred_mv_sad;
  int rv;
  struct MODE_OUTPUT {
    RD_COST this_rdc;
    int_mv mv;
    int interp_filter;
    int tx_size;
    unsigned int sse_y;
  } modeoutput[GPU_INTER_MODES];
} __attribute__ ((aligned(32)));
typedef struct GPU_OUTPUT GPU_OUTPUT;

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

void vp9_gpu_set_mvinfo_offsets(struct VP9_COMP *const cpi,
                                struct macroblock *const x,
                                int mi_row, int mi_col, BLOCK_SIZE bsize);

void vp9_find_mv_refs_dp(const VP9_COMMON *cm, const MACROBLOCKD *xd,
                         MODE_INFO *mi, MV_REFERENCE_FRAME ref_frame,
                         int_mv *mv_ref_list,
                         int mi_row, int mi_col,
                         uint8_t *mode_context);

void vp9_alloc_gpu_interface_buffers(struct VP9_COMP *cpi);
void vp9_free_gpu_interface_buffers(struct VP9_COMP *cpi);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VP9_ENCODER_VP9_EGPU_H_
