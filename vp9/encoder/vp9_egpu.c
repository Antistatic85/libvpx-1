/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_config.h"
#include "vpx_mem/vpx_mem.h"

#include "vp9/common/vp9_mvref_common.h"

#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_egpu.h"

const BLOCK_SIZE vp9_actual_block_size_lookup[GPU_BLOCK_SIZES] = {
    BLOCK_32X32,
    BLOCK_64X64,
};

const BLOCK_SIZE vp9_gpu_block_size_lookup[BLOCK_SIZES] = {
    GPU_BLOCK_INVALID,
    GPU_BLOCK_INVALID,
    GPU_BLOCK_INVALID,
    GPU_BLOCK_INVALID,
    GPU_BLOCK_INVALID,
    GPU_BLOCK_INVALID,
    GPU_BLOCK_INVALID,
    GPU_BLOCK_INVALID,
    GPU_BLOCK_INVALID,
    GPU_BLOCK_32X32,
    GPU_BLOCK_INVALID,
    GPU_BLOCK_INVALID,
    GPU_BLOCK_64X64,
};

void vp9_gpu_set_mvinfo_offsets(VP9_COMP *const cpi, MACROBLOCK *const x,
                                int mi_row, int mi_col, BLOCK_SIZE bsize) {
  const VP9_COMMON *const cm = &cpi->common;
  const GPU_BLOCK_SIZE gpu_bsize = get_gpu_block_size(bsize);
  const int blocks_in_row = (cm->sb_cols * num_mxn_blocks_wide_lookup[bsize]);
  const int block_index_row = (mi_row >> mi_height_log2(bsize));
  const int block_index_col = (mi_col >> mi_width_log2(bsize));

  if (gpu_bsize != GPU_BLOCK_INVALID)
    x->gpu_output[gpu_bsize] = cpi->gpu_output_base[gpu_bsize] +
      (block_index_row * blocks_in_row) + block_index_col;
}

void vp9_find_mv_refs_dp(const VP9_COMMON *cm, const MACROBLOCKD *xd,
                         MODE_INFO *mi, MV_REFERENCE_FRAME ref_frame,
                         int_mv *mv_ref_list,
                         int mi_row, int mi_col,
                         uint8_t *mode_context) {
  int i, refmv_count = 0;
  const POSITION *const mv_ref_search = mv_ref_blocks[mi->mbmi.sb_type];
  const TileInfo *const tile_info = &xd->tile;

  // Blank the reference vector list
  memset(mv_ref_list, 0, sizeof(*mv_ref_list) * MAX_MV_REF_CANDIDATES);

  for (i = 0; i < MVREF_NEIGHBOURS; ++i) {
    const POSITION *const mv_ref = &mv_ref_search[i];
    MV_REF *prev_frame_mvs = cm->prev_frame->mvs + mi_row * cm->mi_cols + mi_col;
    if (is_inside(tile_info, mi_col, mi_row, cm->mi_rows, mv_ref)) {
      prev_frame_mvs += mv_ref->col + mv_ref->row * cm->mi_cols;

      if (prev_frame_mvs->ref_frame[0] == ref_frame)
        ADD_MV_REF_LIST(prev_frame_mvs->mv[0], refmv_count, mv_ref_list, Done);
      else if (prev_frame_mvs->ref_frame[1] == ref_frame) {
        ADD_MV_REF_LIST(prev_frame_mvs->mv[1], refmv_count, mv_ref_list, Done);
      }
    }
  }

  Done:

  mode_context[ref_frame] = 0;

  // Clamp vectors
  for (i = 0; i < MAX_MV_REF_CANDIDATES; ++i)
    clamp_mv_ref(&mv_ref_list[i].as_mv, xd);
}

void vp9_alloc_gpu_interface_buffers(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  GPU_BLOCK_SIZE gpu_bsize;

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
    const BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);
    const int blocks_in_row = (cm->sb_cols * num_mxn_blocks_wide_lookup[bsize]);
    const int blocks_in_col = (cm->sb_rows * num_mxn_blocks_high_lookup[bsize]);

    CHECK_MEM_ERROR(cm, cpi->gpu_output_base[gpu_bsize],
                    vpx_calloc(blocks_in_row * blocks_in_col,
                               sizeof(*cpi->gpu_output_base[gpu_bsize])));
  }
}

void vp9_free_gpu_interface_buffers(VP9_COMP *cpi) {
  GPU_BLOCK_SIZE gpu_bsize;

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
    vpx_free(cpi->gpu_output_base[gpu_bsize]);
    cpi->gpu_output_base[gpu_bsize] = NULL;
  }
}
