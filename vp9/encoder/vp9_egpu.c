/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
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
#if CONFIG_OPENCL
#include "vp9/common/opencl/vp9_opencl.h"
#endif

#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_egpu.h"
#if CONFIG_OPENCL
#include "vp9/encoder/opencl/vp9_eopencl.h"
#endif

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

int vp9_get_gpu_buffer_index(VP9_COMP *const cpi, int mi_row, int mi_col,
                             GPU_BLOCK_SIZE gpu_bsize) {
  const VP9_COMMON *const cm = &cpi->common;
  const BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);
  const int blocks_in_row = (cm->sb_cols * num_mxn_blocks_wide_lookup[bsize]);
  const int bsl = b_width_log2_lookup[bsize] - 1;
  return ((mi_row >> bsl) * blocks_in_row) + (mi_col >> bsl);
}

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

static int get_subframe_offset(int idx, int mi_rows, int sb_rows) {
  const int offset = ((idx * sb_rows) / MAX_SUB_FRAMES) << MI_BLOCK_SIZE_LOG2;
  return MIN(offset, mi_rows);
}

void vp9_subframe_init(SubFrameInfo *subframe, const VP9_COMMON *cm, int idx) {
  subframe->mi_row_start = get_subframe_offset(idx, cm->mi_rows, cm->sb_rows);
  subframe->mi_row_end = get_subframe_offset(idx + 1, cm->mi_rows, cm->sb_rows);
}

int vp9_get_subframe_index(const VP9_COMMON *cm, int mi_row) {
  int idx;

  for (idx = 0; idx < MAX_SUB_FRAMES; ++idx) {
    int mi_row_end = get_subframe_offset(idx + 1, cm->mi_rows, cm->sb_rows);
    if (mi_row < mi_row_end) {
      break;
    }
  }
  assert(idx < MAX_SUB_FRAMES);
  return idx;
}

void vp9_alloc_gpu_interface_buffers(VP9_COMP *cpi) {
#if !CONFIG_GPU_COMPUTE
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
#else
  cpi->egpu.alloc_buffers(cpi);
#endif
}

void vp9_free_gpu_interface_buffers(VP9_COMP *cpi) {
#if !CONFIG_GPU_COMPUTE
  GPU_BLOCK_SIZE gpu_bsize;

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
    vpx_free(cpi->gpu_output_base[gpu_bsize]);
    cpi->gpu_output_base[gpu_bsize] = NULL;
  }
#else
  cpi->egpu.free_buffers(cpi);
#endif
}

#if CONFIG_GPU_COMPUTE

void vp9_egpu_remove(VP9_COMP *cpi) {
  VP9_EGPU *egpu = &cpi->egpu;

  egpu->remove(cpi);
}

int vp9_egpu_init(VP9_COMP *cpi) {
#if CONFIG_OPENCL
  return vp9_eopencl_init(cpi);
#else
  return 1;
#endif
}

static void vp9_gpu_fill_rd_parameters(VP9_COMP *cpi, MACROBLOCK *const x) {
  VP9_EGPU *egpu = &cpi->egpu;
  struct macroblockd_plane *const pd = &x->e_mbd.plane[0];
  GPU_RD_PARAMETERS *rd_param_ptr;

  egpu->acquire_rd_param_buffer(cpi, (void **)&rd_param_ptr);

  rd_param_ptr->tx_mode = cpi->common.tx_mode;
  rd_param_ptr->dc_dequant = pd->dequant[0];
  rd_param_ptr->ac_dequant = pd->dequant[1];
}

static void vp9_gpu_fill_mv_input(VP9_COMP *cpi, const TileInfo * const tile) {
  SPEED_FEATURES * const sf = &cpi->sf;

  switch (sf->partition_search_type) {
    case VAR_BASED_PARTITION:
      vp9_fill_mv_reference_partition(cpi, tile);
      break;
    default:
      assert(0);
      break;
  }
}

void vp9_gpu_mv_compute(VP9_COMP *cpi, MACROBLOCK *const x) {
  VP9_COMMON *const cm = &cpi->common;
  const int tile_cols = 1 << cm->log2_tile_cols;
  int tile_col, tile_row;
  const int tile_rows = 1 << cm->log2_tile_rows;
  VP9_EGPU * const egpu = &cpi->egpu;
  GPU_BLOCK_SIZE gpu_bsize;
  int subframe_idx;

  // fill rd param info
  vp9_gpu_fill_rd_parameters(cpi, x);

  // fill mv info
  for (tile_row = 0; tile_row < tile_rows; ++tile_row) {
    for (tile_col = 0; tile_col < tile_cols; ++tile_col) {
      TileInfo tile;

      vp9_tile_init(&tile, cm, tile_row, tile_col);
      vp9_gpu_fill_mv_input(cpi, &tile);
    }
  }

  // enqueue kernels for gpu
  for (subframe_idx = CPU_SUB_FRAMES; subframe_idx < MAX_SUB_FRAMES;
       subframe_idx++) {
    for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
      egpu->execute(cpi, gpu_bsize, subframe_idx);
    }
  }

  // re-map source and reference pointers before starting cpu side processing
  vp9_acquire_frame_buffer(cm, cpi->Source);
  vp9_acquire_frame_buffer(cm, get_ref_frame_buffer(cpi, LAST_FRAME));
}

#endif
