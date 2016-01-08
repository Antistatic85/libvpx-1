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
#include "vpx_ports/vpx_timer.h"

#include "vp9/common/vp9_mvref_common.h"
#if CONFIG_OPENCL
#include "vp9/common/opencl/vp9_opencl.h"
#endif
#include "vp9/common/vp9_pred_common.h"

#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_egpu.h"
#if CONFIG_OPENCL
#include "vp9/encoder/opencl/vp9_eopencl.h"
#endif
#include "vp9/encoder/vp9_encodeframe.h"

// Maintain the block sizes in ascending order. All memory allocations, offset
// calculations happens on the lowest block size.
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

static void vp9_gpu_fill_segment_rd_parameters(VP9_COMP *cpi,
                                               GPU_RD_SEG_PARAMETERS *seg_rd,
                                               int segment_id) {
  VP9_COMMON *const cm = &cpi->common;
  const int qindex = vp9_get_qindex(&cm->seg, segment_id, cm->base_qindex);
  int rdmult = vp9_compute_rd_mult(cpi, qindex + cm->y_dc_delta_q);
  int64_t thresholds[4] = {cpi->vbp_thresholds[0], cpi->vbp_thresholds[1],
        cpi->vbp_thresholds[2], cpi->vbp_thresholds[3]};
  seg_rd->rd_mult = rdmult;
  seg_rd->dc_dequant = cpi->y_dequant[qindex][0];
  seg_rd->ac_dequant = cpi->y_dequant[qindex][1];
  seg_rd->sad_per_bit = vp9_get_sad_per_bit16(cpi, qindex);

  if (cyclic_refresh_segment_id_boosted(segment_id)) {
    set_vbp_thresholds(cpi, thresholds, qindex);
  }
  seg_rd->vbp_thresholds[0] = thresholds[2];
  seg_rd->vbp_thresholds[1] = thresholds[1];
  seg_rd->vbp_thresholds[2] = thresholds[0];
}

static void vp9_gpu_fill_rd_params_dynamic(VP9_COMP *cpi,
                                           GPU_RD_PARAMS_DYNAMIC *rd_param_ptr,
                                           int q) {
  VP9_COMMON *const cm = &cpi->common;

  rd_param_ptr->vbp_threshold_sad = (cpi->y_dequant[q][1] << 1) > 1000 ?
      (cpi->y_dequant[q][1] << 1) : 1000;
  rd_param_ptr->vbp_threshold_minmax = 15 + (q >> 3);

  vp9_gpu_fill_segment_rd_parameters(cpi, &rd_param_ptr->seg_rd_param[0], 0);
  if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ && cm->seg.enabled)
    vp9_gpu_fill_segment_rd_parameters(cpi, &rd_param_ptr->seg_rd_param[1], 1);
}

static void vp9_gpu_fill_rd_params_static(VP9_COMP *cpi, ThreadData *td,
                                          GPU_RD_PARAMS_STATIC *rd_param_ptr) {
  MACROBLOCK *const x = &td->mb;
  int i;

  rd_param_ptr->rd_div = cpi->rd.RDDIV;
  rd_param_ptr->inter_mode_cost[0] =
      cpi->inter_mode_cost[BOTH_PREDICTED][INTER_OFFSET(ZEROMV)];
  rd_param_ptr->inter_mode_cost[1] =
      cpi->inter_mode_cost[BOTH_PREDICTED][INTER_OFFSET(NEWMV)];
  for (i = 0; i < SWITCHABLE_FILTERS; i++)
    rd_param_ptr->switchable_interp_costs[i] =
        cpi->switchable_interp_costs[SWITCHABLE_FILTERS][i];
  for(i = 0; i < MV_JOINTS; i++) {
    rd_param_ptr->nmvjointcost[i] = x->nmvjointcost[i];
  }
  memcpy(rd_param_ptr->nmvsadcost[0], x->nmvsadcost[0] - MV_MAX,
         sizeof(rd_param_ptr->nmvsadcost[0]));
  memcpy(rd_param_ptr->nmvsadcost[1], x->nmvsadcost[1] - MV_MAX,
         sizeof(rd_param_ptr->nmvsadcost[1]));
}

static void vp9_gpu_fill_rd_parameters(VP9_COMP *cpi, ThreadData *td, int async) {
  VP9_COMMON *const cm = &cpi->common;
  VP9_EGPU *egpu = &cpi->egpu;
  GPU_RD_PARAMS_DYNAMIC *rd_param_ptr_dyn;
  int buff_index, q = cm->base_qindex;

  assert(cpi->common.tx_mode == TX_MODE_SELECT);

  if (cm->current_video_frame <= 2) {
    GPU_RD_PARAMS_STATIC *rd_param_ptr_static;

    egpu->acquire_rd_param_buffer_static(cpi, (void **) &rd_param_ptr_static);
    vp9_gpu_fill_rd_params_static(cpi, td, rd_param_ptr_static);
  }

  buff_index = async ? ((cm->current_video_frame + 1) & 1) :
      (cm->current_video_frame & 1);
  if (cm->current_video_frame > ASYNC_FRAME_COUNT_WAIT &&
      MAX_SUB_FRAMES > 2) {
    q = async ? cpi->rc.q_prediction_next : cpi->rc.q_prediction_curr;
  }
  egpu->acquire_rd_param_buffer_dynamic(cpi, (void **) &rd_param_ptr_dyn,
                                        buff_index);
  vp9_gpu_fill_rd_params_dynamic(cpi, rd_param_ptr_dyn, q);
}

static void vp9_gpu_fill_seg_id(VP9_COMP *cpi, int mi_row_start, int mi_row_end) {
  VP9_COMMON *const cm = &cpi->common;
  VP9_EGPU *egpu = &cpi->egpu;
  GPU_INPUT *gpu_input_base = NULL;
  int mi_row, mi_col;
  GPU_BLOCK_SIZE gpu_bsize = 0;
  BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);
  const int mi_row_step = num_8x8_blocks_high_lookup[bsize];
  const int mi_col_step = num_8x8_blocks_wide_lookup[bsize];

  egpu->acquire_input_buffer(cpi, (void **) &gpu_input_base);

  // NOTE: Although get_segment_id() operates at bsize level, currently
  // the supported segmentation feature in GPU maintains same seg_id for the
  // entire SB. If this is not the case in the future then make seg id as an
  // array and fill it for all GPU_BLOCK_SIZES
  for (mi_row = mi_row_start; mi_row < mi_row_end; mi_row += mi_row_step) {
    for (mi_col = 0; mi_col < cm->mi_cols; mi_col += mi_col_step) {
      GPU_INPUT *gpu_input = gpu_input_base +
          vp9_get_gpu_buffer_index(cpi, mi_row, mi_col);

      if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ && cm->seg.enabled) {
        const uint8_t *const map = cm->seg.update_map ? cpi->segmentation_map :
            cm->last_frame_seg_map;
        gpu_input->seg_id = get_segment_id(cm, map, bsize, mi_row, mi_col);
        // Only 2 segments are supported in GPU
        assert(gpu_input->seg_id <= 1);
      } else {
        gpu_input->seg_id = CR_SEGMENT_ID_BASE;
      }
    }
  }
}

void vp9_gpu_mv_compute(VP9_COMP *cpi, ThreadData *td) {
  VP9_COMMON *const cm = &cpi->common;
  VP9_EGPU *const egpu = &cpi->egpu;
  int subframe_idx, subframe_idx_start;
  SubFrameInfo subframe;

  subframe_idx_start = cpi->b_async;
  vp9_subframe_init(&subframe, cm, subframe_idx_start);

  // fill segmentation map
  vp9_gpu_fill_seg_id(cpi, subframe.mi_row_start, cm->mi_rows);

  if (MAX_SUB_FRAMES <= 2 || !cpi->b_async) {
    // fill rd param info
    vp9_gpu_fill_rd_parameters(cpi, td, 0);
  }

  for (subframe_idx = subframe_idx_start; subframe_idx < MAX_SUB_FRAMES;
       subframe_idx++) {
    // enqueue kernels for gpu
    egpu->execute(cpi, subframe_idx, 0);
  }
}

void vp9_gpu_mv_compute_async(VP9_COMP *cpi, ThreadData *td, int subframe_idx) {
  VP9_COMMON *const cm = &cpi->common;
  VP9_EGPU *const egpu = &cpi->egpu;
  struct lookahead_entry *next_source = NULL;
  struct vpx_usec_timer emr_timer;

  vpx_usec_timer_start(&emr_timer);

  next_source = vp9_lookahead_peek(cpi->lookahead, 0);
  if (next_source == NULL)
    return;

  // enqueue prologue kernels for gpu
  if (subframe_idx == 0) {
    egpu->execute_prologue(cpi);
  }

  // enqueue me kernels for gpu
  if (subframe_idx == MAX_SUB_FRAMES - 1) {
    if (cpi->b_async) {
      SubFrameInfo subframe;
      unsigned char *seg_map = cpi->segmentation_map;
      CYCLIC_REFRESH *const cr = cpi->cyclic_refresh;
      double rate_ratio_qdelta = cr->rate_ratio_qdelta;
      int q = cpi->rc.q_prediction_next;
      int base_qindex = cm->base_qindex;

      vp9_subframe_init(&subframe, cm, 0);

      if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ) {
        if (cr->percent_refresh > 0 &&
            (cpi->rc.frames_since_key + 1) >= (4 * cpi->svc.number_temporal_layers) *
            (100 / cr->percent_refresh)) {
          cr->rate_ratio_qdelta = 2.0;
          if (cr->rate_ratio_qdelta != rate_ratio_qdelta) {
            vp9_gpu_cyclic_refresh_qindex_setup(cpi);
          }
        }
      }
      if (cm->base_qindex != q || cr->rate_ratio_qdelta != rate_ratio_qdelta) {
        vp9_gpu_rewrite_quant_info(cpi, &td->mb, q);
      }

      // fill segmentation map
      if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ) {
        cpi->segmentation_map = cpi->seg_map_pred;
        cyclic_refresh_update_map(cpi, 1);
      }
      vp9_gpu_fill_seg_id(cpi, subframe.mi_row_start, subframe.mi_row_end);
      if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ) {
        cpi->segmentation_map = seg_map;
      }

      // fill rd param info
      vp9_gpu_fill_rd_parameters(cpi, td, 1);

      // enqueue kernels for gpu
      egpu->execute(cpi, 0, 1);

      if (cm->base_qindex != base_qindex ||
          cr->rate_ratio_qdelta != rate_ratio_qdelta) {
        if (cr->rate_ratio_qdelta != rate_ratio_qdelta) {
          cr->rate_ratio_qdelta = rate_ratio_qdelta;
          vp9_gpu_cyclic_refresh_qindex_setup(cpi);
        }
        vp9_gpu_rewrite_quant_info(cpi, &td->mb, base_qindex);
      }
    }
  }

  vpx_usec_timer_mark(&emr_timer);
  cpi->time_gpu_compute += vpx_usec_timer_elapsed(&emr_timer);
}
#endif

int vp9_get_gpu_buffer_index(VP9_COMP *const cpi, int mi_row, int mi_col) {
  const VP9_COMMON *const cm = &cpi->common;
  const BLOCK_SIZE bsize = vp9_actual_block_size_lookup[0];
  const int blocks_in_row = cm->sb_cols * num_mxn_blocks_wide_lookup[bsize];
  const int bsl = b_width_log2_lookup[bsize] - 1;
  return ((mi_row >> bsl) * blocks_in_row) + (mi_col >> bsl);
}

void vp9_gpu_set_mvinfo_offsets(VP9_COMP *const cpi, MACROBLOCK *const x,
                                int mi_row, int mi_col) {
  const VP9_COMMON *const cm = &cpi->common;
  const BLOCK_SIZE bsize = vp9_actual_block_size_lookup[0];
  const int blocks_in_row = cm->sb_cols * num_mxn_blocks_wide_lookup[bsize];
  const int block_index_row = (mi_row >> mi_height_log2(bsize));
  const int block_index_col = (mi_col >> mi_width_log2(bsize));

  x->gpu_output_me = cpi->gpu_output_me_base +
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
  const BLOCK_SIZE bsize = vp9_actual_block_size_lookup[0];

  const int blocks_in_row = cm->sb_cols * num_mxn_blocks_wide_lookup[bsize];
  const int blocks_in_col = cm->sb_rows * num_mxn_blocks_high_lookup[bsize];

  CHECK_MEM_ERROR(cm, cpi->gpu_output_me_base,
                  vpx_calloc(blocks_in_row * blocks_in_col,
                             sizeof(*cpi->gpu_output_me_base)));
#else
  cpi->egpu.alloc_buffers(cpi);
#endif
}

void vp9_free_gpu_interface_buffers(VP9_COMP *cpi) {
#if !CONFIG_GPU_COMPUTE
  vpx_free(cpi->gpu_output_me_base);
  cpi->gpu_output_me_base = NULL;
#else
  cpi->egpu.free_buffers(cpi);
#endif
}

void vp9_enc_sync_gpu(VP9_COMP *cpi, ThreadData *td, int mi_row) {
  VP9_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &td->mb;

  (void) x;
  // When gpu is enabled, before encoding the current row, make sure the
  // necessary dependencies are met.
  if (cm->use_gpu && cpi->sf.use_nonrd_pick_mode) {
    SubFrameInfo subframe;
    int subframe_idx;

    subframe_idx = vp9_get_subframe_index(cm, mi_row);
    vp9_subframe_init(&subframe, cm, subframe_idx);
#if CONFIG_GPU_COMPUTE
    if (!frame_is_intra_only(cm)) {
      if (!x->data_parallel_processing && x->use_gpu) {
        VP9_EGPU *egpu = &cpi->egpu;

        egpu->enc_sync_read(cpi, subframe_idx, 0);
        egpu->acquire_output_pro_me_buffer(
            cpi, (void **) &cpi->gpu_output_pro_me_base, 0);
        if (mi_row == subframe.mi_row_start) {
          GPU_OUTPUT_PRO_ME *gpu_output_pro_me_subframe;
          const int sb_row_index = mi_row >> MI_BLOCK_SIZE_LOG2;
          const int buffer_offset = (cm->mi_cols >> MI_BLOCK_SIZE_LOG2) *
              sb_row_index;

          (void) buffer_offset;
          egpu->acquire_output_pro_me_buffer(
              cpi, (void **) &gpu_output_pro_me_subframe, subframe_idx);
          assert(gpu_output_pro_me_subframe - cpi->gpu_output_pro_me_base ==
              buffer_offset);
          if (cpi->max_threads > 1) {
            const int sb_row = subframe.mi_row_start >> MI_BLOCK_SIZE_LOG2;
            const int sb_col = cm->mi_cols >> MI_BLOCK_SIZE_LOG2;

            vp9_enc_sync_read(cpi, sb_row, sb_col);
          }
          vp9_gpu_mv_compute_async(cpi, td, subframe_idx);
        }
        egpu->enc_sync_read(cpi, subframe_idx, MAX_SUB_FRAMES);
        egpu->acquire_output_me_buffer(
            cpi, (void **) &cpi->gpu_output_me_base, 0);
        if (mi_row == subframe.mi_row_start) {
          GPU_OUTPUT_ME *gpu_output_me_subframe;
          const int buffer_offset =
              vp9_get_gpu_buffer_index(cpi, subframe.mi_row_start, 0);

          (void)buffer_offset;
          egpu->acquire_output_me_buffer(
              cpi, (void **) &gpu_output_me_subframe, subframe_idx);
          assert(gpu_output_me_subframe - cpi->gpu_output_me_base ==
              buffer_offset);
        }
      }
    } else {
      if (mi_row == subframe.mi_row_start) {
        if (cpi->max_threads > 1) {
          const int sb_row = subframe.mi_row_start >> MI_BLOCK_SIZE_LOG2;
          const int sb_col = cm->mi_cols >> MI_BLOCK_SIZE_LOG2;

          vp9_enc_sync_read(cpi, sb_row, sb_col);
        }
        vp9_gpu_mv_compute_async(cpi, td, subframe_idx);
      }
    }
#else
    if (MAX_SUB_FRAMES > 2 &&
        cm->current_video_frame >= ASYNC_FRAME_COUNT_WAIT &&
        cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ &&
        subframe_idx == MAX_SUB_FRAMES - 1 &&
        mi_row == subframe.mi_row_start &&
        !x->data_parallel_processing) {
      unsigned char *seg_map = cpi->segmentation_map;

      if (cpi->max_threads > 1) {
        const int sb_row = subframe.mi_row_start >> MI_BLOCK_SIZE_LOG2;
        const int sb_col = cm->mi_cols >> MI_BLOCK_SIZE_LOG2;

        vp9_enc_sync_read(cpi, sb_row, sb_col);
      }

      assert(cpi->oxcf.content != VP9E_CONTENT_SCREEN);
      cpi->segmentation_map = cpi->seg_map_pred;
      cyclic_refresh_update_map(cpi, 1);
      cpi->segmentation_map = seg_map;
    }
#endif
  }
}
