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

#include "vp9/common/vp9_reconinter.h"

#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_encodeframe.h"

// synchronize encoder threads
void vp9_enc_sync_read(VP9_COMP *cpi, int sb_row, int sb_col) {
  const VP9_COMMON *const cm = &cpi->common;
  const volatile int *const top_sb_col = cpi->cur_sb_col + (sb_row - 1);

  // Check if the dependencies necessary to encode the current SB are
  // resolved. If the dependencies are resolved encode else do a busy wait.
  while (sb_row && !(sb_col & (cpi->sync_range - 1))) {
    // top right dependency
    int idx = sb_col + cpi->sync_range;

    idx = MIN(idx, (cm->sb_cols - 1));
    if (*top_sb_col >= idx)
      break;
    x86_pause_hint();
    thread_sleep(0);
  }
}

// synchronize encoder threads
void vp9_enc_sync_write(struct VP9_COMP *cpi, int sb_row) {
  int *const cur_sb_col = cpi->cur_sb_col + sb_row;

  // update the cur sb col
  (*cur_sb_col)++;
}

// Set up nsync by width.
// The optimal sync_range for different resolution and platform should be
// determined by testing. Currently, it is chosen to be a power-of-2 number.
static int get_sync_range(int width) {
  // TODO(ram-ittiam): nsync numbers have to be picked by testing
  if (width < 640)
    return 1;
  else if (width <= 1280)
    return 2;
  else if (width <= 4096)
    return 4;
  else
    return 8;
}

void vp9_create_encoding_threads(VP9_COMP *cpi) {
  VP9_COMMON * const cm = &cpi->common;
  const VPxWorkerInterface * const winterface = vpx_get_worker_interface();
  int i;

  CHECK_MEM_ERROR(cm, cpi->enc_thread_hndl,
                  vpx_malloc(sizeof(*cpi->enc_thread_hndl) * cpi->max_threads));
  CHECK_MEM_ERROR(cm, cpi->enc_thread_ctxt,
                  vpx_malloc(sizeof(*cpi->enc_thread_ctxt) * cpi->max_threads));
  for (i = 0; i < cpi->max_threads; ++i) {
    VPxWorker *const worker = &cpi->enc_thread_hndl[i];
    winterface->init(worker);
    CHECK_MEM_ERROR(cm, cpi->enc_thread_ctxt[i],
                    vpx_memalign(32, sizeof(thread_context)));

    // Set up pc_tree.
    cpi->enc_thread_ctxt[i]->td.leaf_tree = NULL;
    cpi->enc_thread_ctxt[i]->td.pc_tree = NULL;
    vp9_setup_pc_tree(cm, &cpi->enc_thread_ctxt[i]->td);

    // Allocate frame counters in thread data.
    CHECK_MEM_ERROR(cm, cpi->enc_thread_ctxt[i]->td.counts,
                    vpx_calloc(1, sizeof(*cpi->enc_thread_ctxt[i]->td.counts)));

    worker->data1 = cpi->enc_thread_ctxt[i];
    worker->data2 = NULL;
    if (i < cpi->max_threads - 1 && !winterface->reset(worker)) {
      vpx_internal_error(&cm->error, VPX_CODEC_ERROR,
                         "Tile decoder thread creation failed");
    }
  }
  // set row encoding hook
  for (i = 0; i < cpi->max_threads; ++i) {
    winterface->sync(&cpi->enc_thread_hndl[i]);
    cpi->enc_thread_hndl[i].hook = (VPxWorkerHook) vp9_encoding_thread_process;
  }
  CHECK_MEM_ERROR(cm, cpi->cur_sb_col,
                  vpx_malloc(sizeof(*cpi->cur_sb_col) * cm->sb_rows));
  // init cur sb col
  memset(cpi->cur_sb_col, -1, (sizeof(*cpi->cur_sb_col) * cm->sb_rows));
  // set up nsync (currently unused).
  cpi->sync_range = get_sync_range(cpi->oxcf.width);
}

void vp9_accumulate_rd_opt(ThreadData *td, ThreadData *td_t) {
  int i, j, k, l, m, n;

  for (i = 0; i < REFERENCE_MODES; i++)
    td->rd_counts.comp_pred_diff[i] += td_t->rd_counts.comp_pred_diff[i];

  for (i = 0; i < SWITCHABLE_FILTER_CONTEXTS; i++)
    td->rd_counts.filter_diff[i] += td_t->rd_counts.filter_diff[i];

  for (i = 0; i < TX_SIZES; i++)
    for (j = 0; j < PLANE_TYPES; j++)
      for (k = 0; k < REF_TYPES; k++)
        for (l = 0; l < COEF_BANDS; l++)
          for (m = 0; m < COEFF_CONTEXTS; m++)
            for (n = 0; n < ENTROPY_TOKENS; n++)
              td->rd_counts.coef_counts[i][j][k][l][m][n] +=
                  td_t->rd_counts.coef_counts[i][j][k][l][m][n];
}

void vp9_mb_copy(VP9_COMP *cpi, MACROBLOCK *x_dst, MACROBLOCK *x_src) {
  VP9_COMMON *cm = &cpi->common;
  MACROBLOCKD *const xd_dst = &x_dst->e_mbd;
  MACROBLOCKD *const xd_src = &x_src->e_mbd;
  int i;

  for (i = 0; i < MAX_MB_PLANE; ++i) {
    x_dst->plane[i] = x_src->plane[i];
    xd_dst->plane[i] = xd_src->plane[i];
  }
  xd_dst->mi_stride = xd_src->mi_stride;
  xd_dst->mi = xd_src->mi;
  xd_dst->block_refs[0] = xd_src->block_refs[0];
  xd_dst->block_refs[1] = xd_src->block_refs[1];
  xd_dst->cur_buf = xd_src->cur_buf;
#if CONFIG_VP9_HIGHBITDEPTH
  xd_dst->bd = xd_src->bd;
#endif
  xd_dst->lossless = xd_src->lossless;
  xd_dst->corrupted = 0;
  for (i = 0; i < MAX_MB_PLANE; i++) {
    xd_dst->above_context[i] = xd_src->above_context[i];
  }
  xd_dst->above_seg_context = xd_src->above_seg_context;

  x_dst->mbmi_ext_base = x_src->mbmi_ext_base;
  x_dst->skip_block = x_src->skip_block;
  x_dst->select_tx_size = x_src->select_tx_size;
  x_dst->skip_recode = x_src->skip_recode;
  x_dst->skip_optimize = x_src->skip_optimize;
  x_dst->q_index = x_src->q_index;

  x_dst->errorperbit = x_src->errorperbit;
  x_dst->sadperbit16 = x_src->sadperbit16;
  x_dst->sadperbit4 = x_src->sadperbit4;
  x_dst->rddiv = x_src->rddiv;
  x_dst->rdmult = x_src->rdmult;
  x_dst->mb_energy = x_src->mb_energy;

  for (i = 0; i < MV_JOINTS; i++) {
    x_dst->nmvjointcost[i] = x_src->nmvjointcost[i];
    x_dst->nmvjointsadcost[i] = x_src->nmvjointsadcost[i];
  }
  x_dst->nmvcost[0] = x_src->nmvcost[0];
  x_dst->nmvcost[1] = x_src->nmvcost[1];
  x_dst->nmvcost_hp[0] = x_src->nmvcost_hp[0];
  x_dst->nmvcost_hp[1] = x_src->nmvcost_hp[1];
  x_dst->mvcost = x_src->mvcost;
  x_dst->nmvsadcost[0] = x_src->nmvsadcost[0];
  x_dst->nmvsadcost[1] = x_src->nmvsadcost[1];
  x_dst->nmvsadcost_hp[0] = x_src->nmvsadcost_hp[0];
  x_dst->nmvsadcost_hp[1] = x_src->nmvsadcost_hp[1];
  x_dst->mvsadcost = x_src->mvsadcost;

  x_dst->min_partition_size = x_src->min_partition_size;
  x_dst->max_partition_size = x_src->max_partition_size;

  memcpy(x_dst->token_costs, x_src->token_costs,
         sizeof(x_src->token_costs));

  memcpy(x_dst->rd.threshes, cpi->rd.threshes, sizeof(cpi->rd.threshes));
  // freq scaling factors initialization has to happen only for video frame 1.
  // For all other frames, It self corrects itself while encoding.
  if (cm->current_video_frame == 0) {
    memcpy(x_dst->rd.thresh_freq_fact, cpi->rd.thresh_freq_fact,
           sizeof(cpi->rd.thresh_freq_fact));
    memcpy(x_dst->rd.mode_map, cpi->rd.mode_map,
           sizeof(cpi->rd.mode_map));
  }
  x_dst->rd.RDMULT = cpi->rd.RDMULT;
  x_dst->rd.RDDIV = cpi->rd.RDDIV;

  x_dst->optimize = x_src->optimize;
  x_dst->quant_fp = x_src->quant_fp;
  vp9_zero(x_dst->skip_txfm);
  vp9_zero(x_dst->bsse);

  x_dst->fwd_txm4x4 = x_src->fwd_txm4x4;
  x_dst->itxm_add = x_src->itxm_add;
#if CONFIG_VP9_HIGHBITDEPTH
  x_dst->highbd_itxm_add = x_src->highbd_itxm_add;
#endif
}
