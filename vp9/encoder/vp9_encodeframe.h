/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#ifndef VP9_ENCODER_VP9_ENCODEFRAME_H_
#define VP9_ENCODER_VP9_ENCODEFRAME_H_

#ifdef __cplusplus
extern "C" {
#endif

struct macroblock;
struct yv12_buffer_config;
struct VP9Common;
struct macroblockd;
struct VP9_COMP;

// Constants used in SOURCE_VAR_BASED_PARTITION
#define VAR_HIST_MAX_BG_VAR 1000
#define VAR_HIST_FACTOR 10
#define VAR_HIST_BINS (VAR_HIST_MAX_BG_VAR / VAR_HIST_FACTOR + 1)
#define VAR_HIST_LARGE_CUT_OFF 75
#define VAR_HIST_SMALL_CUT_OFF 45

void duplicate_mode_info_in_sb(struct VP9Common *cm, struct macroblockd *xd,
                               int mi_row, int mi_col,
                               BLOCK_SIZE bsize);

void set_offsets(struct VP9_COMP *cpi, const TileInfo *const tile,
                 struct macroblock *const x, int mi_row, int mi_col,
                 BLOCK_SIZE bsize);

int get_sb_index(struct VP9Common *const cm, int mi_row, int mi_col);

void vp9_setup_src_planes(struct macroblock *x,
                          const struct yv12_buffer_config *src,
                          int mi_row, int mi_col);

int vp9_encoding_thread_process(thread_context *const thread_ctxt, void* data2);

void vp9_encode_frame(struct VP9_COMP *cpi);
void set_vbp_thresholds(struct VP9_COMP *cpi, int64_t thresholds[], int q);
void vp9_set_variance_partition_thresholds(struct VP9_COMP *cpi, int q);
int choose_partitioning(struct VP9_COMP *cpi,
                        const TileInfo *const tile,
                        struct macroblock *x,
                        int mi_row, int mi_col);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VP9_ENCODER_VP9_ENCODEFRAME_H_
