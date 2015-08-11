/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VP9_ENCODER_VP9_ETHREAD_H_
#define VP9_ENCODER_VP9_ETHREAD_H_

/* Thread management macros */
#ifdef _WIN32
  /* Win32 */
  #include <windows.h>
  #define thread_sleep(nms) Sleep(nms)
#elif defined(__OS2__)
  /* OS/2 */
  #define INCL_DOS
  #include <os2.h>
  #define thread_sleep(nms) DosSleep(nms)
#else
  /* POSIX */
  #include <sched.h>
  #define thread_sleep(nms) sched_yield();
#endif

#if ARCH_X86 || ARCH_X86_64
  #include "vpx_ports/x86.h"
#else
  #define x86_pause_hint()
#endif

struct VP9_COMP;

typedef struct RD_COUNTS {
  vp9_coeff_count coef_counts[TX_SIZES][PLANE_TYPES];
  int64_t comp_pred_diff[REFERENCE_MODES];
  int64_t filter_diff[SWITCHABLE_FILTER_CONTEXTS];
} RD_COUNTS;

typedef struct ThreadData {
  MACROBLOCK mb;
  RD_COUNTS rd_counts;
  FRAME_COUNTS *counts;

  PICK_MODE_CONTEXT *leaf_tree;
  PC_TREE *pc_tree;
  PC_TREE *pc_root;
} ThreadData;

typedef struct thread_context {
  struct VP9_COMP *cpi;

  // thread specific mb context
  ThreadData td;

  // threads shall process rows of the video frame. Below params represent
  // the list of row id's the thread processes
  int mi_row_start, mi_row_end;
  int mi_row_step;

  // used by loop filter threads to determine if only y plane needs to be
  // filtered or all mb planes have to be filtered
  int y_only;
} thread_context;

void vp9_enc_sync_read(struct VP9_COMP *cpi, int sb_row, int sb_col);

void vp9_enc_sync_write(struct VP9_COMP *cpi, int sb_row);

void vp9_create_encoding_threads(struct VP9_COMP *cpi);

void vp9_accumulate_rd_opt(ThreadData *td, ThreadData *td_t);

void vp9_mb_copy(struct VP9_COMP *cpi, struct macroblock *x_dst,
                 struct macroblock *x_src);

#endif /* VP9_ETHREAD_H_ */
