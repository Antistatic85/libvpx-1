/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vp9/common/vp9_onyxc_int.h"

#if CONFIG_OPENCL
#include "vp9/common/opencl/vp9_opencl.h"
#endif

#if CONFIG_GPU_COMPUTE

int vp9_gpu_get_frame_buffer(void *cb_priv, size_t min_size,
                             vpx_codec_frame_buffer_t *fb) {
  gpu_cb_priv *const priv = (gpu_cb_priv *)cb_priv;
  VP9_COMMON *cm = priv->cm;

  if (min_size > (size_t)priv->ybf->buffer_alloc_sz) {
    vp9_gpu_free_frame_buffer(cm, priv->ybf);
    fb->data = cm->gpu.alloc_frame_buffers(cm, min_size,
                                           &priv->ybf->frame_buff);
    priv->ybf->buffer_alloc_sz = min_size;
    fb->size = min_size;
    fb->priv = NULL;

    memset(fb->data, 0, min_size);
  } else {
    fb->data = priv->ybf->buffer_alloc;
    fb->size = min_size;
    fb->priv = NULL;
  }

  return 0;
}

int vp9_gpu_free_frame_buffer(VP9_COMMON *cm, YV12_BUFFER_CONFIG *ybf) {
  if (ybf) {
    if (ybf->buffer_alloc_sz > 0) {
      cm->gpu.release_frame_buffers(cm, ybf->frame_buff);
    }
    /* buffer_alloc isn't accessed by most functions.  Rather y_buffer,
     * u_buffer and v_buffer point to buffer_alloc and are used.  Clear out
     * all of this so that a freed pointer isn't inadvertently used */
    memset(ybf, 0, sizeof(YV12_BUFFER_CONFIG));
  } else {
    return -1;
  }

  return 0;
}

void vp9_gpu_remove(VP9_COMMON *cm) {
  if (cm->gpu.remove)
    cm->gpu.remove(cm);
}

int vp9_gpu_init(VP9_COMMON *cm) {
#if CONFIG_OPENCL
  return vp9_opencl_init(cm);
#else
  return 1;
#endif
}

#endif
