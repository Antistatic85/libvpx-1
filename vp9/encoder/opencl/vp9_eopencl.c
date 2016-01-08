/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#include "vp9/common/opencl/vp9_opencl.h"
#include "vp9/encoder/opencl/vp9_eopencl.h"

#define OPENCL_DEVELOPER_MODE 1
#define BUILD_OPTION_LENGTH 128
#if ARCH_ARM
#define PREFIX_PATH "./"
#else
#define PREFIX_PATH "../../vp9/encoder/opencl/"
#endif

static const int pixel_rows_per_workitem_log2_pro_me = 4;

static const int pixel_rows_per_workitem_log2_zeromv = 4;

static const int pixel_rows_per_workitem_log2_inter_pred[GPU_BLOCK_SIZES]
                                                         = {3, 3};

static const int pixel_rows_per_workitem_log2_full_pixel[GPU_BLOCK_SIZES]
                                                                = {4, 4};

static const int pixel_rows_per_workitem_log2_sub_pixel[GPU_BLOCK_SIZES]
                                                                = {4, 5};

static char *read_src(const char *src_file_name) {
  FILE *fp;
  int32_t err;
  int32_t size;
  char *src;

  fp = fopen(src_file_name, "r");
  if (fp == NULL)
    return NULL;

  err = fseek(fp, 0, SEEK_END);
  if (err != 0)
    return NULL;

  size = ftell(fp);
  if (size < 0)
    return NULL;

  err = fseek(fp, 0, SEEK_SET);
  if (err != 0)
    return NULL;

  src = (char *)vpx_malloc(size + 1);
  if (src == NULL)
    return NULL;

  err = fread(src, 1, size, fp);
  if (err != size) {
    vpx_free(src);
    return NULL;
  }

  src[size] = '\0';

  return src;
}

#if OPENCL_PROFILING
static cl_ulong get_event_time_elapsed(cl_event event) {
  cl_ulong startTime, endTime;
  cl_int status = 0;

  status  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                   sizeof(cl_ulong), &startTime, NULL);
  status |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                    sizeof(cl_ulong), &endTime, NULL);
  assert(status == CL_SUCCESS);
  return (endTime - startTime);
}
#endif

static void vp9_eopencl_set_static_kernel_args(VP9_COMP *cpi) {
  VP9_COMMON *cm = &cpi->common;
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  cl_mem *gpu_ip = &eopencl->gpu_input.opencl_mem;
  cl_mem *gpu_op_me = &eopencl->gpu_output_me;
  cl_mem *gpu_scratch = &eopencl->gpu_scratch;
  struct lookahead_ctx *ctx = cpi->lookahead;
  YV12_BUFFER_CONFIG *source_yuv = &ctx->buf[0].img;
  cl_int y_stride = source_yuv->y_stride;
  // TODO(Karthick) : There is huge assumption being made here. We assume all
  // the YUV buffers will be aligned to the same byte boundary. In case of Intel
  // Graphics and ARM Mali, we have verified that all the buffers are aligned to
  // 4096 bytes(page boundary). So no problems so far..
  // But an ideal fix to remove this assumption could affect performance.
  cl_int padding_offset = source_yuv->y_buffer - source_yuv->buffer_alloc;
  int64_t yplane_size = source_yuv->u_buffer - source_yuv->y_buffer;
  int64_t uvplane_size = source_yuv->v_buffer - source_yuv->u_buffer;
  cl_int mi_rows = cm->mi_rows;
  cl_int mi_cols = cm->mi_cols;
  cl_int op_stride;
  cl_int status;
  GPU_BLOCK_SIZE gpu_bsize;

  // PRO ME KERNELS

  // project Source SB Cols of each SB on to a horizontal plane
  status = clSetKernelArg(eopencl->col_row_projection, 2, sizeof(cl_int),
                          &y_stride);
  status |= clSetKernelArg(eopencl->col_row_projection, 3, sizeof(cl_mem),
                           &eopencl->src_1d_set[0]);
  status |= clSetKernelArg(eopencl->col_row_projection, 4, sizeof(cl_mem),
                           &eopencl->pred_1d_set[0]);
  status |= clSetKernelArg(eopencl->col_row_projection, 5, sizeof(cl_mem),
                           &eopencl->src_1d_set[1]);
  status |= clSetKernelArg(eopencl->col_row_projection, 6, sizeof(cl_mem),
                           &eopencl->pred_1d_set[1]);
  status |= clSetKernelArg(eopencl->col_row_projection, 7, sizeof(cl_int),
                           &padding_offset);
  assert(status == CL_SUCCESS);

  // vector match x, y
  status = clSetKernelArg(eopencl->vector_match, 0, sizeof(cl_mem),
                          &eopencl->src_1d_set[0]);
  status |= clSetKernelArg(eopencl->vector_match, 1, sizeof(cl_mem),
                           &eopencl->pred_1d_set[0]);
  status |= clSetKernelArg(eopencl->vector_match, 2, sizeof(cl_mem),
                           &eopencl->src_1d_set[1]);
  status |= clSetKernelArg(eopencl->vector_match, 3, sizeof(cl_mem),
                           &eopencl->pred_1d_set[1]);
  assert(status == CL_SUCCESS);

  // Pro Motion Estimation
  status = clSetKernelArg(eopencl->pro_motion_estimation, 2, sizeof(cl_int),
                          &y_stride);
  status |= clSetKernelArg(eopencl->pro_motion_estimation, 4, sizeof(cl_int),
                           &padding_offset);
  assert(status == CL_SUCCESS);

  // color sensitivity
  status = clSetKernelArg(eopencl->color_sensitivity, 2, sizeof(cl_int),
                          &y_stride);
  status |= clSetKernelArg(eopencl->color_sensitivity, 4, sizeof(cl_long),
                           &yplane_size);
  status |= clSetKernelArg(eopencl->color_sensitivity, 5, sizeof(cl_long),
                           &uvplane_size);
  status |= clSetKernelArg(eopencl->color_sensitivity, 6, sizeof(cl_int),
                           &padding_offset);
  assert(status == CL_SUCCESS);

  // choose partitions
  status = clSetKernelArg(eopencl->choose_partitions, 2, sizeof(cl_int),
                          &y_stride);
  status |= clSetKernelArg(eopencl->choose_partitions, 5, sizeof(cl_mem),
                           gpu_ip);
  op_stride = cm->sb_cols * num_mxn_blocks_high_lookup[BLOCK_32X32];
  status |= clSetKernelArg(eopencl->choose_partitions, 6, sizeof(cl_int),
                           &op_stride);
  status |= clSetKernelArg(eopencl->choose_partitions, 7, sizeof(cl_int),
                           &padding_offset);
  status |= clSetKernelArg(eopencl->choose_partitions, 8, sizeof(cl_int),
                           &mi_rows);
  status |= clSetKernelArg(eopencl->choose_partitions, 9, sizeof(cl_int),
                           &mi_cols);
  assert(status == CL_SUCCESS);

  // ME KERNELS
  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
    status  = clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 2,
                             sizeof(cl_int), &y_stride);
    status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 3,
                             sizeof(cl_mem), gpu_ip);
    status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 4,
                             sizeof(cl_mem), gpu_op_me);
    status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 6,
                             sizeof(cl_long), &yplane_size);
    status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 7,
                             sizeof(cl_long), &uvplane_size);
    status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 8,
                             sizeof(cl_int), &padding_offset);
    assert(status == CL_SUCCESS);

    status  = clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 2,
                             sizeof(cl_int), &y_stride);
    status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 3,
                             sizeof(cl_mem), gpu_ip);
    status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 4,
                             sizeof(cl_mem), gpu_op_me);
    status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 6,
                             sizeof(cl_int), &mi_rows);
    status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 7,
                             sizeof(cl_int), &mi_cols);
    status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 8,
                             sizeof(cl_int), &padding_offset);
    assert(status == CL_SUCCESS);

    status |= clSetKernelArg(eopencl->hpel_search[gpu_bsize], 2,
                             sizeof(cl_int), &y_stride);
    status |= clSetKernelArg(eopencl->hpel_search[gpu_bsize], 3,
                             sizeof(cl_mem), gpu_ip);
    status |= clSetKernelArg(eopencl->hpel_search[gpu_bsize], 4,
                             sizeof(cl_mem), gpu_op_me);
    status |= clSetKernelArg(eopencl->hpel_search[gpu_bsize], 5,
                             sizeof(cl_mem), gpu_scratch);
    status |= clSetKernelArg(eopencl->hpel_search[gpu_bsize], 6,
                             sizeof(cl_int), &padding_offset);
    assert(status == CL_SUCCESS);

    status |= clSetKernelArg(eopencl->qpel_search[gpu_bsize], 2,
                             sizeof(cl_int), &y_stride);
    status |= clSetKernelArg(eopencl->qpel_search[gpu_bsize], 3,
                             sizeof(cl_mem), gpu_ip);
    status |= clSetKernelArg(eopencl->qpel_search[gpu_bsize], 4,
                             sizeof(cl_mem), gpu_op_me);
    status |= clSetKernelArg(eopencl->qpel_search[gpu_bsize], 5,
                             sizeof(cl_mem), gpu_scratch);
    status |= clSetKernelArg(eopencl->qpel_search[gpu_bsize], 6,
                             sizeof(cl_int), &padding_offset);
    assert(status == CL_SUCCESS);

    status |= clSetKernelArg(eopencl->inter_prediction_and_sse[gpu_bsize], 2,
                             sizeof(cl_int), &y_stride);
    status |= clSetKernelArg(eopencl->inter_prediction_and_sse[gpu_bsize], 3,
                             sizeof(cl_mem), gpu_ip);
    status |= clSetKernelArg(eopencl->inter_prediction_and_sse[gpu_bsize], 4,
                             sizeof(cl_mem), gpu_op_me);
    status |= clSetKernelArg(eopencl->inter_prediction_and_sse[gpu_bsize], 5,
                             sizeof(cl_mem), gpu_scratch);
    status |= clSetKernelArg(eopencl->inter_prediction_and_sse[gpu_bsize], 6,
                             sizeof(cl_int), &padding_offset);
    assert(status == CL_SUCCESS);

    status |= clSetKernelArg(eopencl->rd_calculation_newmv[gpu_bsize], 0,
                             sizeof(cl_mem), gpu_ip);
    status |= clSetKernelArg(eopencl->rd_calculation_newmv[gpu_bsize], 1,
                             sizeof(cl_mem), gpu_op_me);
    status |= clSetKernelArg(eopencl->rd_calculation_newmv[gpu_bsize], 3,
                             sizeof(cl_mem), gpu_scratch);
    assert(status == CL_SUCCESS);
  }
}

static void vp9_eopencl_set_dynamic_kernel_args_pro_me(VP9_COMP *cpi) {
  VP9_COMMON *cm = &cpi->common;
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  YV12_BUFFER_CONFIG *img_src;
  YV12_BUFFER_CONFIG *last_img_src = cpi->Source;
  struct lookahead_entry *next_source = NULL;
  cl_mem *gpu_op_pro_me =
      &eopencl->gpu_output_pro_me[(cm->current_video_frame + 1) & 1];
  cl_int status;

  next_source = vp9_lookahead_peek(cpi->lookahead, 0);
  img_src = &next_source->img;

  // project Source SB Cols of each SB on to a horizontal plane
  status = clSetKernelArg(eopencl->col_row_projection, 0, sizeof(cl_mem),
                          &img_src->gpu_mem);
  status |= clSetKernelArg(eopencl->col_row_projection, 1, sizeof(cl_mem),
                           &last_img_src->gpu_mem);
  assert(status == CL_SUCCESS);

  // vector match x, y
  status = clSetKernelArg(eopencl->vector_match, 4, sizeof(cl_mem),
                          gpu_op_pro_me);
  assert(status == CL_SUCCESS);

  // Pro Motion Estimation
  status = clSetKernelArg(eopencl->pro_motion_estimation, 0, sizeof(cl_mem),
                          &img_src->gpu_mem);
  status |= clSetKernelArg(eopencl->pro_motion_estimation, 1, sizeof(cl_mem),
                           &last_img_src->gpu_mem);
  status |= clSetKernelArg(eopencl->pro_motion_estimation, 3, sizeof(cl_mem),
                           gpu_op_pro_me);
  assert(status == CL_SUCCESS);

  // color sensitivity
  status = clSetKernelArg(eopencl->color_sensitivity, 0, sizeof(cl_mem),
                          &img_src->gpu_mem);
  status |= clSetKernelArg(eopencl->color_sensitivity, 1, sizeof(cl_mem),
                           &last_img_src->gpu_mem);
  status |= clSetKernelArg(eopencl->color_sensitivity, 3, sizeof(cl_mem),
                           gpu_op_pro_me);
  assert(status == CL_SUCCESS);
}

static void vp9_eopencl_set_dynamic_kernel_args_me(VP9_COMP *cpi, int async) {
  VP9_COMMON *cm = &cpi->common;
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  YV12_BUFFER_CONFIG *img_src = cpi->Source;
  YV12_BUFFER_CONFIG *frm_ref = get_ref_frame_buffer(cpi, LAST_FRAME);
  cl_mem *gpu_op_pro_me =
      &eopencl->gpu_output_pro_me[cm->current_video_frame & 1];
  cl_mem *rdopt_parameters =
      &eopencl->rdopt_parameters[cm->current_video_frame & 1].opencl_mem;
  GPU_BLOCK_SIZE gpu_bsize;
  cl_int status;

  if (async) {
    struct lookahead_entry *next_source = NULL;

    next_source = vp9_lookahead_peek(cpi->lookahead, 0);
    img_src = &next_source->img;
    frm_ref = get_frame_new_buffer(cm);
    gpu_op_pro_me =
          &eopencl->gpu_output_pro_me[(cm->current_video_frame + 1) & 1];
    rdopt_parameters =
        &eopencl->rdopt_parameters[(cm->current_video_frame + 1) & 1].opencl_mem;
  }

  // choose partitions
  status = clSetKernelArg(eopencl->choose_partitions, 0, sizeof(cl_mem),
                          &img_src->gpu_mem);
  status |= clSetKernelArg(eopencl->choose_partitions, 1, sizeof(cl_mem),
                           &frm_ref->gpu_mem);
  status |= clSetKernelArg(eopencl->choose_partitions, 3, sizeof(cl_mem),
                           gpu_op_pro_me);
  status |= clSetKernelArg(eopencl->choose_partitions, 4, sizeof(cl_mem),
                           rdopt_parameters);
  assert(status == CL_SUCCESS);

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {

    status = clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 0,
                            sizeof(cl_mem), &frm_ref->gpu_mem);
    status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 1,
                             sizeof(cl_mem), &img_src->gpu_mem);
    status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 5,
                             sizeof(cl_mem), rdopt_parameters);
    assert(status == CL_SUCCESS);

    status = clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 0,
                            sizeof(cl_mem), &frm_ref->gpu_mem);
    status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 1,
                             sizeof(cl_mem), &img_src->gpu_mem);
    status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 5,
                             sizeof(cl_mem), rdopt_parameters);
    assert(status == CL_SUCCESS);

    status = clSetKernelArg(eopencl->hpel_search[gpu_bsize], 0,
                            sizeof(cl_mem), &frm_ref->gpu_mem);
    status |= clSetKernelArg(eopencl->hpel_search[gpu_bsize], 1,
                             sizeof(cl_mem), &img_src->gpu_mem);
    assert(status == CL_SUCCESS);

    status = clSetKernelArg(eopencl->qpel_search[gpu_bsize], 0,
                            sizeof(cl_mem), &frm_ref->gpu_mem);
    status |= clSetKernelArg(eopencl->qpel_search[gpu_bsize], 1,
                             sizeof(cl_mem), &img_src->gpu_mem);
    assert(status == CL_SUCCESS);
  }

  // Lowest GPU Block size selected for the merged kernels
  gpu_bsize = GPU_BLOCK_32X32;

  status = clSetKernelArg(eopencl->inter_prediction_and_sse[gpu_bsize], 0,
                          sizeof(cl_mem), &frm_ref->gpu_mem);
  status |= clSetKernelArg(eopencl->inter_prediction_and_sse[gpu_bsize], 1,
                           sizeof(cl_mem), &img_src->gpu_mem);
  assert(status == CL_SUCCESS);

  status |= clSetKernelArg(eopencl->rd_calculation_newmv[gpu_bsize], 2,
                           sizeof(cl_mem), rdopt_parameters);
  assert(status == CL_SUCCESS);
}

static void vp9_eopencl_alloc_buffers(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  VP9_EGPU *gpu = &cpi->egpu;
  VP9_EOPENCL *eopencl = gpu->compute_framework;
  VP9_OPENCL *const opencl = eopencl->opencl;
  cl_int status;
  opencl_buffer *gpuinput_b_args = &eopencl->gpu_input;
  // Allocating for the lowest block size (worst case memory requirement)
  const BLOCK_SIZE bsize = vp9_actual_block_size_lookup[0];
  int blocks_in_col, blocks_in_row;
  int alloc_size, projection_buf_size;
  int subframe_idx;
  int i;

  blocks_in_col = cm->sb_rows;
  blocks_in_row = cm->sb_cols;
  alloc_size = blocks_in_row * blocks_in_col;

  // alloc buffer for 1D src and pred buffers for pro motion estimation
  projection_buf_size = (blocks_in_row + 1) * 64 * (blocks_in_col + 1);
  eopencl->pred_1d_set[0] = clCreateBuffer(
      opencl->context, CL_MEM_READ_WRITE,
      projection_buf_size * sizeof(int16_t), NULL, &status);
  if (status != CL_SUCCESS)
    goto fail;

  eopencl->src_1d_set[0] = clCreateBuffer(
      opencl->context, CL_MEM_READ_WRITE,
      projection_buf_size * sizeof(int16_t), NULL, &status);
  if (status != CL_SUCCESS)
    goto fail;

  eopencl->pred_1d_set[1] = clCreateBuffer(
      opencl->context, CL_MEM_READ_WRITE,
      projection_buf_size * sizeof(int16_t), NULL, &status);
  if (status != CL_SUCCESS)
    goto fail;

  eopencl->src_1d_set[1] = clCreateBuffer(
      opencl->context, CL_MEM_READ_WRITE,
      projection_buf_size * sizeof(int16_t), NULL, &status);
  if (status != CL_SUCCESS)
    goto fail;

  // alloc output buffers for pro motion estimation
  for (i = 0; i < NUM_PING_PONG_BUFFERS; i++) {
    eopencl->gpu_output_pro_me[i] = clCreateBuffer(
        opencl->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        alloc_size * sizeof(GPU_OUTPUT_PRO_ME), NULL, &status);
    if (status != CL_SUCCESS)
      goto fail;

    // create output sub buffers for pro motion estimation
    for (subframe_idx = 0; subframe_idx < MAX_SUB_FRAMES; ++subframe_idx) {
      cl_buffer_region sf_region;
      SubFrameInfo subframe;
      int block_row_offset;
      int block_rows_sf;
      int alloc_size_sf;

      vp9_subframe_init(&subframe, cm, subframe_idx);

      block_row_offset = subframe.mi_row_start >> MI_BLOCK_SIZE_LOG2;
      block_rows_sf = (mi_cols_aligned_to_sb(subframe.mi_row_end) -
          subframe.mi_row_start) >> MI_BLOCK_SIZE_LOG2;

      alloc_size_sf = blocks_in_row * block_rows_sf;

      sf_region.origin =
          block_row_offset * blocks_in_row * sizeof(GPU_OUTPUT_PRO_ME);
      sf_region.size = alloc_size_sf * sizeof(GPU_OUTPUT_PRO_ME);
      eopencl->gpu_output_pro_me_sub_buf[i][subframe_idx].size = sf_region.size;
      eopencl->gpu_output_pro_me_sub_buf[i][subframe_idx].opencl_mem =
          clCreateSubBuffer(eopencl->gpu_output_pro_me[i],
                            CL_MEM_READ_WRITE,
                            CL_BUFFER_CREATE_TYPE_REGION,
                            &sf_region, &status);
      if (status != CL_SUCCESS)
        goto fail;
    }

    // alloc buffer for gpu rd params
    eopencl->rdopt_parameters[i].size = sizeof(GPU_RD_PARAMETERS);
    eopencl->rdopt_parameters[i].opencl_mem = clCreateBuffer(
        opencl->context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        eopencl->rdopt_parameters[i].size, NULL, &status);
    if (status != CL_SUCCESS)
      goto fail;
  }

  blocks_in_col = (cm->sb_rows * num_mxn_blocks_high_lookup[bsize]);
  blocks_in_row = (cm->sb_cols * num_mxn_blocks_wide_lookup[bsize]);
  alloc_size = blocks_in_row * blocks_in_col;

  // alloc buffer for gpu input
  gpuinput_b_args->size = alloc_size * sizeof(GPU_INPUT);
  gpuinput_b_args->opencl_mem = clCreateBuffer(
      opencl->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
      gpuinput_b_args->size, NULL, &status);
  if (status != CL_SUCCESS)
    goto fail;

  // alloc buffer for gpu output
  eopencl->gpu_output_me = clCreateBuffer(
      opencl->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
      alloc_size * sizeof(GPU_OUTPUT_ME), NULL, &status);
  if (status != CL_SUCCESS)
    goto fail;

  // alloc space of rd calc tmp buffers
  eopencl->gpu_scratch = clCreateBuffer(
      opencl->context,  CL_MEM_READ_WRITE,
      alloc_size * sizeof(GPU_SCRATCH), NULL, &status);
  if (status != CL_SUCCESS)
    goto fail;

  // create output sub buffers
  for (subframe_idx = 0; subframe_idx < MAX_SUB_FRAMES; ++subframe_idx) {
    cl_buffer_region sf_region;
    SubFrameInfo subframe;
    int block_row_offset;
    int block_rows_sf;
    int alloc_size_sf;

    vp9_subframe_init(&subframe, cm, subframe_idx);

    block_row_offset = subframe.mi_row_start >> mi_height_log2(bsize);
    block_rows_sf = (mi_cols_aligned_to_sb(subframe.mi_row_end) -
        subframe.mi_row_start) >> mi_height_log2(bsize);

    alloc_size_sf = blocks_in_row * block_rows_sf;

    sf_region.origin = block_row_offset * blocks_in_row * sizeof(GPU_OUTPUT_ME);
    sf_region.size = alloc_size_sf * sizeof(GPU_OUTPUT_ME);
    eopencl->gpu_output_me_sub_buf[subframe_idx].size = sf_region.size;
    eopencl->gpu_output_me_sub_buf[subframe_idx].opencl_mem =
        clCreateSubBuffer(eopencl->gpu_output_me,
                          CL_MEM_READ_WRITE,
                          CL_BUFFER_CREATE_TYPE_REGION,
                          &sf_region, &status);
    if (status != CL_SUCCESS)
      goto fail;
  }

  vp9_eopencl_set_static_kernel_args(cpi);

  return;

fail:
  // TODO(karthick-ittiam): The error set below is ignored by the encoder. This
  // error needs to be handled appropriately. Adding assert as a temporary fix.
  vpx_internal_error(&cm->error, VPX_CODEC_MEM_ERROR,
                     "Failed to allocate OpenCL buffers");
  assert(0);
}

static void vp9_eopencl_free_buffers(VP9_COMP *cpi) {
  VP9_EOPENCL *eopencl = cpi->egpu.compute_framework;
  VP9_OPENCL *const opencl = eopencl->opencl;
  cl_int status;
  int subframe_id;
  int i;

  for (i = 0; i < 2; i++) {
    status = clReleaseMemObject(eopencl->pred_1d_set[i]);
    if (status != CL_SUCCESS)
      goto fail;
    status = clReleaseMemObject(eopencl->src_1d_set[i]);
    if (status != CL_SUCCESS)
      goto fail;
  }

  for (i = 0; i < NUM_PING_PONG_BUFFERS; i++) {
    for (subframe_id = 0; subframe_id < MAX_SUB_FRAMES; ++subframe_id) {
      opencl_buffer *gpu_output_pro_me_sub_buffer =
          &eopencl->gpu_output_pro_me_sub_buf[i][subframe_id];

      if (vp9_opencl_unmap_buffer(opencl, gpu_output_pro_me_sub_buffer, CL_TRUE))
        goto fail;

      status = clReleaseMemObject(gpu_output_pro_me_sub_buffer->opencl_mem);
      if (status != CL_SUCCESS)
        goto fail;
    }
    status = clReleaseMemObject(eopencl->gpu_output_pro_me[i]);
    if (status != CL_SUCCESS)
      goto fail;

    if (vp9_opencl_unmap_buffer(opencl, &eopencl->rdopt_parameters[i], CL_TRUE)) {
      goto fail;
    }
    status = clReleaseMemObject(eopencl->rdopt_parameters[i].opencl_mem);
    if (status != CL_SUCCESS)
      goto fail;
  }

  for (subframe_id = 0; subframe_id < MAX_SUB_FRAMES; ++subframe_id) {
    opencl_buffer *gpu_output_me_sub_buffer =
        &eopencl->gpu_output_me_sub_buf[subframe_id];

    if (vp9_opencl_unmap_buffer(opencl, gpu_output_me_sub_buffer, CL_TRUE)) {
      goto fail;
    }
    status = clReleaseMemObject(
        eopencl->gpu_output_me_sub_buf[subframe_id].opencl_mem);
    if (status != CL_SUCCESS)
      goto fail;
  }

  if (vp9_opencl_unmap_buffer(opencl, &eopencl->gpu_input, CL_TRUE)) {
    goto fail;
  }
  status = clReleaseMemObject(eopencl->gpu_input.opencl_mem);
  if (status != CL_SUCCESS)
    goto fail;

  status = clReleaseMemObject(eopencl->gpu_output_me);
  if (status != CL_SUCCESS)
    goto fail;

  status = clReleaseMemObject(eopencl->gpu_scratch);
  if (status != CL_SUCCESS)
    goto fail;

  return;

fail:
  // TODO(karthick-ittiam): The error set below is ignored by the encoder. This
  // error needs to be handled appropriately. Adding assert as a temporary fix.
  vpx_internal_error(&cpi->common.error, VPX_CODEC_MEM_ERROR,
                     "Failed to release OpenCL metadata buffers");
  assert(0);
}

static void vp9_eopencl_acquire_rd_param_buffer(VP9_COMP *cpi,
                                                void **host_ptr,
                                                int index) {
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  opencl_buffer *rdopt_parameters = &eopencl->rdopt_parameters[index];

  if (!vp9_opencl_map_buffer(eopencl->opencl, rdopt_parameters, CL_MAP_WRITE)) {
    *host_ptr = rdopt_parameters->mapped_pointer;
    return;
  }

  *host_ptr = NULL;
  assert(0);
}

static void vp9_eopencl_acquire_input_buffer(VP9_COMP *cpi, void **host_ptr) {
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  opencl_buffer *gpu_input = &eopencl->gpu_input;

  if (!vp9_opencl_map_buffer(eopencl->opencl, gpu_input, CL_MAP_WRITE)) {
    *host_ptr = gpu_input->mapped_pointer;
    return;
  }

  *host_ptr = NULL;
  assert(0);
}

static void vp9_eopencl_acquire_output_me_buffer(VP9_COMP *cpi, void **host_ptr,
                                                 int sub_frame_idx) {
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  opencl_buffer *gpu_output_me_sub_buffer =
      &eopencl->gpu_output_me_sub_buf[sub_frame_idx];

  if (!vp9_opencl_map_buffer(eopencl->opencl, gpu_output_me_sub_buffer,
                             CL_MAP_READ)) {
    *host_ptr = gpu_output_me_sub_buffer->mapped_pointer;
    return;
  }

  *host_ptr = NULL;
  assert(0);
}

static void vp9_eopencl_acquire_output_pro_me_buffer(VP9_COMP *cpi,
                                                     void **host_ptr,
                                                     int sub_frame_idx) {
  VP9_COMMON *const cm = &cpi->common;
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  opencl_buffer *gpu_output_pro_me_sub_buffer =
      &eopencl->gpu_output_pro_me_sub_buf[cm->current_video_frame & 1]
                                         [sub_frame_idx];

  if (!vp9_opencl_map_buffer(eopencl->opencl, gpu_output_pro_me_sub_buffer,
                             CL_MAP_READ)) {
    *host_ptr = gpu_output_pro_me_sub_buffer->mapped_pointer;
    return;
  }

  *host_ptr = NULL;
  assert(0);
}

static void vp9_eopencl_enc_sync_read(VP9_COMP *cpi, cl_int event_id,
                                      cl_int offset) {
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  cl_int status;

  assert(event_id < MAX_SUB_FRAMES);
  status = clWaitForEvents(1, &eopencl->event[offset + event_id]);
  if (status != CL_SUCCESS)
    vpx_internal_error(&cpi->common.error, VPX_CODEC_ERROR,
                       "Wait for event failed");
}

static void vp9_eopencl_execute_prologue(VP9_COMP *cpi) {
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  VP9_OPENCL *const opencl = eopencl->opencl;
  VP9_COMMON *const cm = &cpi->common;
  YV12_BUFFER_CONFIG *img_src;
  YV12_BUFFER_CONFIG *last_img_src = cpi->Source;
  opencl_buffer *gpu_output_pro_me_sub_buffer =
      &eopencl->gpu_output_pro_me_sub_buf[(cm->current_video_frame + 1) & 1][0];
  struct lookahead_entry *next_source = NULL;
  int blocks_in_col, blocks_in_row;
  FRAME_TYPE frame_type = cm->frame_type;
  size_t local_size[2];
  size_t global_size[2];
  size_t global_offset[2];
  cl_int status = CL_SUCCESS;
#if OPENCL_PROFILING
  cl_event event[NUM_KERNELS_PRO_ME - 1];
#endif
  cl_event *event_ptr[NUM_KERNELS_PRO_ME - 1];
  int i;

  for (i = 0; i < NUM_KERNELS_PRO_ME - 1; i++) {
#if OPENCL_PROFILING
    event_ptr[i] = &event[i];
#else
    event_ptr[i] = NULL;
#endif
  }

  (void)status;

  blocks_in_row = cm->sb_cols;
  blocks_in_col = cm->sb_rows;

  vp9_eopencl_set_dynamic_kernel_args_pro_me(cpi);

  next_source = vp9_lookahead_peek(cpi->lookahead, 0);
  img_src = &next_source->img;

  // before launching pro motion estimation kernels make sure the
  // source and reference buffers are available for GPU

  // release source buffer to GPU
  if (img_src->buffer_alloc_dup != NULL) {
    status = clEnqueueUnmapMemObject(opencl->cmd_queue, img_src->gpu_mem_sub,
                                     img_src->buffer_alloc_dup, 0, NULL, NULL);
    img_src->buffer_alloc_dup = NULL;
    assert(status == CL_SUCCESS);
  }

  // release last source buffer to GPU
  if (last_img_src->buffer_alloc_dup != NULL && frame_type == KEY_FRAME) {
    status = clEnqueueUnmapMemObject(opencl->cmd_queue, last_img_src->gpu_mem_sub,
                                     last_img_src->buffer_alloc_dup, 0, NULL, NULL);
    last_img_src->buffer_alloc_dup = NULL;
    assert(status == CL_SUCCESS);
  }

  // before launching pro motion estimation kernels unmap the output buffers
  // release pro me gpu output buffer
  for (i = 0; i < MAX_SUB_FRAMES; i++) {
    if (vp9_opencl_unmap_buffer(opencl, &gpu_output_pro_me_sub_buffer[i],
                                CL_FALSE)) {
      assert(0);
    }
  }

  // project Source/Reference SB Cols of each SB on to a horizontal plane
  local_size[0] = 64;
  local_size[1] = 1;

  global_size[0] = (blocks_in_row + 1) * local_size[0];
  global_size[1] = (blocks_in_col + 1) * local_size[1];

  global_offset[0] = 0;
  global_offset[1] = 0;

  status = clEnqueueNDRangeKernel(opencl->cmd_queue,
                                  eopencl->col_row_projection,
                                  2, global_offset, global_size, local_size,
                                  0, NULL, event_ptr[0]);
  assert(status == CL_SUCCESS);

  // vector match x, y
  local_size[0] = 8;
  local_size[1] = 1;

  global_size[0] = (blocks_in_row * local_size[0]);
  global_size[1] = (blocks_in_col * local_size[1]);

  global_offset[0] = 0;
  global_offset[1] = 0;

  status = clEnqueueNDRangeKernel(opencl->cmd_queue,
                                  eopencl->vector_match,
                                  2, global_offset, global_size, local_size,
                                  0, NULL, event_ptr[1]);
  assert(status == CL_SUCCESS);

  // Pro Motion Estimation
  local_size[0] = 8 * 4;
  local_size[1] = 64 >> pixel_rows_per_workitem_log2_pro_me;

  global_size[0] = (blocks_in_row * local_size[0]);
  global_size[1] = (blocks_in_col * local_size[1]);

  global_offset[0] = 0;
  global_offset[1] = 0;

  status = clEnqueueNDRangeKernel(opencl->cmd_queue,
                                  eopencl->pro_motion_estimation,
                                  2, global_offset, global_size, local_size,
                                  0, NULL, event_ptr[2]);
  assert(status == CL_SUCCESS);

  // Color Sensitivity
  local_size[0] = 4;
  local_size[1] = 32 >> pixel_rows_per_workitem_log2_pro_me;

  global_size[0] = (blocks_in_row * local_size[0]);
  global_size[1] = (blocks_in_col * local_size[1]);

  global_offset[0] = 0;
  global_offset[1] = 0;

  status = clEnqueueNDRangeKernel(opencl->cmd_queue,
                                  eopencl->color_sensitivity,
                                  2, global_offset, global_size, local_size,
                                  0, NULL, event_ptr[3]);
  assert(status == CL_SUCCESS);

#if OPENCL_PROFILING
  for (i = 0; i < NUM_KERNELS_PRO_ME - 1; i++) {
    cl_ulong time_elapsed;
    status = clWaitForEvents(1, event_ptr[i]);
    assert(status == CL_SUCCESS);
    time_elapsed = get_event_time_elapsed(*event_ptr[i]);
    eopencl->total_time_taken_pro_me[i] += time_elapsed / 1000;
    status = clReleaseEvent(*event_ptr[i]);
    assert(status == CL_SUCCESS);
  }
#endif

  // acquire current & last source buffer
  if (img_src->buffer_alloc_dup == NULL) {
    img_src->buffer_alloc_dup =
        clEnqueueMapBuffer(opencl->cmd_queue, img_src->gpu_mem_sub, CL_FALSE,
                           CL_MAP_READ | CL_MAP_WRITE, 0,
                           img_src->buffer_alloc_sz, 0, NULL, NULL, &status);
    assert(status == CL_SUCCESS);
  }

  if (last_img_src->buffer_alloc_dup == NULL) {
    last_img_src->buffer_alloc_dup =
        clEnqueueMapBuffer(opencl->cmd_queue, last_img_src->gpu_mem_sub, CL_FALSE,
                           CL_MAP_READ | CL_MAP_WRITE, 0,
                           last_img_src->buffer_alloc_sz, 0, NULL, NULL, &status);
    assert(status == CL_SUCCESS);
  }

  return;
}

static void vp9_eopencl_execute(VP9_COMP *cpi, int sub_frame_idx, int async) {
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  VP9_OPENCL *const opencl = eopencl->opencl;
  VP9_COMMON *const cm = &cpi->common;
  YV12_BUFFER_CONFIG *frm_ref = get_ref_frame_buffer(cpi, LAST_FRAME);
  opencl_buffer *gpu_output_pro_me_sub_buffer =
      &eopencl->gpu_output_pro_me_sub_buf[cm->current_video_frame & 1]
                                         [sub_frame_idx];
  opencl_buffer *gpu_output_me_sub_buffer =
      &eopencl->gpu_output_me_sub_buf[sub_frame_idx];
  opencl_buffer *rdopt_parameters =
      &eopencl->rdopt_parameters[cm->current_video_frame & 1];
  SubFrameInfo subframe;
  int subframe_height;
  int blocks_in_col, blocks_in_row;
  int block_row_offset;
  size_t local_size[2];
  size_t global_size[2];
  size_t global_offset[2];
  const size_t workitem_size[2] = {NUM_PIXELS_PER_WORKITEM, 1};
  cl_int status = CL_SUCCESS;
#if OPENCL_PROFILING
  cl_event event[NUM_KERNELS_ME + 1];
#endif
  cl_event *event_ptr[NUM_KERNELS_ME + 1];
  GPU_BLOCK_SIZE gpu_bsize;
  int i;

  for (i = 0; i < NUM_KERNELS_ME + 1; i++) {
#if OPENCL_PROFILING
    event_ptr[i] = &event[i];
#else
    event_ptr[i] = NULL;
#endif
  }

  (void)status;

  if (async) {
    frm_ref = get_frame_new_buffer(cm);
    gpu_output_pro_me_sub_buffer =
        &eopencl->gpu_output_pro_me_sub_buf[(cm->current_video_frame + 1) & 1]
                                            [sub_frame_idx];
    rdopt_parameters =
          &eopencl->rdopt_parameters[(cm->current_video_frame + 1) & 1];
  }

  // before launching kernels make sure the buffers needed by GPU are cache
  // synced
  // release reference buffer to GPU
  if (frm_ref->buffer_alloc_dup != NULL &&
      (sub_frame_idx == 0 || sub_frame_idx == cpi->b_async)) {
    status = clEnqueueUnmapMemObject(opencl->cmd_queue, frm_ref->gpu_mem_sub,
                                     frm_ref->buffer_alloc_dup, 0, NULL, NULL);
    frm_ref->buffer_alloc_dup = NULL;
    assert(status == CL_SUCCESS);
  }

  // release gpu rd buffers
  if (vp9_opencl_unmap_buffer(opencl, rdopt_parameters, CL_FALSE)) {
    assert(0);
  }

  // release gpu input buffers
  if (vp9_opencl_unmap_buffer(opencl, &eopencl->gpu_input, CL_FALSE)) {
    assert(0);
  }

  // release gpu output sub buffer
  if (vp9_opencl_unmap_buffer(opencl, gpu_output_me_sub_buffer, CL_FALSE)) {
    assert(0);
  }

  // set up kernel args
  if (sub_frame_idx == 0) {
    vp9_eopencl_set_dynamic_kernel_args_me(cpi, async);
  }

  vp9_subframe_init(&subframe, cm, sub_frame_idx);

  //=====   CHOOSE PARTITIONING KERNELS   =====
  //--------------------------------------------
  blocks_in_row = cm->sb_cols;
  blocks_in_col = (mi_cols_aligned_to_sb(subframe.mi_row_end) -
      subframe.mi_row_start) >> MI_BLOCK_SIZE_LOG2;
  block_row_offset = subframe.mi_row_start >> MI_BLOCK_SIZE_LOG2;

  local_size[0] = 8;
  local_size[1] = 8;

  global_size[0] = (blocks_in_row * local_size[0]);
  global_size[1] = (blocks_in_col * local_size[1]);

  global_offset[0] = 0;
  global_offset[1] = (block_row_offset * local_size[1]);

  status = clEnqueueNDRangeKernel(opencl->cmd_queue,
                                  eopencl->choose_partitions,
                                  2, global_offset, global_size, local_size,
                                  0, NULL, event_ptr[NUM_KERNELS_ME]);
  assert(status == CL_SUCCESS);

  gpu_output_pro_me_sub_buffer->mapped_pointer =
      clEnqueueMapBuffer(opencl->cmd_queue,
                         gpu_output_pro_me_sub_buffer->opencl_mem,
                         CL_FALSE,
                         CL_MAP_READ,
                         0, gpu_output_pro_me_sub_buffer->size, 0,
                         NULL, NULL, &status);
  assert(status == CL_SUCCESS);

  if (eopencl->event[sub_frame_idx] != NULL) {
    status = clReleaseEvent(eopencl->event[sub_frame_idx]);
    eopencl->event[sub_frame_idx] = NULL;
    assert(status == CL_SUCCESS);
  }
  status = clEnqueueMarker(opencl->cmd_queue, &eopencl->event[sub_frame_idx]);
  assert(status == CL_SUCCESS);


  //=====   MOTION ESTIMATION KERNELS   =====
  //--------------------------------------------
  subframe_height =
      (subframe.mi_row_end - subframe.mi_row_start) << MI_SIZE_LOG2;

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
    const BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);

    const int b_width_in_pixels_log2 = b_width_log2_lookup[bsize] + 2;
    const int b_width_in_pixels = 1 << b_width_in_pixels_log2;
    const int b_height_in_pixels_log2 = b_height_log2_lookup[bsize] + 2;
    const int b_height_in_pixels = 1 << b_height_in_pixels_log2;
    const int b_height_mask = b_height_in_pixels - 1;

    size_t local_size_zeromv[2];
    size_t local_size_full_pixel[2], local_size_sub_pixel[2];
    const int ms_pixels = (num_8x8_blocks_wide_lookup[bsize] / 2) * 8;

    block_row_offset = subframe.mi_row_start >> mi_height_log2(bsize);

    blocks_in_col = subframe_height >> b_height_in_pixels_log2;
    blocks_in_row = cm->sb_cols * num_mxn_blocks_wide_lookup[bsize];

    if (sub_frame_idx == MAX_SUB_FRAMES - 1) {
      if ((cm->height & b_height_mask) > ms_pixels) {
        blocks_in_col++;
      }
    }

    // For very small resolutions, this could happen for the last few sub-frames
    if (blocks_in_col == 0) {
      goto skip_execution;
    }

    local_size[0] = b_width_in_pixels / workitem_size[0];
    local_size[1] = b_height_in_pixels / workitem_size[1];

    // launch full pixel search kernel zero mv analysis
    local_size_zeromv[0] = local_size[0];
    local_size_zeromv[1] = local_size[1] >> pixel_rows_per_workitem_log2_zeromv;

    global_size[0] = blocks_in_row * local_size_zeromv[0];
    global_size[1] = blocks_in_col * local_size_zeromv[1];

    global_offset[0] = 0;
    global_offset[1] = block_row_offset * local_size_zeromv[1];

    status = clEnqueueNDRangeKernel(opencl->cmd_queue,
                                    eopencl->rd_calculation_zeromv[gpu_bsize],
                                    2, global_offset, global_size,
                                    local_size_zeromv,
                                    0, NULL, event_ptr[0]);
    assert(status == CL_SUCCESS);

    // launch full pixel search new mv analysis kernel
    local_size_full_pixel[0] = local_size[0] * 4;
    local_size_full_pixel[1] =
        local_size[1] >> pixel_rows_per_workitem_log2_full_pixel[gpu_bsize];

    global_size[0] = blocks_in_row * local_size_full_pixel[0];
    global_size[1] = blocks_in_col * local_size_full_pixel[1];

    global_offset[0] = 0;
    global_offset[1] = block_row_offset * local_size_full_pixel[1];

    status = clEnqueueNDRangeKernel(opencl->cmd_queue,
                                    eopencl->full_pixel_search[gpu_bsize],
                                    2, global_offset, global_size,
                                    local_size_full_pixel,
                                    0, NULL, event_ptr[1]);
    assert(status == CL_SUCCESS);

    // launch sub pixel search kernel (half pel)
    local_size_sub_pixel[0] = local_size[0];
    local_size_sub_pixel[1] =
        local_size[1] >> pixel_rows_per_workitem_log2_sub_pixel[gpu_bsize];

    local_size_sub_pixel[0] *= 8;

    global_size[0] = blocks_in_row * local_size_sub_pixel[0];
    global_size[1] = blocks_in_col * local_size_sub_pixel[1];

    global_offset[0] = 0;
    global_offset[1] = block_row_offset * local_size_sub_pixel[1];

    status = clEnqueueNDRangeKernel(opencl->cmd_queue,
                                    eopencl->hpel_search[gpu_bsize],
                                    2, global_offset, global_size,
                                    local_size_sub_pixel,
                                    0, NULL, event_ptr[2]);
    assert(status == CL_SUCCESS);

    // launch sub pixel search kernel (quarter pel)
    global_size[0] = blocks_in_row * local_size_sub_pixel[0];
    global_size[1] = blocks_in_col * local_size_sub_pixel[1];

    global_offset[0] = 0;
    global_offset[1] = block_row_offset * local_size_sub_pixel[1];

    status = clEnqueueNDRangeKernel(opencl->cmd_queue,
                                    eopencl->qpel_search[gpu_bsize],
                                    2, global_offset, global_size,
                                    local_size_sub_pixel,
                                    0, NULL, event_ptr[3]);
    assert(status == CL_SUCCESS);

#if OPENCL_PROFILING
    for (i = 0; i < NUM_KERNELS_ME - 2; i++) {
      cl_ulong time_elapsed;
      status = clWaitForEvents(1, event_ptr[i]);
      assert(status == CL_SUCCESS);
      time_elapsed = get_event_time_elapsed(*event_ptr[i]);
      eopencl->total_time_taken_me[gpu_bsize][i] += time_elapsed / 1000;
      status = clReleaseEvent(*event_ptr[i]);
      assert(status == CL_SUCCESS);
    }
#endif
  }

  // Lowest GPU Block size selected for the merged kernels
  gpu_bsize = GPU_BLOCK_32X32;
  {
    const BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);
    const int b_width_in_pixels_log2 = b_width_log2_lookup[bsize] + 2;
    const int b_width_in_pixels = 1 << b_width_in_pixels_log2;
    const int b_height_in_pixels_log2 = b_height_log2_lookup[bsize] + 2;
    const int b_height_in_pixels = 1 << b_height_in_pixels_log2;
    const int b_height_mask = b_height_in_pixels - 1;
    const int ms_pixels = (num_8x8_blocks_wide_lookup[bsize] / 2) * 8;
    size_t local_size_inter_pred[2];

    block_row_offset = subframe.mi_row_start >> mi_height_log2(bsize);
    blocks_in_col = subframe_height >> b_height_in_pixels_log2;
    blocks_in_row = cm->sb_cols * num_mxn_blocks_wide_lookup[bsize];

    if (sub_frame_idx == MAX_SUB_FRAMES - 1)
      if ((cm->height & b_height_mask) > ms_pixels)
        blocks_in_col++;

    local_size[0] = b_width_in_pixels / workitem_size[0];
    local_size[1] = b_height_in_pixels / workitem_size[1];

    // launch inter prediction and sse compute kernel
    local_size_inter_pred[0] = local_size[0];
    local_size_inter_pred[1] =
        local_size[1] >> pixel_rows_per_workitem_log2_inter_pred[gpu_bsize];

    global_size[0] = blocks_in_row * local_size_inter_pred[0] * 2;
    global_size[1] = blocks_in_col * local_size_inter_pred[1];

    global_offset[0] = 0;
    global_offset[1] = block_row_offset * local_size_inter_pred[1];

    status = clEnqueueNDRangeKernel(opencl->cmd_queue,
                                    eopencl->inter_prediction_and_sse[gpu_bsize],
                                    2, global_offset, global_size,
                                    local_size_inter_pred,
                                    0, NULL, event_ptr[4]);
    assert(status == CL_SUCCESS);

    // launch rd compute kernel
    global_size[0] = blocks_in_row;
    global_size[1] = blocks_in_col;

    global_offset[0] = 0;
    global_offset[1] = block_row_offset;

    status = clEnqueueNDRangeKernel(opencl->cmd_queue,
                                    eopencl->rd_calculation_newmv[gpu_bsize],
                                    2, global_offset, global_size, NULL,
                                    0, NULL, event_ptr[5]);
    assert(status == CL_SUCCESS);

#if OPENCL_PROFILING
    for ( ; i < NUM_KERNELS_ME; i++) {
      cl_ulong time_elapsed;
      status = clWaitForEvents(1, event_ptr[i]);
      assert(status == CL_SUCCESS);
      time_elapsed = get_event_time_elapsed(*event_ptr[i]);
      eopencl->total_time_taken_me[gpu_bsize][i] += time_elapsed / 1000;
      status = clReleaseEvent(*event_ptr[i]);
      assert(status == CL_SUCCESS);
    }

    {
      cl_ulong time_elapsed;
      status = clWaitForEvents(1, event_ptr[i]);
      assert(status == CL_SUCCESS);
      time_elapsed = get_event_time_elapsed(*event_ptr[i]);
      eopencl->total_time_taken_pro_me[NUM_KERNELS_PRO_ME - 1] +=
          time_elapsed / 1000;
      status = clReleaseEvent(*event_ptr[i]);
      assert(status == CL_SUCCESS);
    }
#endif
  }

skip_execution:

  if (frm_ref->buffer_alloc_dup == NULL) {
    frm_ref->buffer_alloc_dup =
        clEnqueueMapBuffer(opencl->cmd_queue, frm_ref->gpu_mem_sub, CL_FALSE,
                           CL_MAP_READ | CL_MAP_WRITE, 0,
                           frm_ref->buffer_alloc_sz, 0, NULL, NULL, &status);
    assert(status == CL_SUCCESS);
  }

  gpu_output_me_sub_buffer->mapped_pointer =
      clEnqueueMapBuffer(opencl->cmd_queue,
                         gpu_output_me_sub_buffer->opencl_mem,
                         CL_FALSE,
                         CL_MAP_READ,
                         0, gpu_output_me_sub_buffer->size, 0,
                         NULL, NULL, &status);
  assert(status == CL_SUCCESS);

  status = clFlush(opencl->cmd_queue);
  assert(status == CL_SUCCESS);

  if (eopencl->event[MAX_SUB_FRAMES + sub_frame_idx] != NULL) {
    status = clReleaseEvent(eopencl->event[MAX_SUB_FRAMES + sub_frame_idx]);
    eopencl->event[MAX_SUB_FRAMES + sub_frame_idx] = NULL;
    assert(status == CL_SUCCESS);
  }

  status = clEnqueueMarker(opencl->cmd_queue,
                           &eopencl->event[MAX_SUB_FRAMES + sub_frame_idx]);
  assert(status == CL_SUCCESS);

  return;
}

void vp9_eopencl_remove(VP9_COMP *cpi) {
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  GPU_BLOCK_SIZE gpu_bsize;
  cl_int status;
  int i;
#if OPENCL_PROFILING
  cl_ulong total[NUM_KERNELS_ME] = {0};
  cl_ulong grand_total = 0;
  fprintf(stdout, "\nOPENCL PROFILE RESULTS\n");
#endif

  for (i = 0; i < 2 * MAX_SUB_FRAMES; i++) {
    if (eopencl->event[i] != NULL) {
      status = clReleaseEvent(eopencl->event[i]);
      eopencl->event[i] = NULL;
      assert(status == CL_SUCCESS);
    }
  }

  status = clReleaseKernel(eopencl->col_row_projection);
  if (status != CL_SUCCESS)
    goto fail;

  status = clReleaseKernel(eopencl->vector_match);
  if (status != CL_SUCCESS)
    goto fail;

  status = clReleaseKernel(eopencl->pro_motion_estimation);
  if (status != CL_SUCCESS)
    goto fail;

  status = clReleaseKernel(eopencl->color_sensitivity);
  if (status != CL_SUCCESS)
    goto fail;

  status = clReleaseKernel(eopencl->choose_partitions);
  if (status != CL_SUCCESS)
    goto fail;

#if OPENCL_PROFILING
  fprintf(stdout, "\nPRO ME KERNELS\n");
  for (i = 0; i < NUM_KERNELS_PRO_ME; i++) {
    fprintf(stdout, "\tKernel %d - TOTAL = %"PRIu64" microseconds\n", i,
            eopencl->total_time_taken_pro_me[i]);
  }
#endif

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
#if OPENCL_PROFILING
    fprintf(stdout, "\nBlock size idx = %d\n", gpu_bsize);
    for (i = 0; i < NUM_KERNELS_ME; i++) {
      total[i] += eopencl->total_time_taken_me[gpu_bsize][i];
      fprintf(stdout, "\tKernel %d - TOTAL = %"PRIu64" microseconds\n", i,
              eopencl->total_time_taken_me[gpu_bsize][i]);
    }
#endif
    status = clReleaseKernel(eopencl->rd_calculation_zeromv[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;
    status = clReleaseKernel(eopencl->full_pixel_search[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;
    status = clReleaseKernel(eopencl->hpel_search[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;
    status = clReleaseKernel(eopencl->qpel_search[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;
    status = clReleaseKernel(eopencl->inter_prediction_and_sse[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;
    status = clReleaseKernel(eopencl->rd_calculation_newmv[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;
  }

#if OPENCL_PROFILING
  fprintf(stdout, "\nTOTAL FOR ALL BLOCK SIZES\n");
  for (i = 0; i < NUM_KERNELS_ME; i++) {
    grand_total += total[i];
    fprintf(stdout,
            "\tKernel %d - TOTAL ALL BLOCK SIZES = %"PRIu64" microseconds\n",
            i, total[i]);
  }
  total[0] = 0;
  for (i = 0; i < NUM_KERNELS_PRO_ME; i++) {
    total[0] += eopencl->total_time_taken_pro_me[i];
  }
  fprintf(stdout, "\nPRO ME TOTAL = %"PRIu64"\n", total[0]);
  grand_total += total[0];
  fprintf(stdout, "\nGRAND TOTAL = %"PRIu64"\n", grand_total);
#endif

  return;

fail:
  assert(0);
  return;
}

static int vp9_eopencl_build_subpel_kernel(VP9_COMP *cpi) {
  VP9_OPENCL *opencl = cpi->common.gpu.compute_framework;
  VP9_EOPENCL *eopencl = cpi->egpu.compute_framework;
  cl_int status = CL_SUCCESS;
  cl_device_id device = opencl->device;
  cl_program program;
  const char *kernel_file_name= PREFIX_PATH"vp9_subpel.cl";
  char build_options[BUILD_OPTION_LENGTH];
  char *kernel_src = NULL;
  GPU_BLOCK_SIZE gpu_bsize;
  BLOCK_SIZE bsize;

  // Read kernel source files
  kernel_src = read_src(kernel_file_name);
  if (kernel_src == NULL)
    goto fail;

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
    bsize = get_actual_block_size(gpu_bsize);
    program = clCreateProgramWithSource(opencl->context, 1,
                                        (const char**)(void *)&kernel_src,
                                        NULL,
                                        &status);
    if (status != CL_SUCCESS)
      goto fail;

    sprintf(build_options,
            "-I %s -DBLOCK_SIZE_IN_PIXELS=%d -DPIXEL_ROWS_PER_WORKITEM=%d",
            PREFIX_PATH,
            num_8x8_blocks_wide_lookup[bsize] * 8,
            1 << pixel_rows_per_workitem_log2_sub_pixel[gpu_bsize]);

    // Build the program
    status = clBuildProgram(program, 1, &device,
                            build_options,
                            NULL, NULL);
    if (status != CL_SUCCESS) {
      // Enable this if you are a OpenCL developer and need to print the build
      // errors of the OpenCL kernel
#if OPENCL_DEVELOPER_MODE
      uint8_t *build_log;
      size_t build_log_size;

      clGetProgramBuildInfo(program,
                            device,
                            CL_PROGRAM_BUILD_LOG,
                            0,
                            NULL,
                            &build_log_size);
      build_log = (uint8_t*)vpx_malloc(build_log_size);
      if (build_log == NULL)
        goto fail;

      clGetProgramBuildInfo(program,
                            device,
                            CL_PROGRAM_BUILD_LOG,
                            build_log_size,
                            build_log,
                            NULL);
      build_log[build_log_size-1] = '\0';
      fprintf(stderr, "Build Log:\n%s\n", build_log);
      vpx_free(build_log);
#endif
      goto fail;
    }

    eopencl->hpel_search[gpu_bsize] = clCreateKernel(
        program, "vp9_sub_pixel_search_halfpel_filtering", &status);
    if (status != CL_SUCCESS)
      goto fail;

    eopencl->qpel_search[gpu_bsize] = clCreateKernel(
        program, "vp9_sub_pixel_search_quarterpel_filtering", &status);
    if (status != CL_SUCCESS)
      goto fail;

    status = clReleaseProgram(program);
    if (status != CL_SUCCESS)
      goto fail;
  }

  vpx_free(kernel_src);
  return 0;

  fail:
  if (kernel_src != NULL)
    vpx_free(kernel_src);
  return 1;
}

static int vp9_eopencl_build_fullpel_kernel(VP9_COMP *cpi) {
  VP9_OPENCL *opencl = cpi->common.gpu.compute_framework;
  VP9_EOPENCL *eopencl = cpi->egpu.compute_framework;
  cl_int status = CL_SUCCESS;
  cl_device_id device = opencl->device;
  cl_program program;
  const char *kernel_file_name= PREFIX_PATH"vp9_fullpel.cl";
  char build_options[BUILD_OPTION_LENGTH];
  char *kernel_src = NULL;
  GPU_BLOCK_SIZE gpu_bsize;
  BLOCK_SIZE bsize;

  // Read kernel source files
  kernel_src = read_src(kernel_file_name);
  if (kernel_src == NULL)
    goto fail;

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
    bsize = get_actual_block_size(gpu_bsize);
    program = clCreateProgramWithSource(opencl->context, 1,
                                        (const char**)(void *)&kernel_src,
                                        NULL,
                                        &status);
    if (status != CL_SUCCESS)
      goto fail;

    sprintf(build_options,
            "-I %s -DBLOCK_SIZE_IN_PIXELS=%d -DPIXEL_ROWS_PER_WORKITEM=%d",
            PREFIX_PATH,
            num_8x8_blocks_wide_lookup[bsize] * 8,
            1 << pixel_rows_per_workitem_log2_full_pixel[gpu_bsize]);

    // Build the program
    status = clBuildProgram(program, 1, &device, build_options,
                            NULL, NULL);
    if (status != CL_SUCCESS) {
      // Enable this if you are a OpenCL developer and need to print the build
      // errors of the OpenCL kernel
#if OPENCL_DEVELOPER_MODE
      uint8_t *build_log;
      size_t build_log_size;

      clGetProgramBuildInfo(program,
                            device,
                            CL_PROGRAM_BUILD_LOG,
                            0,
                            NULL,
                            &build_log_size);
      build_log = (uint8_t*)vpx_malloc(build_log_size);
      if (build_log == NULL)
        goto fail;

      clGetProgramBuildInfo(program,
                            device,
                            CL_PROGRAM_BUILD_LOG,
                            build_log_size,
                            build_log,
                            NULL);
      build_log[build_log_size-1] = '\0';
      fprintf(stderr, "Build Log:\n%s\n", build_log);
      vpx_free(build_log);
#endif
      goto fail;
    }

    eopencl->full_pixel_search[gpu_bsize] = clCreateKernel(
        program, "vp9_full_pixel_search", &status);
    if (status != CL_SUCCESS)
      goto fail;

    status = clReleaseProgram(program);
    if (status != CL_SUCCESS)
      goto fail;
  }

  vpx_free(kernel_src);
  return 0;

  fail:
  if (kernel_src != NULL)
    vpx_free(kernel_src);
  return 1;
}

static int vp9_eopencl_build_zeromv_kernel(VP9_COMP *cpi) {
  VP9_OPENCL *opencl = cpi->common.gpu.compute_framework;
  VP9_EOPENCL *eopencl = cpi->egpu.compute_framework;
  cl_int status = CL_SUCCESS;
  cl_device_id device = opencl->device;
  cl_program program;
  const char *kernel_file_name= PREFIX_PATH"vp9_rd.cl";
  char build_options[BUILD_OPTION_LENGTH];
  char *kernel_src = NULL;
  GPU_BLOCK_SIZE gpu_bsize;
  BLOCK_SIZE bsize;

  // Read kernel source files
  kernel_src = read_src(kernel_file_name);
  if (kernel_src == NULL)
    goto fail;

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
    bsize = get_actual_block_size(gpu_bsize);
    program = clCreateProgramWithSource(opencl->context, 1,
                                        (const char**)(void *)&kernel_src,
                                        NULL,
                                        &status);
    if (status != CL_SUCCESS)
      goto fail;

    sprintf(build_options,
            "-I %s -DBLOCK_SIZE_IN_PIXELS=%d -DPIXEL_ROWS_PER_WORKITEM=%d",
            PREFIX_PATH,
            num_8x8_blocks_wide_lookup[bsize] * 8,
            1 << pixel_rows_per_workitem_log2_zeromv);

    // Build the program
    status = clBuildProgram(program, 1, &device,
                            build_options,
                            NULL, NULL);
    if (status != CL_SUCCESS) {
      // Enable this if you are a OpenCL developer and need to print the build
      // errors of the OpenCL kernel
#if OPENCL_DEVELOPER_MODE
      uint8_t *build_log;
      size_t build_log_size;

      clGetProgramBuildInfo(program,
                            device,
                            CL_PROGRAM_BUILD_LOG,
                            0,
                            NULL,
                            &build_log_size);
      build_log = (uint8_t*)vpx_malloc(build_log_size);
      if (build_log == NULL)
        goto fail;

      clGetProgramBuildInfo(program,
                            device,
                            CL_PROGRAM_BUILD_LOG,
                            build_log_size,
                            build_log,
                            NULL);
      build_log[build_log_size-1] = '\0';
      fprintf(stderr, "Build Log:\n%s\n", build_log);
      vpx_free(build_log);
#endif
      goto fail;
    }

    eopencl->rd_calculation_zeromv[gpu_bsize] =
        clCreateKernel(program, "vp9_zero_motion_search", &status);
    if (status != CL_SUCCESS)
      goto fail;

    status = clReleaseProgram(program);
    if (status != CL_SUCCESS)
      goto fail;
  }

  vpx_free(kernel_src);
  return 0;

fail:
  if (kernel_src != NULL)
    vpx_free(kernel_src);
  return 1;
}

static int vp9_eopencl_build_rd_kernel(VP9_COMP *cpi) {
  VP9_OPENCL *opencl = cpi->common.gpu.compute_framework;
  VP9_EOPENCL *eopencl = cpi->egpu.compute_framework;
  cl_int status = CL_SUCCESS;
  cl_device_id device = opencl->device;
  cl_program program;
  const char *kernel_file_name= PREFIX_PATH"vp9_rd.cl";
  char build_options[BUILD_OPTION_LENGTH];
  char *kernel_src = NULL;
  GPU_BLOCK_SIZE gpu_bsize;
  BLOCK_SIZE bsize;

  // Read kernel source files
  kernel_src = read_src(kernel_file_name);
  if (kernel_src == NULL)
    goto fail;

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
    bsize = get_actual_block_size(gpu_bsize);
    program = clCreateProgramWithSource(opencl->context, 1,
                                        (const char**)(void *)&kernel_src,
                                        NULL,
                                        &status);
    if (status != CL_SUCCESS)
      goto fail;

    sprintf(build_options,
            "-I %s -DBLOCK_SIZE_IN_PIXELS=%d -DPIXEL_ROWS_PER_WORKITEM=%d",
            PREFIX_PATH,
            num_8x8_blocks_wide_lookup[bsize] * 8,
            1 << pixel_rows_per_workitem_log2_inter_pred[gpu_bsize]);

    // Build the program
    status = clBuildProgram(program, 1, &device,
                            build_options,
                            NULL, NULL);
    if (status != CL_SUCCESS) {
      // Enable this if you are a OpenCL developer and need to print the build
      // errors of the OpenCL kernel
#if OPENCL_DEVELOPER_MODE
      uint8_t *build_log;
      size_t build_log_size;

      clGetProgramBuildInfo(program,
                            device,
                            CL_PROGRAM_BUILD_LOG,
                            0,
                            NULL,
                            &build_log_size);
      build_log = (uint8_t*)vpx_malloc(build_log_size);
      if (build_log == NULL)
        goto fail;

      clGetProgramBuildInfo(program,
                            device,
                            CL_PROGRAM_BUILD_LOG,
                            build_log_size,
                            build_log,
                            NULL);
      build_log[build_log_size-1] = '\0';
      fprintf(stderr, "Build Log:\n%s\n", build_log);
      vpx_free(build_log);
#endif
      goto fail;
    }

    eopencl->rd_calculation_newmv[gpu_bsize] = clCreateKernel(
        program, "vp9_rd_calculation", &status);
    if (status != CL_SUCCESS)
      goto fail;

    eopencl->inter_prediction_and_sse[gpu_bsize] =
        clCreateKernel(program, "vp9_inter_prediction_and_sse", &status);
    if (status != CL_SUCCESS)
      goto fail;

    status = clReleaseProgram(program);
    if (status != CL_SUCCESS)
      goto fail;
  }

  vpx_free(kernel_src);
  return 0;

  fail:
  if (kernel_src != NULL)
    vpx_free(kernel_src);
  return 1;
}

static int vp9_eopencl_build_choose_partitioning_kernel(VP9_COMP *cpi) {
  VP9_OPENCL *opencl = cpi->common.gpu.compute_framework;
  VP9_EOPENCL *eopencl = cpi->egpu.compute_framework;
  cl_int status = CL_SUCCESS;
  cl_device_id device = opencl->device;
  cl_program program;
  const char *kernel_file_name= PREFIX_PATH"vp9_partitioning.cl";
  char build_options[BUILD_OPTION_LENGTH];
  char *kernel_src = NULL;

  // Read kernel source files
  kernel_src = read_src(kernel_file_name);
  if (kernel_src == NULL)
    goto fail;

  program = clCreateProgramWithSource(opencl->context, 1,
                                      (const char**)(void *)&kernel_src,
                                      NULL,
                                      &status);
  if (status != CL_SUCCESS)
    goto fail;

  sprintf(build_options,
          "-I %s -DBLOCK_SIZE_IN_PIXELS=%d -DPIXEL_ROWS_PER_WORKITEM=%d",
          PREFIX_PATH,
          64,
          1 << pixel_rows_per_workitem_log2_pro_me);

  // Build the program
  status = clBuildProgram(program, 1, &device, build_options, NULL, NULL);
  if (status != CL_SUCCESS) {
    // Enable this if you are a OpenCL developer and need to print the build
    // errors of the OpenCL kernel
#if OPENCL_DEVELOPER_MODE
    uint8_t *build_log;
    size_t build_log_size;

    clGetProgramBuildInfo(program,
                          device,
                          CL_PROGRAM_BUILD_LOG,
                          0,
                          NULL,
                          &build_log_size);
    build_log = (uint8_t*)vpx_malloc(build_log_size);
    if (build_log == NULL)
      goto fail;

    clGetProgramBuildInfo(program,
                          device,
                          CL_PROGRAM_BUILD_LOG,
                          build_log_size,
                          build_log,
                          NULL);
    build_log[build_log_size-1] = '\0';
    fprintf(stderr, "Build Log:\n%s\n", build_log);
    vpx_free(build_log);
#endif
    goto fail;
  }

  eopencl->col_row_projection = clCreateKernel(program, "vp9_col_row_projection",
                                           &status);
  if (status != CL_SUCCESS)
    goto fail;

  eopencl->vector_match = clCreateKernel(program, "vp9_vector_match", &status);
  if (status != CL_SUCCESS)
    goto fail;

  eopencl->pro_motion_estimation = clCreateKernel(program,
                                                  "vp9_pro_motion_estimation",
                                                  &status);
  if (status != CL_SUCCESS)
    goto fail;

  eopencl->color_sensitivity = clCreateKernel(program, "vp9_color_sensitivity",
                                              &status);
  if (status != CL_SUCCESS)
    goto fail;

  eopencl->choose_partitions = clCreateKernel(program, "vp9_choose_partitions",
                                              &status);
  if (status != CL_SUCCESS)
    goto fail;

  status = clReleaseProgram(program);
  if (status != CL_SUCCESS)
    goto fail;

  vpx_free(kernel_src);
  return 0;

fail:
  if (kernel_src != NULL)
    vpx_free(kernel_src);
  return 1;
}

int vp9_eopencl_init(VP9_COMP *cpi) {
  VP9_COMMON *cm = &cpi->common;
  VP9_GPU *gpu = &cm->gpu;
  VP9_OPENCL *opencl = gpu->compute_framework;
  VP9_EGPU *egpu = &cpi->egpu;
  VP9_EOPENCL *eopencl;

  egpu->compute_framework = vpx_calloc(1, sizeof(VP9_EOPENCL));
  egpu->alloc_buffers = vp9_eopencl_alloc_buffers;
  egpu->free_buffers = vp9_eopencl_free_buffers;
  egpu->acquire_input_buffer = vp9_eopencl_acquire_input_buffer;
  egpu->acquire_output_me_buffer = vp9_eopencl_acquire_output_me_buffer;
  egpu->acquire_output_pro_me_buffer = vp9_eopencl_acquire_output_pro_me_buffer;
  egpu->acquire_rd_param_buffer = vp9_eopencl_acquire_rd_param_buffer;
  egpu->enc_sync_read = vp9_eopencl_enc_sync_read;
  egpu->execute = vp9_eopencl_execute;
  egpu->execute_prologue = vp9_eopencl_execute_prologue;
  egpu->remove = vp9_eopencl_remove;
  eopencl = egpu->compute_framework;
  eopencl->opencl = opencl;

  if (vp9_eopencl_build_choose_partitioning_kernel(cpi))
    return 1;

  if (vp9_eopencl_build_zeromv_kernel(cpi))
    return 1;

  if (vp9_eopencl_build_rd_kernel(cpi))
    return 1;

  if (vp9_eopencl_build_fullpel_kernel(cpi))
    return 1;

  if (vp9_eopencl_build_subpel_kernel(cpi))
    return 1;

  return 0;
}

