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

static void *vp9_opencl_alloc_frame_buffers(VP9_COMMON *cm, int frame_size,
                                            void **frame_buffer) {
  VP9_OPENCL *opencl = cm->gpu.compute_framework;
  opencl_pic_buf *frame_buff = NULL;
  cl_mem opencl_mem, opencl_mem_sub;
  void *mapped_pointer;
  cl_buffer_region sf_region;
  cl_int status = CL_SUCCESS;

  frame_buff = calloc(1, sizeof(*frame_buff));
  *frame_buffer = frame_buff;
  if (frame_buff == NULL)
    return NULL;

  // allocate buffer to store source/reference
  opencl_mem = clCreateBuffer(opencl->context,
                              CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                              frame_size, NULL, &status);
  if (status != CL_SUCCESS)
    return NULL;
  mapped_pointer = clEnqueueMapBuffer(opencl->cmd_queue, opencl_mem, CL_TRUE,
                                      CL_MAP_READ | CL_MAP_WRITE, 0, frame_size,
                                      0, NULL, NULL, &status);
  if (status != CL_SUCCESS) {
    clReleaseMemObject(opencl_mem);
    opencl_mem = NULL;
    return NULL;
  }
  frame_buff->frame_buffer.mapped_pointer = mapped_pointer;
  frame_buff->frame_buffer.opencl_mem = opencl_mem;
  frame_buff->frame_buffer.size = frame_size;

  // create a duplicate buffer for the above buffer
  // used while mantaining cache sync b/w cpu & gpu
  sf_region.origin = 0;
  sf_region.size = frame_size;
  opencl_mem_sub = clCreateSubBuffer(opencl_mem, CL_MEM_READ_ONLY,
                                     CL_BUFFER_CREATE_TYPE_REGION, &sf_region,
                                     &status);
  if (status != CL_SUCCESS)
    return NULL;
  mapped_pointer = clEnqueueMapBuffer(opencl->cmd_queue, opencl_mem_sub,
                                      CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
                                      frame_size, 0, NULL, NULL, &status);
  if (status != CL_SUCCESS) {
    clReleaseMemObject(opencl_mem_sub);
    opencl_mem_sub = NULL;
    return NULL;
  }
  frame_buff->sub_buffer.mapped_pointer = mapped_pointer;
  frame_buff->sub_buffer.opencl_mem = opencl_mem_sub;
  frame_buff->sub_buffer.size = frame_size;

  return frame_buff->frame_buffer.mapped_pointer;
}

static void vp9_opencl_release_frame_buffers(VP9_COMMON *cm,
                                             void *frame_buffer) {
  VP9_OPENCL *opencl = cm->gpu.compute_framework;
  opencl_pic_buf *frame_buff = frame_buffer;
  cl_int status = CL_SUCCESS;

  if (frame_buff == NULL)
    return;

  if (frame_buff->sub_buffer.mapped_pointer != NULL) {
    status = clEnqueueUnmapMemObject(opencl->cmd_queue,
                                     frame_buff->sub_buffer.opencl_mem,
                                     frame_buff->sub_buffer.mapped_pointer,
                                     0, NULL, NULL);
    status |= clFinish(opencl->cmd_queue);
    if (status != CL_SUCCESS)
      goto fail;
    frame_buff->sub_buffer.mapped_pointer = NULL;
  }
  if (frame_buff->sub_buffer.opencl_mem != NULL) {
    status = clReleaseMemObject(frame_buff->sub_buffer.opencl_mem);
    if (status != CL_SUCCESS)
      goto fail;
    frame_buff->sub_buffer.opencl_mem = NULL;
  }

  if (frame_buff->frame_buffer.mapped_pointer != NULL) {
    status = clEnqueueUnmapMemObject(opencl->cmd_queue,
                                     frame_buff->frame_buffer.opencl_mem,
                                     frame_buff->frame_buffer.mapped_pointer,
                                     0, NULL, NULL);
    status |= clFinish(opencl->cmd_queue);
    if (status != CL_SUCCESS)
      goto fail;
    frame_buff->frame_buffer.mapped_pointer = NULL;

  }
  if (frame_buff->frame_buffer.opencl_mem != NULL) {
    status = clReleaseMemObject(frame_buff->frame_buffer.opencl_mem);
    if (status != CL_SUCCESS)
      goto fail;
    frame_buff->frame_buffer.opencl_mem = NULL;
    frame_buff->frame_buffer.size = 0;
  }

  free(frame_buff);
  frame_buff = NULL;

  return;

fail:
  assert(0);
}

int vp9_opencl_map_buffer(VP9_OPENCL *const opencl,
                          opencl_buffer *opencl_buf,
                          cl_map_flags map_flags) {
  cl_int status = CL_SUCCESS;

  if (opencl_buf->mapped_pointer == NULL) {
    // GPU jobs that acquire pointer to the host are queued in to a different
    // command queue
    opencl_buf->mapped_pointer = clEnqueueMapBuffer(opencl->cmd_queue_memory,
                                                    opencl_buf->opencl_mem,
                                                    CL_TRUE,
                                                    map_flags,
                                                    0,
                                                    opencl_buf->size,
                                                    0, NULL, NULL, &status);
    if (status != CL_SUCCESS)
      goto fail;
  }

fail:
  return (int)status;
}

int vp9_opencl_unmap_buffer(VP9_OPENCL *const opencl,
                            opencl_buffer *opencl_buf,
                            cl_bool is_blocking) {
  cl_int status = CL_SUCCESS;

  if (opencl_buf->mapped_pointer != NULL) {
    // unmap jobs are queued to the same command queue where GPU ME and PRO-ME
    // jobs are queued
    status = clEnqueueUnmapMemObject(opencl->cmd_queue,
                                     opencl_buf->opencl_mem,
                                     opencl_buf->mapped_pointer,
                                     0, NULL, NULL);
    opencl_buf->mapped_pointer = NULL;
    if (status != CL_SUCCESS)
      goto fail;

    if (is_blocking == CL_TRUE) {
      status = clFinish(opencl->cmd_queue);
      if (status != CL_SUCCESS)
        goto fail;
    }
  }

fail:
  return (int)status;
}

int vp9_opencl_remove(VP9_COMMON *cm) {
  VP9_OPENCL *opencl = cm->gpu.compute_framework;
  cl_int status = CL_SUCCESS;

  if (cm->gpu.compute_framework == NULL)
    return (int)status;

  if (opencl->cmd_queue) {
    status = clReleaseCommandQueue(opencl->cmd_queue);
    if (status != CL_SUCCESS)
      goto fail;
    opencl->cmd_queue = NULL;
  }
  if (opencl->cmd_queue_memory) {
    status = clReleaseCommandQueue(opencl->cmd_queue_memory);
    if (status != CL_SUCCESS)
      goto fail;
    opencl->cmd_queue_memory = NULL;
  }
  if (opencl->context) {
    status = clReleaseContext(opencl->context);
    if (status != CL_SUCCESS)
      goto fail;
    opencl->context = NULL;
  }
  if (opencl->device) {
    free(opencl->device);
    opencl->device = NULL;
  }
  vpx_free(cm->gpu.compute_framework);
  cm->gpu.compute_framework = NULL;

fail:
  return (int)status;
}

int vp9_opencl_init(VP9_COMMON *cm) {
  VP9_GPU *gpu = &cm->gpu;
  VP9_OPENCL *opencl;
  cl_int status;
  cl_uint num_platforms = 0;
  cl_platform_id platform;
  cl_uint num_devices = 0;
  cl_device_id *device = NULL;
  cl_command_queue_properties command_queue_properties = 0;

#if OPENCL_PROFILING
  command_queue_properties = CL_QUEUE_PROFILING_ENABLE;
#endif
  CHECK_MEM_ERROR(cm, gpu->compute_framework,
                  vpx_calloc(1, sizeof(VP9_OPENCL)));
  gpu->alloc_frame_buffers = vp9_opencl_alloc_frame_buffers;
  gpu->release_frame_buffers = vp9_opencl_release_frame_buffers;
  gpu->remove = vp9_opencl_remove;
  opencl = gpu->compute_framework;

  // Get the number of platforms in the system.
  status = clGetPlatformIDs(0, NULL, &num_platforms);
  if (status != CL_SUCCESS || num_platforms == 0)
    goto fail;

  // Get the platform ID for one platform
  status = clGetPlatformIDs(1, &platform, NULL);
  if (status != CL_SUCCESS)
    goto fail;

  // Get the number of devices available on the platform
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  if (status != CL_SUCCESS || num_devices == 0)
    goto fail;
  opencl->num_devices = num_devices;

  // Get the device ID for one device
  device = malloc(sizeof(*device) * num_devices);
  if (device == NULL)
    goto fail;
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, device,
                          NULL);
  if (status != CL_SUCCESS)
    goto fail;
  opencl->device = device;

  // Create OpenCL context for devices
  opencl->context = clCreateContext(NULL, num_devices, device, NULL, NULL,
                                    &status);
  if (status != CL_SUCCESS || opencl->context == NULL)
    goto fail;

  // Create command queues for the device
  opencl->cmd_queue_memory = clCreateCommandQueue(opencl->context, device[0],
                                                  command_queue_properties,
                                                  &status);
  if (status != CL_SUCCESS || opencl->cmd_queue_memory == NULL)
    goto fail;

  opencl->cmd_queue = clCreateCommandQueue(opencl->context, device[0],
                                           command_queue_properties,
                                           &status);
  if (status != CL_SUCCESS || opencl->cmd_queue == NULL)
    goto fail;

fail:
  return (int)status;
}
