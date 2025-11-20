#include "selfdrive/modeld/models/commonmodel.h"

#include <cmath>
#include <cstring>

#include "common/clutil.h"

DrivingModelFrame::DrivingModelFrame(cl_device_id device_id, cl_context context, int _temporal_skip) : ModelFrame(device_id, context) {

  full_input_frame = std::make_unique<uint8_t[]>(full_img_size);
  input_frames = std::make_unique<uint8_t[]>(buf_size);
  input_frames_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, buf_size, NULL, &err));

  input_frames = std::make_unique<uint8_t[]>(buf_size);
  temporal_skip = _temporal_skip;
  input_frames_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, buf_size, NULL, &err));
  single_frame_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, full_img_size, NULL, &err));
  img_buffer_20hz_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, (temporal_skip+1)*frame_size_bytes, NULL, &err));
  region.origin = temporal_skip * frame_size_bytes;
  region.size = frame_size_bytes;
  last_img_cl = CL_CHECK_ERR(clCreateSubBuffer(img_buffer_20hz_cl, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err));

}

uint8_t* ModelFrame::array_from_vision_buf(cl_mem *vision_buf) {
  CL_CHECK(clEnqueueReadBuffer(q, *vision_buf, CL_TRUE, 0, full_img_size * sizeof(uint8_t), &full_input_frame[0], 0, nullptr, nullptr));
  clFinish(q);
  return &full_input_frame[0];
}

cl_mem* ModelFrame::cl_from_vision_buf(cl_mem *vision_buf) {
  CL_CHECK(clEnqueueCopyBuffer(q, *vision_buf, single_frame_cl,  0, 0, full_img_size * sizeof(uint8_t), 0, nullptr, nullptr));
  clFinish(q);
  return &single_frame_cl;
}

  
DrivingModelFrame::~DrivingModelFrame() {
  CL_CHECK(clReleaseMemObject(input_frames_cl));
  CL_CHECK(clReleaseMemObject(img_buffer_20hz_cl));
  CL_CHECK(clReleaseMemObject(last_img_cl));
  CL_CHECK(clReleaseMemObject(single_frame_cl));
  CL_CHECK(clReleaseCommandQueue(q));
}


MonitoringModelFrame::MonitoringModelFrame(cl_device_id device_id, cl_context context) : ModelFrame(device_id, context) {
  input_frames = std::make_unique<uint8_t[]>(buf_size);
  input_frame_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, buf_size, NULL, &err));
  full_input_frame = std::make_unique<uint8_t[]>(full_img_size);
}


MonitoringModelFrame::~MonitoringModelFrame() {
  CL_CHECK(clReleaseMemObject(input_frame_cl));
  CL_CHECK(clReleaseCommandQueue(q));
}
