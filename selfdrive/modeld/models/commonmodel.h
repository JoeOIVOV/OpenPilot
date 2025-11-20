#pragma once

#include <cfloat>
#include <cstdlib>
#include <cassert>

#include <memory>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "common/mat.h"
#include "selfdrive/modeld/transforms/loadyuv.h"

class ModelFrame {
public:
  ModelFrame(cl_device_id device_id, cl_context context) {
    q = CL_CHECK_ERR(clCreateCommandQueue(context, device_id, 0, &err));
  }
  virtual ~ModelFrame() {}

  int MODEL_WIDTH;
  int MODEL_HEIGHT;
  int MODEL_FRAME_SIZE;
  int buf_size;
  uint8_t* array_from_vision_buf(cl_mem *vision_buf);
  cl_mem* cl_from_vision_buf(cl_mem *vision_buf);

  // DONT HARDCODE THIS
  const int RAW_IMG_HEIGHT = 1208;
  const int RAW_IMG_WIDTH = 1928;
  const int full_img_size = RAW_IMG_HEIGHT * RAW_IMG_WIDTH * 3 / 2;

protected:
  cl_command_queue q;
  cl_mem single_frame_cl;
  std::unique_ptr<uint8_t[]> full_input_frame;
};

class DrivingModelFrame : public ModelFrame {
public:
  DrivingModelFrame(cl_device_id device_id, cl_context context, int _temporal_skip);
  ~DrivingModelFrame();

  const int MODEL_WIDTH = 512;
  const int MODEL_HEIGHT = 256;
  const int MODEL_FRAME_SIZE = MODEL_WIDTH * MODEL_HEIGHT * 3 / 2;
  const int buf_size = MODEL_FRAME_SIZE * 2; // 2 frames are temporal_skip frames apart

  const size_t frame_size_bytes = MODEL_FRAME_SIZE * sizeof(uint8_t);

};

class MonitoringModelFrame : public ModelFrame {
public:
  MonitoringModelFrame(cl_device_id device_id, cl_context context);
  ~MonitoringModelFrame();

  const int MODEL_WIDTH = 1440;
  const int MODEL_HEIGHT = 960;
  const int MODEL_FRAME_SIZE = MODEL_WIDTH * MODEL_HEIGHT;
  const int buf_size = MODEL_FRAME_SIZE;

private:
  cl_mem input_frame_cl;
};
