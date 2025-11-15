#!/usr/bin/env python3
import os
from openpilot.system.hardware import TICI
os.environ['DEV'] = 'QCOM' if TICI else 'CPU'
USBGPU = "USBGPU" in os.environ
if USBGPU:
  os.environ['DEV'] = 'AMD'
  os.environ['AMD_IFACE'] = 'USB'
from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
import time
import pickle
import numpy as np
import cereal.messaging as messaging
from cereal import car, log
from pathlib import Path
from cereal.messaging import PubMaster, SubMaster
from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from opendbc.car.car_helpers import get_demo_car_params
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, smooth_value, get_curvature_from_plan
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.fill_model_msg import fill_model_msg, fill_pose_msg, PublishState
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.models.commonmodel_pyx import DrivingModelFrame, CLContext
from openpilot.selfdrive.modeld.runners.tinygrad_helpers import qcom_tensor_from_opencl_address


from tinygrad.tensor import Tensor
import ctypes, array
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv, to_mv, mv_address
from tinygrad.device import Device

Tensor.manual_seed(1337)
Tensor.no_grad = True

PROCESS_NAME = "selfdrive.modeld.modeld"
SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')

VISION_PKL_PATH = Path(__file__).parent / 'models/driving_vision_tinygrad.pkl'
POLICY_PKL_PATH = Path(__file__).parent / 'models/driving_policy_tinygrad.pkl'
VISION_METADATA_PATH = Path(__file__).parent / 'models/driving_vision_metadata.pkl'
POLICY_METADATA_PATH = Path(__file__).parent / 'models/driving_policy_metadata.pkl'

LAT_SMOOTH_SECONDS = 0.1
LONG_SMOOTH_SECONDS = 0.3
MIN_LAT_CONTROL_SPEED = 0.3

MODEL_WIDTH = 512
MODEL_HEIGHT = 256
MODEL_FRAME_SIZE = MODEL_WIDTH * MODEL_HEIGHT * 3 // 2
IMG_INPUT_SHAPE = (1, 30, 128, 256)



def tensor_arange(end):
    return Tensor([float(i) for i in range(end)])

def tensor_round(tensor):
    return (tensor + 0.5).floor()

def warp_perspective_tinygrad(src, M_inv, dsize):
    h_dst, w_dst = dsize[1], dsize[0]
    h_src, w_src = src.shape[:2]

    x = tensor_arange(w_dst).reshape(1, w_dst).expand(h_dst, w_dst)
    y = tensor_arange(h_dst).reshape(h_dst, 1).expand(h_dst, w_dst)
    ones = Tensor.ones_like(x)
    dst_coords = x.reshape((1,-1)).cat(y.reshape((1,-1))).cat(ones.reshape((1,-1)))


    src_coords = M_inv @ dst_coords
    src_coords = src_coords / src_coords[2:3, :]

    x_src = src_coords[0].reshape(h_dst, w_dst)
    y_src = src_coords[1].reshape(h_dst, w_dst)

    x_nearest = tensor_round(x_src).clip(0, w_src - 1).cast('int')
    y_nearest = tensor_round(y_src).clip(0, h_src - 1).cast('int')

    # TODO: make 2d indexing fast
    idx = y_nearest*src.shape[1] + x_nearest
    dst = src.flatten()[idx]
    return dst.reshape(h_dst, w_dst)


def frame_prepare_tinygrad(input_frame, M_inv, M_inv_uv, W, H):
  y = warp_perspective_tinygrad(input_frame[:H*W].reshape((H,W)), M_inv, (MODEL_WIDTH, MODEL_HEIGHT)).flatten()
  u = warp_perspective_tinygrad(input_frame[H*W::2].reshape((H//2,W//2)), M_inv_uv, (MODEL_WIDTH//2, MODEL_HEIGHT//2)).flatten()
  v = warp_perspective_tinygrad(input_frame[H*W+1::2].reshape((H//2,W//2)), M_inv_uv, (MODEL_WIDTH//2, MODEL_HEIGHT//2)).flatten()
  yuv = y.cat(u).cat(v).reshape((1,MODEL_HEIGHT*3//2,MODEL_WIDTH))
  tensor = frames_to_tensor(yuv)
  print(tensor.shape)
  return tensor

def update_img_input_tinygrad(tensor, frame, M_inv, M_inv_uv, w, h):
  tensor[:,:6] = tensor[:,-6:]
  tensor[:,-6:] = frame_prepare_tinygrad(frame, M_inv, M_inv_uv, w, h)
  return tensor, Tensor.cat(tensor[:,:6], tensor[:,-6:], dim=1)


def frames_to_tensor(frames):
  H = (frames.shape[1]*2)//3
  W = frames.shape[2]
  in_img1 = Tensor.cat(frames[:, 0:H:2, 0::2],
                        frames[:, 1:H:2, 0::2],
                        frames[:, 0:H:2, 1::2],
                        frames[:, 1:H:2, 1::2],
                        frames[:, H:H+H//4].reshape((-1, H//2,W//2)),
                        frames[:, H+H//4:H+H//2].reshape((-1, H//2,W//2)), dim=1).reshape((frames.shape[0], 6, H//2, W//2))
  return in_img1


def get_action_from_model(model_output: dict[str, np.ndarray], prev_action: log.ModelDataV2.Action,
                          lat_action_t: float, long_action_t: float, v_ego: float) -> log.ModelDataV2.Action:
    plan = model_output['plan'][0]
    desired_accel, should_stop = get_accel_from_plan(plan[:,Plan.VELOCITY][:,0],
                                                     plan[:,Plan.ACCELERATION][:,0],
                                                     ModelConstants.T_IDXS,
                                                     action_t=long_action_t)
    desired_accel = smooth_value(desired_accel, prev_action.desiredAcceleration, LONG_SMOOTH_SECONDS)

    desired_curvature = get_curvature_from_plan(plan[:,Plan.T_FROM_CURRENT_EULER][:,2],
                                                plan[:,Plan.ORIENTATION_RATE][:,2],
                                                ModelConstants.T_IDXS,
                                                v_ego,
                                                lat_action_t)
    if v_ego > MIN_LAT_CONTROL_SPEED:
      desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, LAT_SMOOTH_SECONDS)
    else:
      desired_curvature = prev_action.desiredCurvature

    return log.ModelDataV2.Action(desiredCurvature=float(desired_curvature),
                                  desiredAcceleration=float(desired_accel),
                                  shouldStop=bool(should_stop))

class FrameMeta:
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def __init__(self, vipc=None):
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof

class InputQueues:
  def __init__ (self, model_fps, env_fps, n_frames_input):
    assert env_fps % model_fps == 0
    assert env_fps >= model_fps
    self.model_fps = model_fps
    self.env_fps = env_fps
    self.n_frames_input = n_frames_input

    self.dtypes = {}
    self.shapes = {}
    self.q = {}

  def update_dtypes_and_shapes(self, input_dtypes, input_shapes) -> None:
    self.dtypes.update(input_dtypes)
    if self.env_fps == self.model_fps:
      self.shapes.update(input_shapes)
    else:
      for k in input_shapes:
        shape = list(input_shapes[k])
        if 'img' in k:
          n_channels = shape[1] // self.n_frames_input
          shape[1] = (self.env_fps // self.model_fps + (self.n_frames_input - 1)) * n_channels
        else:
          shape[1] = (self.env_fps // self.model_fps) * shape[1]
        self.shapes[k] = tuple(shape)

  def reset(self) -> None:
    self.q = {k: np.zeros(self.shapes[k], dtype=self.dtypes[k]) for k in self.dtypes.keys()}

  def enqueue(self, inputs:dict[str, np.ndarray]) -> None:
    for k in inputs.keys():
      if inputs[k].dtype != self.dtypes[k]:
        raise ValueError(f'supplied input <{k}({inputs[k].dtype})> has wrong dtype, expected {self.dtypes[k]}')
      input_shape = list(self.shapes[k])
      input_shape[1] = -1
      single_input = inputs[k].reshape(tuple(input_shape))
      sz = single_input.shape[1]
      self.q[k][:,:-sz] = self.q[k][:,sz:]
      self.q[k][:,-sz:] = single_input

  def get(self, *names) -> dict[str, np.ndarray]:
    if self.env_fps == self.model_fps:
      return {k: self.q[k] for k in names}
    else:
      out = {}
      for k in names:
        shape = self.shapes[k]
        if 'img' in k:
          n_channels = shape[1] // (self.env_fps // self.model_fps + (self.n_frames_input - 1))
          out[k] = np.concatenate([self.q[k][:, s:s+n_channels] for s in np.linspace(0, shape[1] - n_channels, self.n_frames_input, dtype=int)], axis=1)
        elif 'pulse' in k:
          # any pulse within interval counts
          out[k] = self.q[k].reshape((shape[0], shape[1] * self.model_fps // self.env_fps, self.env_fps // self.model_fps, -1)).max(axis=2)
        else:
          idxs = np.arange(-1, -shape[1], -self.env_fps // self.model_fps)[::-1]
          out[k] = self.q[k][:, idxs]
      return out

class ModelState:
  frames: dict[str, DrivingModelFrame]
  inputs: dict[str, np.ndarray]
  output: np.ndarray
  prev_desire: np.ndarray  # for tracking the rising edge of the pulse

  def __init__(self, context: CLContext):
    with open(VISION_METADATA_PATH, 'rb') as f:
      vision_metadata = pickle.load(f)
      self.vision_input_shapes =  vision_metadata['input_shapes']
      self.vision_input_names = list(self.vision_input_shapes.keys())
      self.vision_output_slices = vision_metadata['output_slices']
      vision_output_size = vision_metadata['output_shapes']['outputs'][1]

    with open(POLICY_METADATA_PATH, 'rb') as f:
      policy_metadata = pickle.load(f)
      self.policy_input_shapes =  policy_metadata['input_shapes']
      self.policy_output_slices = policy_metadata['output_slices']
      policy_output_size = policy_metadata['output_shapes']['outputs'][1]

    self.frames = {name: DrivingModelFrame(context, ModelConstants.MODEL_RUN_FREQ//ModelConstants.MODEL_CONTEXT_FREQ) for name in self.vision_input_names}
    self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)

    # policy inputs
    self.numpy_inputs = {k: np.zeros(self.policy_input_shapes[k], dtype=np.float32) for k in self.policy_input_shapes}
    self.full_input_queues = InputQueues(ModelConstants.MODEL_CONTEXT_FREQ, ModelConstants.MODEL_RUN_FREQ, ModelConstants.N_FRAMES)
    for k in ['desire_pulse', 'features_buffer']:
      self.full_input_queues.update_dtypes_and_shapes({k: self.numpy_inputs[k].dtype}, {k: self.numpy_inputs[k].shape})
    self.full_input_queues.reset()

    # img buffers are managed in openCL transform code
    self.full_img_input = {'img': Tensor.zeros(IMG_INPUT_SHAPE, dtype='uint8').contiguous().realize(),
                                             'big_img': Tensor.zeros(IMG_INPUT_SHAPE, dtype='uint8').contiguous().realize(),}
    self.vision_output = np.zeros(vision_output_size, dtype=np.float32)
    self.policy_inputs = {k: Tensor(v, device='NPY').realize() for k,v in self.numpy_inputs.items()}
    self.policy_output = np.zeros(policy_output_size, dtype=np.float32)
    self.parser = Parser()

    self.update_img_jit = None

    with open(VISION_PKL_PATH, "rb") as f:
      self.vision_run = pickle.load(f)

    with open(POLICY_PKL_PATH, "rb") as f:
      self.policy_run = pickle.load(f)

  def slice_outputs(self, model_outputs: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
    parsed_model_outputs = {k: model_outputs[np.newaxis, v] for k,v in output_slices.items()}
    return parsed_model_outputs

  def run(self, bufs: dict[str, VisionBuf], transforms: dict[str, np.ndarray],
                inputs: dict[str, np.ndarray], prepare_only: bool) -> dict[str, np.ndarray] | None:
    # Model decides when action is completed, so desire input is just a pulse triggered on rising edge
    inputs['desire_pulse'][0] = 0
    new_desire = np.where(inputs['desire_pulse'] - self.prev_desire > .99, inputs['desire_pulse'], 0)
    self.prev_desire[:] = inputs['desire_pulse']

    #imgs_cl = {name: self.frames[name].prepare(bufs[name], transforms[name].flatten()) for name in self.vision_input_names}

    #if TICI and not USBGPU:
    #  # The imgs tensors are backed by opencl memory, only need init once
    #  for key in imgs_cl:
    #    if key not in self.vision_inputs:
    #      self.vision_inputs[key] = qcom_tensor_from_opencl_address(imgs_cl[key].mem_address, self.vision_input_shapes[key], dtype=dtypes.uint8)
    #else:
    #  for key in imgs_cl:
    #    frame_input = self.frames[key].buffer_from_cl(imgs_cl[key]).reshape(self.vision_input_shapes[key])
    #    self.vision_inputs[key] = Tensor(frame_input, dtype=dtypes.uint8).realize()


    #for k, v in self.numpy_inputs.items():
    #   self.vision_inputs[k] = Tensor(v)

    #assert False, transforms.keys()
    vision_inputs = {}
    if self.update_img_jit is None:
      self.update_img_jit = TinyJit(update_img_input_tinygrad, prune=True)
    
    for key in bufs.keys():
      scale_matrix = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
      transform = transforms[key]
      M_inv = Tensor(transform, dtype=dtypes.float32)
      M_inv_uv = Tensor(scale_matrix @ transform @ np.linalg.inv(scale_matrix), dtype=dtypes.float32)

      frame = Tensor(self.frames[key].array_from_vision_buf(bufs[key]))

      t0 = time.perf_counter()
      
      self.full_img_input[key], vision_inputs[key] = self.update_img_jit(self.full_img_input[key], frame, M_inv, M_inv_uv, bufs[key].width, bufs[key].height)
      vision_inputs[key] = vision_inputs[key].clone()
      Device.default.synchronize()
      t1 = time.perf_counter()
      print(f"update_img_jit took {(t1 - t0) * 1000:.2f} ms")

    if prepare_only:
      return None

    self.vision_output = self.vision_run(**vision_inputs).contiguous().realize().uop.base.buffer.numpy()
    vision_outputs_dict = self.parser.parse_vision_outputs(self.slice_outputs(self.vision_output, self.vision_output_slices))

    self.full_input_queues.enqueue({'features_buffer': vision_outputs_dict['hidden_state'], 'desire_pulse': new_desire})
    for k in ['desire_pulse', 'features_buffer']:
      self.numpy_inputs[k][:] = self.full_input_queues.get(k)[k]
    self.numpy_inputs['traffic_convention'][:] = inputs['traffic_convention']

    self.policy_output = self.policy_run(**self.policy_inputs).contiguous().realize().uop.base.buffer.numpy()
    policy_outputs_dict = self.parser.parse_policy_outputs(self.slice_outputs(self.policy_output, self.policy_output_slices))

    combined_outputs_dict = {**vision_outputs_dict, **policy_outputs_dict}
    if SEND_RAW_PRED:
      combined_outputs_dict['raw_pred'] = np.concatenate([self.vision_output.copy(), self.policy_output.copy()])

    return combined_outputs_dict


def main(demo=False):
  cloudlog.warning("modeld init")

  if not USBGPU:
    # USB GPU currently saturates a core so can't do this yet,
    # also need to move the aux USB interrupts for good timings
    config_realtime_process(7, 54)

  st = time.monotonic()
  cloudlog.warning("setting up CL context")
  cl_context = CLContext()
  cloudlog.warning("CL context ready; loading model")
  model = ModelState(cl_context)
  cloudlog.warning(f"models loaded in {time.monotonic() - st:.1f}s, modeld starting")

  # visionipc clients
  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      use_extra_client = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and VisionStreamType.VISION_STREAM_ROAD in available_streams
      main_wide_camera = VisionStreamType.VISION_STREAM_ROAD not in available_streams
      break
    time.sleep(.1)

  vipc_client_main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_ROAD
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True, cl_context)
  vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False, cl_context)
  cloudlog.warning(f"vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  cloudlog.warning(f"connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
  if use_extra_client:
    cloudlog.warning(f"connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")

  # messaging
  pm = PubMaster(["modelV2", "drivingModelData", "cameraOdometry"])
  sm = SubMaster(["deviceState", "carState", "roadCameraState", "liveCalibration", "driverMonitoringState", "carControl", "liveDelay"])

  publish_state = PublishState()
  params = Params()

  # setup filter to track dropped frames
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / ModelConstants.MODEL_RUN_FREQ)
  frame_id = 0
  last_vipc_frame_id = 0
  run_count = 0

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  live_calib_seen = False
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()


  if demo:
    CP = get_demo_car_params()
  else:
    CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("modeld got CarParams: %s", CP.brand)

  # TODO this needs more thought, use .2s extra for now to estimate other delays
  # TODO Move smooth seconds to action function
  long_delay = CP.longitudinalActuatorDelay + LONG_SMOOTH_SECONDS
  prev_action = log.ModelDataV2.Action()

  DH = DesireHelper()

  while True:
    # Keep receiving frames until we are at least 1 frame ahead of previous extra frame
    while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
      buf_main = vipc_client_main.recv()
      meta_main = FrameMeta(vipc_client_main)
      if buf_main is None:
        break

    if buf_main is None:
      cloudlog.debug("vipc_client_main no frame")
      continue

    if use_extra_client:
      # Keep receiving extra frames until frame id matches main camera
      while True:
        buf_extra = vipc_client_extra.recv()
        meta_extra = FrameMeta(vipc_client_extra)
        if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
          break

      if buf_extra is None:
        cloudlog.debug("vipc_client_extra no frame")
        continue

      if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
        cloudlog.error(f"frames out of sync! main: {meta_main.frame_id} ({meta_main.timestamp_sof / 1e9:.5f}),\
                         extra: {meta_extra.frame_id} ({meta_extra.timestamp_sof / 1e9:.5f})")

    else:
      # Use single camera
      buf_extra = buf_main
      meta_extra = meta_main

    sm.update(0)
    desire = DH.desire
    is_rhd = sm["driverMonitoringState"].isRHD
    frame_id = sm["roadCameraState"].frameId
    v_ego = max(sm["carState"].vEgo, 0.)
    lat_delay = sm["liveDelay"].lateralDelay + LAT_SMOOTH_SECONDS
    if sm.updated["liveCalibration"] and sm.seen['roadCameraState'] and sm.seen['deviceState']:
      device_from_calib_euler = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)
      dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))]
      model_transform_main = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics if main_wide_camera else dc.fcam.intrinsics, False).astype(np.float32)
      model_transform_extra = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics, True).astype(np.float32)
      live_calib_seen = True

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1

    vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < ModelConstants.DESIRE_LEN:
      vec_desire[desire] = 1

    # tracked dropped frames
    vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
    frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
    if run_count < 10: # let frame drops warm up
      frame_dropped_filter.x = 0.
      frames_dropped = 0.
    run_count = run_count + 1

    frame_drop_ratio = frames_dropped / (1 + frames_dropped)
    prepare_only = vipc_dropped_frames > 0
    if prepare_only:
      cloudlog.error(f"skipping model eval. Dropped {vipc_dropped_frames} frames")

    bufs = {name: buf_extra if 'big' in name else buf_main for name in model.vision_input_names}
    transforms = {name: model_transform_extra if 'big' in name else model_transform_main for name in model.vision_input_names}
    inputs:dict[str, np.ndarray] = {
      'desire_pulse': vec_desire,
      'traffic_convention': traffic_convention,
    }

    mt1 = time.perf_counter()
    model_output = model.run(bufs, transforms, inputs, prepare_only)
    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1

    if model_output is not None:
      modelv2_send = messaging.new_message('modelV2')
      drivingdata_send = messaging.new_message('drivingModelData')
      posenet_send = messaging.new_message('cameraOdometry')

      action = get_action_from_model(model_output, prev_action, lat_delay + DT_MDL, long_delay + DT_MDL, v_ego)
      prev_action = action
      fill_model_msg(drivingdata_send, modelv2_send, model_output, action,
                     publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id,
                     frame_drop_ratio, meta_main.timestamp_eof, model_execution_time, live_calib_seen)

      desire_state = modelv2_send.modelV2.meta.desireState
      l_lane_change_prob = desire_state[log.Desire.laneChangeLeft]
      r_lane_change_prob = desire_state[log.Desire.laneChangeRight]
      lane_change_prob = l_lane_change_prob + r_lane_change_prob
      DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob)
      modelv2_send.modelV2.meta.laneChangeState = DH.lane_change_state
      modelv2_send.modelV2.meta.laneChangeDirection = DH.lane_change_direction
      drivingdata_send.drivingModelData.meta.laneChangeState = DH.lane_change_state
      drivingdata_send.drivingModelData.meta.laneChangeDirection = DH.lane_change_direction

      fill_pose_msg(posenet_send, model_output, meta_main.frame_id, vipc_dropped_frames, meta_main.timestamp_eof, live_calib_seen)
      pm.send('modelV2', modelv2_send)
      pm.send('drivingModelData', drivingdata_send)
      pm.send('cameraOdometry', posenet_send)
    last_vipc_frame_id = meta_main.frame_id


if __name__ == "__main__":
  try:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='A boolean for demo mode.')
    args = parser.parse_args()
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning("got SIGINT")
