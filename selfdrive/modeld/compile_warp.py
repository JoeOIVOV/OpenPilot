#!/usr/bin/env python
from pathlib import Path
import time
from tinygrad.tensor import Tensor
import numpy as np


WARP_PKL_PATH = Path(__file__).parent / 'models/warp_tinygrad.pkl'

MODEL_WIDTH = 512
MODEL_HEIGHT = 256
MODEL_FRAME_SIZE = MODEL_WIDTH * MODEL_HEIGHT * 3 // 2
IMG_INPUT_SHAPE = (1, 12, 128, 256)

UV_SCALE_MATRIX = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
UV_SCALE_MATRIX_INV = np.linalg.inv(UV_SCALE_MATRIX)


def tensor_arange(end):
    return Tensor([float(i) for i in range(end)])

def tensor_round(tensor):
    return (tensor + 0.5).floor()


h_src, w_src = 1208, 1928
#h_dst, w_dst = MODEL_HEIGHT, MODEL_WIDTH

def warp_perspective_tinygrad(src, M_inv, dst_shape):
  w_dst, h_dst = dst_shape
  h_src, w_src = src.shape[:2]

  x = tensor_arange(w_dst).reshape(1, w_dst).expand(h_dst, w_dst)
  y = tensor_arange(h_dst).reshape(h_dst, 1).expand(h_dst, w_dst)
  ones = Tensor.ones_like(x)
  dst_coords = x.reshape(1, -1).cat(y.reshape(1, -1)).cat(ones.reshape(1, -1))  # (3, N)

  src_coords = M_inv @ dst_coords                      # (3, N)
  src_coords = src_coords / src_coords[2:3, :]         # divide by last row

  x_src = src_coords[0].reshape(h_dst, w_dst)
  y_src = src_coords[1].reshape(h_dst, w_dst)

  x_nn = tensor_round(x_src)
  y_nn = tensor_round(y_src)

  valid = (x_nn >= 0) & (x_nn < w_src) & (y_nn >= 0) & (y_nn < h_src)

  x_nn_clipped = x_nn.clip(0, w_src - 1).cast('int')
  y_nn_clipped = y_nn.clip(0, h_src - 1).cast('int')

  idx = (y_nn_clipped * w_src + x_nn_clipped).reshape(-1)   # (N,)

  src_flat = src.reshape(h_src * w_src)     # (H*W,)
  sampled = src_flat[idx]                   # (N,)

  valid_flat = valid.reshape(-1)
  zeros = Tensor.zeros_like(sampled)
  dst_flat = Tensor.where(valid_flat, sampled, zeros)

  return dst_flat.reshape(h_dst, w_dst)

def frames_to_tensor(frames):
  H = (frames.shape[0]*2)//3
  W = frames.shape[1]
  in_img1 = Tensor.cat(frames[0:H:2, 0::2],
                        frames[1:H:2, 0::2],
                        frames[0:H:2, 1::2],
                        frames[1:H:2, 1::2],
                        frames[H:H+H//4].reshape((H//2,W//2)),
                        frames[H+H//4:H+H//2].reshape((H//2,W//2)), dim=0).reshape((6, H//2, W//2))
  return in_img1

def frame_prepare_tinygrad(input_frame, M_inv, W=1928, H=1208):
  tg_scale = Tensor(UV_SCALE_MATRIX)
  M_inv_uv = tg_scale @ M_inv @ Tensor(UV_SCALE_MATRIX_INV)
  y = warp_perspective_tinygrad(input_frame[:H*W].reshape((H,W)), M_inv, (MODEL_WIDTH, MODEL_HEIGHT)).flatten()
  u = warp_perspective_tinygrad(input_frame[H*W::2].reshape((H//2,W//2)), M_inv_uv, (MODEL_WIDTH//2, MODEL_HEIGHT//2)).flatten()
  v = warp_perspective_tinygrad(input_frame[H*W+1::2].reshape((H//2,W//2)), M_inv_uv, (MODEL_WIDTH//2, MODEL_HEIGHT//2)).flatten()
  yuv = y.cat(u).cat(v).reshape((MODEL_HEIGHT*3//2,MODEL_WIDTH))
  tensor = frames_to_tensor(yuv)
  return tensor

def update_img_input_tinygrad(tensor, frame, M_inv):
  tensor_out = Tensor.cat(tensor[6:], frame_prepare_tinygrad(frame, M_inv), dim=0)
  return tensor_out, Tensor.cat(tensor_out[:6], tensor_out[-6:], dim=0)

def update_both_imgs_tinygrad(calib_img_buffer, new_img, M_inv,
                              calib_big_img_buffer, new_big_img, M_inv_big):
  calib_img_buffer, calib_img_pair = update_img_input_tinygrad(calib_img_buffer, new_img, M_inv)
  calib_big_img_buffer, calib_big_img_pair = update_img_input_tinygrad(calib_big_img_buffer, new_big_img, M_inv_big)
  return calib_img_buffer, calib_img_pair, calib_big_img_buffer, calib_big_img_pair

import numpy as np

def warp_perspective_numpy(src, M_inv, dst_shape):
    w_dst, h_dst = dst_shape
    h_src, w_src = src.shape[:2]
    xs, ys = np.meshgrid(np.arange(w_dst), np.arange(h_dst))  # shapes (h_dst, w_dst)
    ones = np.ones_like(xs)
    dst_hom = np.stack([xs, ys, ones], axis=0).reshape(3, -1)  # (3, N)

    # Map to source
    src_hom = M_inv @ dst_hom  # (3, N)
    src_hom /= src_hom[2:3, :]  # divide by last row (broadcast)

    x_src = src_hom[0, :]
    y_src = src_hom[1, :]

    # Nearest-neighbor sampling
    x_nn = np.round(x_src).astype(int)
    y_nn = np.round(y_src).astype(int)

    dst = np.zeros((h_dst, w_dst), dtype=src.dtype)

    # Keep only coordinates that fall inside the source image
    valid = (
        (x_nn >= 0) & (x_nn < w_src) &
        (y_nn >= 0) & (y_nn < h_src)
    )

    dst_x = xs.reshape(-1)[valid]
    dst_y = ys.reshape(-1)[valid]
    src_x = x_nn[valid]
    src_y = y_nn[valid]

    dst[dst_y, dst_x] = src[src_y, src_x]

    return dst

def frames_to_tensor_np(frames):
  H = (frames.shape[0]*2)//3
  W = frames.shape[1]
  p1 = frames[0:H:2, 0::2]
  p2 = frames[1:H:2, 0::2]
  p3 = frames[0:H:2, 1::2]
  p4 = frames[1:H:2, 1::2]
  p5 = frames[H:H+H//4].reshape((H//2, W//2))
  p6 = frames[H+H//4:H+H//2].reshape((H//2, W//2))
  return np.concatenate([p1, p2, p3, p4, p5, p6], axis=0)\
           .reshape((6, H//2, W//2))

def frame_prepare_np(input_frame, M_inv, W=1928, H=1208):
  M_inv_uv = UV_SCALE_MATRIX @ M_inv @ UV_SCALE_MATRIX_INV
  y  = warp_perspective_numpy(input_frame[:H*W].reshape(H, W),
                                 M_inv, (MODEL_WIDTH, MODEL_HEIGHT)).ravel()
  u  = warp_perspective_numpy(input_frame[H*W::2].reshape(H//2, W//2),
                                 M_inv_uv, (MODEL_WIDTH//2, MODEL_HEIGHT//2)).ravel()
  v  = warp_perspective_numpy(input_frame[H*W+1::2].reshape(H//2, W//2),
                                 M_inv_uv, (MODEL_WIDTH//2, MODEL_HEIGHT//2)).ravel()
  yuv = np.concatenate([y, u, v]).reshape( MODEL_HEIGHT*3//2, MODEL_WIDTH)
  return frames_to_tensor_np(yuv)

def update_img_input_np(tensor, frame, M_inv):
  tensor[:-6]  = tensor[6:]
  tensor[-6:] = frame_prepare_np(frame, M_inv)
  return tensor, np.concatenate([tensor[:6], tensor[-6:]], axis=0)

def update_both_imgs_np(calib_img_buffer, new_img, M_inv,
                        calib_big_img_buffer, new_big_img, M_inv_big):
  calib_img_buffer, calib_img_pair = update_img_input_np(calib_img_buffer, new_img, M_inv)
  calib_big_img_buffer, calib_big_img_pair = update_img_input_np(calib_big_img_buffer, new_big_img, M_inv_big)
  return calib_img_buffer, calib_img_pair, calib_big_img_buffer, calib_big_img_pair

def run_and_save_pickle(path):
  from tinygrad.engine.jit import TinyJit
  from tinygrad.device import Device
  update_img_jit = TinyJit(update_both_imgs_tinygrad, prune=True)
  #update_img_jit = update_both_imgs_tinygrad

  # run 20 times
  step_times = []
  for _ in range(20):
    img_inputs = [(32*Tensor.randn(30, 128, 256) + 128).cast(dtype='uint8').realize(), (32*Tensor.randn(1928*1208*3//2) + 128).cast(dtype='uint8').realize(), Tensor.randn(3,3).realize()]
    big_img_inputs = [(32*Tensor.randn(30, 128, 256) + 128).cast(dtype='uint8').realize(), (32*Tensor.randn(1928*1208*3//2) + 128).cast(dtype='uint8').realize(), Tensor.randn(3,3).realize()]
    inputs = img_inputs + big_img_inputs
    Device.default.synchronize()
    inputs_np = [x.numpy() for x in inputs]
    st = time.perf_counter()
    out = update_img_jit(*inputs)
    mt = time.perf_counter()
    Device.default.synchronize()
    et = time.perf_counter()
    step_times.append((et-st)*1e3)
    print(f"enqueue {(mt-st)*1e3:6.2f} ms -- total run {step_times[-1]:6.2f} ms")
    out_np = update_both_imgs_np(*inputs_np)
    np.testing.assert_allclose(out_np[0][0], out[0][0].numpy())

    for a, b in zip(out_np, (x.numpy() for x in out)):
      mismatch = np.abs(a - b) > 0
      mismatch_percent = sum(mismatch.flatten()) / len(mismatch.flatten()) * 100
      mismatch_percent_tol = 1e-2
      assert mismatch_percent < mismatch_percent_tol, f"input mismatch percent {mismatch_percent} exceeds tolerance {mismatch_percent_tol}"


  import pickle
  with open(path, "wb") as f:
    pickle.dump(update_img_jit, f)

if __name__ == "__main__":
    run_and_save_pickle(WARP_PKL_PATH)
