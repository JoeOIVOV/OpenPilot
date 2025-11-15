from pathlib import Path
import time
from tinygrad.tensor import Tensor

WARP_PKL_PATH = Path(__file__).parent / 'models/warp_tinygrad.pkl'
WARP_BIG_PKL_PATH = Path(__file__).parent / 'models/warp_big_tinygrad.pkl'

MODEL_WIDTH = 512
MODEL_HEIGHT = 256
MODEL_FRAME_SIZE = MODEL_WIDTH * MODEL_HEIGHT * 3 // 2
IMG_INPUT_SHAPE = (1, 12, 128, 256)


def tensor_arange(end):
    return Tensor([float(i) for i in range(end)])

def tensor_round(tensor):
    return (tensor + 0.5).floor()


h_src, w_src = 1208, 1928
#h_dst, w_dst = MODEL_HEIGHT, MODEL_WIDTH

def warp_perspective_tinygrad(src, M_inv, dst_shape):
  w_dst, h_dst = dst_shape
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

def frame_prepare_tinygrad(input_frame, M_inv, M_inv_uv, W, H):
  y = warp_perspective_tinygrad(input_frame[:H*W].reshape((H,W)), M_inv, (MODEL_WIDTH, MODEL_HEIGHT)).flatten()
  u = warp_perspective_tinygrad(input_frame[H*W::2].reshape((H//2,W//2)), M_inv_uv, (MODEL_WIDTH//2, MODEL_HEIGHT//2)).flatten()
  v = warp_perspective_tinygrad(input_frame[H*W+1::2].reshape((H//2,W//2)), M_inv_uv, (MODEL_WIDTH//2, MODEL_HEIGHT//2)).flatten()
  yuv = y.cat(u).cat(v).reshape((1,MODEL_HEIGHT*3//2,MODEL_WIDTH))
  tensor = frames_to_tensor(yuv)
  return tensor

def update_img_input_tinygrad(tensor, frame, M_inv, M_inv_uv, w, h):
  tensor[:,:6] = tensor[:,-6:]
  tensor[:,-6:] = frame_prepare_tinygrad(frame, M_inv, M_inv_uv, w, h)
  return tensor, Tensor.cat(tensor[:,:6], tensor[:,-6:], dim=1)



def run_and_save_pickle(path):
  from tinygrad.engine.jit import TinyJit
  from tinygrad.device import Device
  update_img_jit = TinyJit(update_img_input_tinygrad, prune=True)

  inputs = [Tensor.randn((1, 30, 128, 256), dtype='uint8').realize(), Tensor.randn(1928*1208*3//2, dtype='uint8').realize(), Tensor.randn(3,3).realize(), Tensor.randn(3,3).realize(), 1928, 1208]
  # run 20 times
  step_times = []
  for _ in range(20):
    st = time.perf_counter()
    out = update_img_jit(*inputs)
    mt = time.perf_counter()
    Device.default.synchronize()
    et = time.perf_counter()
    step_times.append((et-st)*1e3)
    print(f"enqueue {(mt-st)*1e3:6.2f} ms -- total run {step_times[-1]:6.2f} ms")

  import pickle
  with open(path, "wb") as f:
    pickle.dump(update_img_jit, f)

if __name__ == "__main__":
    run_and_save_pickle(WARP_PKL_PATH)
    run_and_save_pickle(WARP_BIG_PKL_PATH)