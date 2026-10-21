[Skip to content](https://github.com/NVIDIA/TensorRT/issues/3640#start-of-content)
You signed in with another tab or window. [Reload](https://github.com/NVIDIA/TensorRT/issues/3640) to refresh your session. You signed out in another tab or window. [Reload](https://github.com/NVIDIA/TensorRT/issues/3640) to refresh your session. You switched accounts on another tab or window. [Reload](https://github.com/NVIDIA/TensorRT/issues/3640) to refresh your session. Dismiss alert
#  Could not find any implementation for node /model.0/conv/Conv + PWN(PWN(/model.0/act/Sigmoid), /model.0/act/Mul)
Copy link
Copy link
[Could not find any implementation for node /model.0/conv/Conv + PWN(PWN(/model.0/act/Sigmoid), /model.0/act/Mul)](https://github.com/NVIDIA/TensorRT/issues/3640#top)#3640
Labels
[Issue has been triaged by maintainers](https://github.com/NVIDIA/TensorRT/issues?q=state%3Aopen%20label%3A%22triaged%22)Issue has been triaged by maintainers
## Description
Issue body actions
## Description
INT8 quantization fails with error in title on Jetson AGX Orin in latest JetPack 6.0 container. I also tried this previously on JetPack 5.1.2 and it failed with the same issue so I tried the latest version of TensorRT 8.6.2. Can I get some help with this? Thanks. Full log below.
Details ONNX: starting export with onnx 1.16.0 opset 17... ONNX: simplifying with onnxsim 0.4.33... ONNX: export success ✅ 7.5s, saved as 'model.onnx' (137.0 MB) 
TensorRT: starting export with TensorRT 8.6.2... [01/29/2024-20:52:28] [TRT] [I] [MemUsageChange] Init CUDA: CPU +2, GPU +0, now: CPU 995, GPU 15955 (MiB) [01/29/2024-20:52:34] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU +1153, GPU +722, now: CPU 2184, GPU 16678 (MiB) [01/29/2024-20:52:34] [TRT] [I] ---------------------------------------------------------------- [01/29/2024-20:52:34] [TRT] [I] Input filename: model.onnx [01/29/2024-20:52:34] [TRT] [I] ONNX IR version: 0.0.8 [01/29/2024-20:52:34] [TRT] [I] Opset version: 17 [01/29/2024-20:52:34] [TRT] [I] Producer name: pytorch [01/29/2024-20:52:34] [TRT] [I] Producer version: 2.1.0 [01/29/2024-20:52:34] [TRT] [I] Domain: [01/29/2024-20:52:34] [TRT] [I] Model version: 0 [01/29/2024-20:52:34] [TRT] [I] Doc string: [01/29/2024-20:52:34] [TRT] [I] ---------------------------------------------------------------- [01/29/2024-20:52:34] [TRT] [W] onnx2trt_utils.cpp:372: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32. TensorRT: input "images" with shape(1, 3, 960, 1280) DataType.HALF TensorRT: output "output0" with shape(1, 40, 25200) DataType.HALF TensorRT: output "output1" with shape(1, 32, 240, 320) DataType.HALF 2024-01-29 20:52:34 - calibrator - INFO - Skipping calibration files, using calibration cache: calibration.cache TensorRT: building INT8 engine as model.engine [01/29/2024-20:52:35] [TRT] [W] The CUDA context changed between createInferBuilder and buildSerializedNetwork. A Builder holds CUDA resources which cannot be shared across CUDA contexts, so access these in different CUDA context results in undefined behavior. If using pycuda, try import pycuda.autoinit before importing tensorrt. [01/29/2024-20:52:40] [TRT] [I] Graph optimization time: 0.0168623 seconds. 2024-01-29 20:52:40 - calibrator - INFO - Using calibration cache to save time: calibration.cache [01/29/2024-20:52:40] [TRT] [I] Reading Calibration Cache for calibrator: EntropyCalibration2 [01/29/2024-20:52:40] [TRT] [I] Generated calibration scales using calibration cache. Make sure that calibration cache has latest scales. [01/29/2024-20:52:40] [TRT] [I] To regenerate calibration cache, please delete the existing one. TensorRT will generate a new calibration cache. 2024-01-29 20:52:41 - calibrator - INFO - Using calibration cache to save time: calibration.cache [01/29/2024-20:52:41] [TRT] [W] Missing scale and zero-point for tensor /model.22/dfl/Softmax_output_0, expect fall back to non-int8 implementation for any layer consuming or producing given tensor [01/29/2024-20:52:41] [TRT] [W] Missing scale and zero-point for tensor (Unnamed Layer* 405) [Constant]_output, expect fall back to non-int8 implementation for any layer consuming or producing given tensor [01/29/2024-20:52:46] [TRT] [I] Graph optimization time: 0.193492 seconds. [01/29/2024-20:52:46] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored. [01/29/2024-20:52:47] [TRT] [E] 10: Could not find any implementation for node /model.0/conv/Conv + PWN(PWN(/model.0/act/Sigmoid), /model.0/act/Mul). [01/29/2024-20:52:47] [TRT] [E] 10: [optimizer.cpp::computeCosts::3869] Error Code 10: Internal Error (Could not find any implementation for node /model.0/conv/Conv + PWN(PWN(/model.0/act/Sigmoid), /model.0/act/Mul).) TensorRT: export failure ❌ 26.5s: **enter** Traceback (most recent call last): File "/workspace/tensorrt_export.py", line 33, in main() File "/workspace/tensorrt_export.py", line 29, in main export(model=model, format='engine', imgsz=(960,1280), workspace=24, half=True, int8=True, opset=17) File "/workspace/tensorrt_export.py", line 16, in export return Exporter(overrides=args, _callbacks=model.callbacks)(model=model.model) File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 115, in decorate_context return func(*args, **kwargs) File "/workspace/int8_exporter.py", line 254, in **call** f[1], _ = self.export_engine() File "/workspace/int8_exporter.py", line 124, in outer_func raise e File "/workspace/int8_exporter.py", line 119, in outer_func f, model = inner_func(*args, **kwargs) File "/workspace/int8_exporter.py", line 421, in export_engine with builder.build_engine(network, config) as engine, open(f, 'wb') as t: AttributeError: **enter**
## Environment
**TensorRT Version** : 8.6.2
**NVIDIA GPU** : Ampere on Jetson AGX Orin
**NVIDIA Driver Version** :
**CUDA Version** : 12.2
**CUDNN Version** :
Operating System:
Python Version (if applicable): 3.10.12
PyTorch Version (if applicable): 2.1.0
Baremetal or Container (if so, version): [dustynv/l4t-pytorch:r36.2.0](https://hub.docker.com/r/dustynv/l4t-pytorch/tags)
Reactions are currently unavailable
## Activity
  1. ### zerollzeng commented on Feb 1, 2024 
Collaborator
More actions
Could you please share a reproduce? Thanks!
  2. self-assigned this
[on Feb 1, 2024](https://github.com/NVIDIA/TensorRT/issues/3640#event-11671125094)
  3. added 
[Issue has been triaged by maintainers](https://github.com/NVIDIA/TensorRT/issues?q=state%3Aopen%20label%3A%22triaged%22)Issue has been triaged by maintainers
[on Feb 1, 2024](https://github.com/NVIDIA/TensorRT/issues/3640#event-11671126285)
  4. ### alsozatch commented on Feb 5, 2024 
More actions
Here is a reproducible example. Run 'python3 reproducible_int8_exporter.py'. [reproducible_int8_exporter.zip](https://github.com/NVIDIA/TensorRT/files/14171774/reproducible_int8_exporter.zip)
  5. ### zerollzeng commented on Feb 8, 2024 
Collaborator
More actions
I can also reproduce the issue on TRT 8.6 on x86. But in my test it's been fixed in TRT 10.
  6. ### zerollzeng commented on Feb 8, 2024 
Collaborator
More actions
9.2 also pass.
  7. ### alsozatch commented on Feb 8, 2024 
More actions
Thanks. I’m on a Jetson AGX Orin though, so aarch64 not x86 and JetPack 6.0 is the latest for at least half a year probably more, so I’m stuck at TensorRT 8.6. Is there any more information you can obtain to help with this?
  8. ### zerollzeng commented on Feb 19, 2024 
Collaborator
More actions
Sorry for the late reply, I'm checking internally.
  9. ### zerollzeng commented on Feb 19, 2024 
Last edited by zerollzeng
Collaborator
More actions
Could you please try mark `/model.0/conv/Conv ` or `/model.0/act/Sigmoid` as network output so that the layer fusion can be break? this can be done quickly with `polygraphy run model.onnx --mark ouptut_layer_name` to test, for details please check `polygraphy run -h`
  10. ### ttyio commented on Mar 26, 2024 
Collaborator
More actions
closing since no activity for more than 3 weeks, pls reopen if you still have question. Thanks!
  11.   12. ### Orfeasfil commented on Sep 26, 2024 
Last edited by Orfeasfil
Any update on this? Same issue here on `jetpack 6.0`.....
On `jetpack 5.1.0` works fine. Is this related to `cuda 12`? (same issue when using `Tensorrt 10.4` and `cuda 12.2` on a x86 machine)


[Sign up for free](https://github.com/signup?return_to=https://github.com/NVIDIA/TensorRT/issues/3640)**to join this conversation on GitHub.** Already have an account? [Sign in to comment](https://github.com/login?return_to=https://github.com/NVIDIA/TensorRT/issues/3640)
## Metadata
## Metadata
[Issue has been triaged by maintainers](https://github.com/NVIDIA/TensorRT/issues?q=state%3Aopen%20label%3A%22triaged%22)Issue has been triaged by maintainers
No projects
No milestone
### Relationships
None yet
No branches or pull requests
## Issue actions
  * Open in GitHub Copilot app


You can’t perform that action at this time. 
