[Skip to content](https://github.com/NVIDIA/TensorRT/issues/4348#start-of-content)
You signed in with another tab or window. [Reload](https://github.com/NVIDIA/TensorRT/issues/4348) to refresh your session. You signed out in another tab or window. [Reload](https://github.com/NVIDIA/TensorRT/issues/4348) to refresh your session. You switched accounts on another tab or window. [Reload](https://github.com/NVIDIA/TensorRT/issues/4348) to refresh your session. Dismiss alert
#  Lack of Performance Gain with Mixed Precison using TensorRT 10.3 when running DinoV2S on jetson-orin/RTX 4090
Copy link
Copy link
[Lack of Performance Gain with Mixed Precison using TensorRT 10.3 when running DinoV2S on jetson-orin/RTX 4090](https://github.com/NVIDIA/TensorRT/issues/4348#top)#4348
Labels
[Module:PerformanceGeneral performance issues](https://github.com/NVIDIA/TensorRT/issues?q=state%3Aopen%20label%3A%22Module%3APerformance%22)General performance issues[Issue has been triaged by maintainers](https://github.com/NVIDIA/TensorRT/issues?q=state%3Aopen%20label%3A%22triaged%22)Issue has been triaged by maintainers
## Description
Issue body actions
## Description
I tried to convert a DinoV2-S (with reg) using trtexec, I see no speed improvements when testing fp16 and the best flag, in fact I consistently see a very slight performance degradation. I am measuring median GPU Compute time over 100 runs  
| Model  | TRT 10.4 (Jetson Orin)  | TRT 10.8 (RTX 4090)  |  
| --- | --- | --- |  
| dinos (raw + best)  | 23.1739 ms  | 0.928772 ms  |  
| dinos (raw + fp16)  | 23.107 ms  | 0.926697 ms  |  
| dinos (raw + fp8)  | 92.1541 ms  | 3.44415 ms  |  
| dinos (raw + fp16+fp8+int4)  | 23.7127 ms  | 0.927734 ms  |  
| dinos (raw + fp16+int8+int4)  | 22.8879 ms  | 0.927734 ms  |  
| dinos (int8 PTQ)  | 22.1177 ms  | 0.97998 ms  |  
| dinos (int8smoothquant PTQ)  | 22.0298 ms  | 0.9021 ms  |  
considering the model is compute bound and Jetson is an Ampere Chip, I am focussing on INT8 and INT8_SQ calibration  
| Model  | GPU Compute Time (ms)  | Throughput (qps)  | Median Latency (ms)  |  
| --- | --- | --- | --- |  
| dinos (raw + best)  | 23.1739  | 39.11  | 0.280327  |  
| dinos (raw + fp16)  | 23.107  | 42.6883  | 23.5626  |  
| dinos (int8 ptq)  | 22.1177  | 44.8785  | 22.4112  |  
| dinos (int8 SQ)  | 22.0298  | 45.0604  | 22.3307  |  
The speed up is ALOT smaller than expected - would love some insight
## Environment
Tested on 2 different setups, evirinment variables defined for both. I am using Jetpack 6.1 for the Jetson Orin 8Gb.
**TensorRT Version** : 10.4 | 10.8
**NVIDIA GPU** : Jetson Orin Nano 4GB | RTX 4090
**NVIDIA Driver Version** : NVIDIA UNIX Open Kernel Module for arm64 540.4.0 | NVIDIA UNIX x86_64 Kernel Module 565.77
**CUDA Version** : 12.6.1 | 12.7
**CUDNN Version** : 9.3.0 | 9.1
Operating System:
Python Version (if applicable): 3.12.3
Tensorflow Version (if applicable):
PyTorch Version (if applicable): 2.5.1+cu124
Baremetal or Container (if so, version):
## Relevant Files
**Model link** : <https://github.com/facebookresearch/dinov2> - with a minor tweak to allow onnx conversion. The below code will allow the conversion of onnx with the saved model from the repo.
## Steps To Reproduce
**Commands or scripts** :
### Code to generate onnx file

```
import math
import torch
import torch.nn as nn

INPUT_SIZE = (3, 364, 490)

class DinoV2Vits14Reg(torch.nn.Module):
    """
    Dinov2 Vits14 Reg model from Facebook Research with bicubic interpolation replaced with bilinear.
    Modified to work correctly with ONNX export.
    """
    import math
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import modelopt.torch.quantization as mtq

DATA_PATH = Path("/scratch/dino_data")
INPUT_SIZE = (3, 364, 490)

class DinoV2Vits14Reg(torch.nn.Module):
    """
    Dinov2 Vits14 Reg model from Facebook Research with bicubic interpolation replaced with bilinear.
    Modified to work correctly with ONNX export.
    """
    def __init__(self):
        super().__init__()
        # Load the base model
        base_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        
        # Store all the model components we need
        self.patch_size = base_model.patch_size
        self.interpolate_offset = base_model.interpolate_offset
        self.interpolate_antialias = False
        self.patch_embed = base_model.patch_embed
        self.cls_token = base_model.cls_token
        self.pos_embed = base_model.pos_embed
        self.blocks = base_model.blocks
        self.norm = base_model.norm
        self.head = base_model.head
        
    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="nearest",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        
        # Add class token
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        
        # Add positional encoding
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        return x  # Removed pos_drop as it's not needed/present

    def forward(self, x):
        # Initial token preparation
        x = self.prepare_tokens_with_masks(x)
        
        # Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x)
            
        # Final norm and head
        x = self.norm(x)
        return self.head(x[:, 0])
    

dinov2_vits14_reg = DinoV2Vits14Reg() #torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
dinov2_vits14_reg.eval()
random_input = torch.randn(2, *INPUT_SIZE)

# RAW ONNX EXPORT FOR JUST FLAG CHANGING
# torch.onnx.export(model, random_input, "/scratch/dinov2_vits14_reg.onnx", opset_version=20, verbose=True)


def create_dummy_data(folder: Path = DATA_PATH, num_samples: int = 1000):
    # make folder if not exists
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(num_samples):
        torch.save(torch.randn(*INPUT_SIZE), folder / f"dummy_{i}.pt")

create_dummy_data()

class DummyDataset(Dataset):
    def __init__(self, folder: Path, transform=None):
        self.folder = folder
        self.files = list(folder.glob("*.pt"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = torch.load(file)
        
        if self.transform:
            data = self.transform(data)
        
        return data

# Define the dataloader for the dummy data
def get_dataloader(batch_size: int = 32):
    dataset = DummyDataset(folder=DATA_PATH)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


config = mtq.INT8_DEFAULT_CFG
data_loader = get_dataloader(batch_size=20)


# Define forward_loop. Please wrap the data loader in the forward_loop
def forward_loop(model):
    for batch in data_loader:
        model(batch)

# forward_loop(dinov2_vits14_reg)

# # Quantize the model and perform calibration (PTQ)
model = mtq.quantize(dinov2_vits14_reg, config, forward_loop)


torch.onnx.export(model, random_input, "/scratch/dinov2_vits14_reg_int8_ptq.onnx", opset_version=20, verbose=True)

```

### trtexec command
`trtexec --onnx=/scratch/dinov2_vits14_reg.onnx --avgRuns=100 --useCudaGraph --saveEngine=/scratch/dino_best.trt --tacticSources=-CUBLAS,-CUBLAS_LT,-CUDNN --versionCompatible --excludeLeanRuntime --best`
**Have you tried[the latest release](https://developer.nvidia.com/tensorrt)?**: yes
**Can this model run on other frameworks?** For example run ONNX model with ONNXRuntime (`polygraphy run <model.onnx> --onnxrt`): yes
Reactions are currently unavailable
## Activity
  1. added 
[Module:PerformanceGeneral performance issues](https://github.com/NVIDIA/TensorRT/issues?q=state%3Aopen%20label%3A%22Module%3APerformance%22)General performance issues
[Issue has been triaged by maintainers](https://github.com/NVIDIA/TensorRT/issues?q=state%3Aopen%20label%3A%22triaged%22)Issue has been triaged by maintainers
[on Feb 7, 2025](https://github.com/NVIDIA/TensorRT/issues/4348#event-16228492807)
  2. ### kevinch-nv commented on Feb 7, 2025 
Collaborator
More actions
From these results it suggests that FP16 kernels have similar performance to INT8. This is true for a few model architectures - Tagging [@nvpohanh](https://github.com/nvpohanh) to help investigate.
  3. assigned [on Feb 7, 2025](https://github.com/NVIDIA/TensorRT/issues/4348#event-16228605834)
  4. ### SConsul commented on Mar 5, 2025 
More actions
[@kevinch-nv](https://github.com/kevinch-nv) [@nvpohanh](https://github.com/nvpohanh) Any updates? Any advice on what modifications could be done attain faster INT8 kernels?
  5. ### nvpohanh commented on Mar 6, 2025 
Collaborator
More actions
Could you share the logs with these four commands?
     * `trtexec --onnx=/scratch/dinov2_vits14_reg.onnx --fp16 --verbose --profilingVerbosity=detailed --dumpLayerInfo --useCudaGraph --noDataTransfers --useSpinWait --dumpProfile --separateProfileRun`
     * `trtexec --onnx=/scratch/dinov2_vits14_reg.onnx --best --verbose --profilingVerbosity=detailed --dumpLayerInfo --useCudaGraph --noDataTransfers --useSpinWait --dumpProfile --separateProfileRun`
     * `trtexec --onnx=/scratch/dinov2_vits14_reg.onnx --fp16 --verbose --profilingVerbosity=detailed --dumpLayerInfo --useCudaGraph --noDataTransfers --useSpinWait --dumpProfile --separateProfileRun --versionCompatible`
     * `trtexec --onnx=/scratch/dinov2_vits14_reg.onnx --best --verbose --profilingVerbosity=detailed --dumpLayerInfo --useCudaGraph --noDataTransfers --useSpinWait --dumpProfile --separateProfileRun --versionCompatible`
  6. ### SConsul commented on Mar 12, 2025 
More actions
Thanks for getting back. The `--versionCompatible` flag was erroring out without `--excludeLeanRuntime` so I have also included logs that
### FP16

```
trtexec --onnx=./dinov2_vits14_reg.onnx --fp16 --verbose --profilingVerbosity=detailed --dumpLayerInfo --useCudaGraph --noDataTransfers --useSpinWait --dumpProfile --separateProfileRun > dino_fp16.txt
[03/11/2025-16:50:27] [W] * GPU compute time is unstable, with coefficient of variance = 9.52036%.
[03/11/2025-16:50:27] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.

```

### BEST

```
trtexec --onnx=./dinov2_vits14_reg.onnx --best --verbose --profilingVerbosity=detailed --dumpLayerInfo --useCudaGraph --noDataTransfers --useSpinWait --dumpProfile --separateProfileRun > dino_best.txt
[03/11/2025-16:51:26] [W] [TRT] Calibrator is not being used. Users must provide dynamic range for all tensors that are not Int32 or Bool.

```

### FP16 with versionCompatible

```
trtexec --onnx=./dinov2_vits14_reg.onnx --fp16 --verbose --profilingVerbosity=detailed --dumpLayerInfo --useCudaGraph --noDataTransfers --useSpinWait --dumpProfile --separateProfileRun --versionCompatible > dino_fp16_vc.txt
[03/11/2025-16:55:53] [E] Error[6]: IBuilder::buildSerializedNetwork: Error Code 6: API Usage Error (Unable to load library: libnvinfer_lean.so.10: libnvinfer_lean.so.10: cannot open shared object file: No such file or directory)
[03/11/2025-16:55:53] [E] Engine could not be created from network
[03/11/2025-16:55:53] [E] Building engine failed
[03/11/2025-16:55:53] [E] Failed to create engine from model or file.
[03/11/2025-16:55:53] [E] Engine set up failed

```

### BEST with versionCompatible

```
trtexec --onnx=./dinov2_vits14_reg.onnx --best --verbose --profilingVerbosity=detailed --dumpLayerInfo --useCudaGraph --noDataTransfers --useSpinWait --dumpProfile --separateProfileRun --versionCompatible > dino_best_vc.txt
[03/11/2025-16:56:37] [W] [TRT] Calibrator is not being used. Users must provide dynamic range for all tensors that are not Int32 or Bool.
[03/11/2025-16:59:03] [E] Error[6]: IBuilder::buildSerializedNetwork: Error Code 6: API Usage Error (Unable to load library: libnvinfer_lean.so.10: libnvinfer_lean.so.10: cannot open shared object file: No such file or directory)
[03/11/2025-16:59:03] [E] Engine could not be created from network
[03/11/2025-16:59:03] [E] Building engine failed
[03/11/2025-16:59:03] [E] Failed to create engine from model or file.
[03/11/2025-16:59:03] [E] Engine set up failed

```

### FP16 with versionCompatible and excludeLeanRuntime

```
trtexec --onnx=./dinov2_vits14_reg.onnx --fp16 --verbose --profilingVerbosity=detailed --dumpLayerInfo --useCudaGraph --noDataTransfers -
-useSpinWait --dumpProfile --separateProfileRun --versionCompatible --excludeLeanRuntime > dino_fp16_vc_lean.txt

```

### BEST with versionCompatible and excludeLeanRuntime

```
trtexec --onnx=./dinov2_vits14_reg.onnx --best --verbose --profilingVerbosity=detailed --dumpLayerInfo --useCudaGraph --noDataTransfers --useSpinWait --dumpProfile --separateProfileRun --versionCompatible --excludeLeanRuntime > dino_best_vc_lean.txt
[03/11/2025-17:11:14] [W] [TRT] Calibrator is not being used. Users must provide dynamic range for all tensors that are not Int32 or Bool.
[03/11/2025-17:13:57] [W] * GPU compute time is unstable, with coefficient of variance = 8.38653%.
[03/11/2025-17:13:57] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.

```

  7. ### nvpohanh commented on Mar 12, 2025 
Collaborator
More actions
I see. This is Transformer-like networks. For Transformer-like network, please use ModelOpt to insert Q/DQ ops into the ONNX file to utilize INT8/FP8 perf improvements.
Here is an example: <https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html#performance-benchmarking-with-onnx-quantization>


[Sign up for free](https://github.com/signup?return_to=https://github.com/NVIDIA/TensorRT/issues/4348)**to join this conversation on GitHub.** Already have an account? [Sign in to comment](https://github.com/login?return_to=https://github.com/NVIDIA/TensorRT/issues/4348)
## Metadata
## Metadata
[Module:PerformanceGeneral performance issues](https://github.com/NVIDIA/TensorRT/issues?q=state%3Aopen%20label%3A%22Module%3APerformance%22)General performance issues[Issue has been triaged by maintainers](https://github.com/NVIDIA/TensorRT/issues?q=state%3Aopen%20label%3A%22triaged%22)Issue has been triaged by maintainers
No projects
No milestone
### Relationships
None yet
No branches or pull requests
## Issue actions
  * Open in GitHub Copilot app


You can’t perform that action at this time. 
