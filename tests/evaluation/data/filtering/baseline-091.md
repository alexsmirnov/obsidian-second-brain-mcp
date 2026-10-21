arXiv is now an independent nonprofit! [Learn more](https://info.arxiv.org/about) ×
[ License: arXiv.org perpetual non-exclusive license ](https://info.arxiv.org/help/license/index.html#licenses-available)
arXiv:2510.13760v2 [eess.IV] 04 Nov 2025
# Invited Paper: BitMedViT: Ternary-Quantized Vision Transformer for Medical AI Assistants on the Edge
Mikolaj Walczak, Uttej Kallakuri, Edward Humes, Xiaomin Lin, Tinoosh Mohsenin  Affiliation: Department of Electrical and Computer Engineering Johns Hopkins University {mwalcza1, ukallak1, ehumes2, xlin52, tinoosh}@jh.edu 
###### Abstract
Vision Transformers (ViTs) have demonstrated strong capabilities in interpreting complex medical imaging data. However, their significant computational and memory demands pose challenges for deployment in real-time, resource-constrained mobile and wearable devices used in clinical environments. We introduce, BiTMedViT, a new class of Edge ViTs serving as medical AI assistants that perform structured analysis of medical images directly on the edge. BiTMedViT utilizes ternary-quantized linear layers tailored for medical imaging and combines a training procedure with multi-query attention, preserving stability under ternary weights with low-precision activations. Furthermore, BiTMedViT employs task-aware distillation from a high-capacity teacher to recover accuracy lost due to extreme quantization. Lastly, we also present a pipeline that maps the ternarized ViTs to a custom CUDA kernel for efficient memory bandwidth utilization and latency reduction on the Jetson Orin Nano. Finally, BiTMedViT achieves 86% diagnostic accuracy (89% SOTA) on MedMNIST across 12 datasets, while reducing model size by 43, memory traffic by 39, and enabling 16.8 ms inference at an energy efficiency up to 41 that of SOTA models at 183.62 GOPs/J on the Orin Nano. Our results demonstrate a practical and scientifically grounded route for extreme-precision medical imaging ViTs deployable on the edge, narrowing the gap between algorithmic advances and deployable clinical tools. Github is available at <https://github.com/M-iki/BitMedViT>
###### Index Terms: 
Vision Transformers, Medical Imaging, Ternary Quantization, Edge Computing, Real Time. 
## Introduction
Machine learning for medical imaging and disease detection are rapidly advancing fields with the potential to transform healthcare by enabling real-time, automated analysis of imaging modalities such as X-rays, MRIs and CT scans. This automation supports accurate disease identification, improves diagnostic precision and accelerates clinical decision-making. Historically, Convolutional Neural Networks (CNNs), especially those with residual architectures like ResNet, have demonstrated strong performance. Recent breakthroughs have advanced medical image classification by surpassing traditional CNN accuracy through sophisticated architectures such as Vision Transformers (ViTs) and Vision Mamba .
Fig. 1: Jetson Orin Nano hierarchy and SOTA medical image classification model comparisons for average accuracy on the MedMNIST dataset vs operations and model size compared to the memory available within the GPU consisting of 4 TPCs each with two SMs containing their own L1 cache. Marker size corresponds to the parameter count within each model.
Despite these advances, deploying ViTs in clinical environments remains challenging due to their high computational and memory demands, which restrict adoption in clinical settings with limited hardware resources or unreliable connectivity. Furthermore, transmitting sensitive medical data over wireless networks introduces privacy and security risks - a major concern given HIPAA and other stringent medical data regulations . To address these issues, we investigate deploying these models on the resource-constrained NVIDIA Jetson Orin Nano edge device. Equipped with Arm Cortex-A78AE CPU and an NVIDIA Ampere GPU, the Orin Nano is tailored for low-power, resource-constrained environments. Deploying models optimized for its core computing architecture ensures efficient resource utilization, reduced power consumption, and optimal performance within strict energy budgets. As shown in Figure , the state-of-the-art medical classification models, even under ideal settings, exceed the low memory hierarchy (L1, L2, L3) limits, causing high latency, low throughput, and increased energy use due to the reliance on external accesses to high-latency DRAM. To overcome these deployment constraints, recent research has emphasized model compression techniques such as pruning , quantization , and Knowledge Distillation (KD) that reduce model size and inference costs while maintaining accuracy. KD has proven especially effective, and when combined with quantization, it yields lightweight models retaining high performance at a fraction of the original footprint, which is ideal for edge medical AI applications. In ViTs, further reductions in parameters and memory can be achieved by replacing standard Multi-Head Self-Attention with Multi-Query Attention, significantly cutting computational overhead with minimal impact on accuracy .
Fig. 2: Model training paradigm and BiTMedViT Architecture. Experiments illustrate the overall parameter breakdown and reduction when comparing Multi-Head Self-Attention (MHSA) and Multi-Query Attention (MQA) mechanisms within BiTMedViT. The blue plots depict the average validation accuracy achieved when training 12 student models across each of the MedMNIST2D datasets rounded to the nearest percentage point after 100 epochs.
In this work, we introduce BiTMedViT, which integrates KD with extreme quantization to enhance model compression. We focus on 2-bit ternary quantization of feed-forward layers using the BitNet-1.58B framework and are one of the first works to explore MQA in ViTs to compress the model footprint for medical image classification. We evaluate our method for accuracy and efficiency on edge hardware, benchmarking against MedViTV2 deployed on the NVIDIA Jetson Orin Nano with custom CUDA kernels optimized for performance.
Our key contributions are summarized as follows:
  * •
We evaluate the efficacy of MQA in ViTs for medical image classification, demonstrating more efficient parameter use while preserving accuracy.
  * •
We apply feature and logit distillation from a high-performing state-of-the-art teacher model and propose a distilled student BiTMedViT for medical applications that enables robust classification .
  * •
We integrate BitNet linear layers within BiTMedViT, enabling the training of scalable, memory- and parameter-efficient edge-based ViTs.
  * •
We develop a optimized CUDA kernel compatible with TensorRT and compatible with Nvidia Ampere GPU architectures, found within the Jetson Orin Nano, enabling real-time inference and efficient memory utilization for edge-based medical AI applications.


## II Related Works
### Knowledge Distillation for Medical ViTs
Knowledge distillation (KD) is a proven strategy for compressing large models, particularly effective in data-scarce medical imaging scenarios. Feature-based KD enhances student learning by aligning internal representations with those of a powerful teacher model. Medical-specific distillation methods such as address limited and imbalanced data, while quantization-aware techniques further reduce inference cost. BiTMedViT builds on these by combining feature distillation with low-bit quantization to produce accurate, lightweight ViTs for edge devices.
### Multi-Query Attention in Vision Models
Multi Query Attention (MQA) reduces attention complexity by sharing key-value pairs across queries, offering significant efficiency gains over Multi-Head Self Attention (MHSA). Although adopted in Large Language Models (LLMs) , MQA has seen little application in ViTs. Ainslie et al. show that MHSA architectures can be converted to MQA with minimal changes. BiTMedViT is among the first to evaluate MQA in ViTs for medical imaging.
### Ternary ViTs and Edge Deployment
Ternary quantization reduces weights and activations to three discrete values, balancing efficiency and accuracy. Prior works like Tervit and BitNet-ViT demonstrate that ViTs can be effectively quantized with minimal accuracy loss. However, these methods often overlook deployment feasibility on constrained hardware. BiTMedViT builds on the BitNet-1.58B framework , integrating BitLinear ternary layers and demonstrating practical, low-latency deployment on edge devices like the Jetson Orin Nano. Furthermore, deploying ViTs on edge devices requires tight alignment between model design and hardware constraints. While frameworks like BitNet-Efficient and FPGA-based ternary transformers demonstrate efficient execution, they target general-purpose or LLM scenarios. Prior works have also explored KD, quantization, and ternary ViTs for efficient model compression, but often lacks real-world deployment on medical setting. In addition, MQA is common in LLMs but is rarely applied to ViTs. Our proposed work bridges these gaps by combining feature distillation, extreme ternary quantization, and MQA into a compact ViT pipeline, optimized and deployed with custom CUDA kernels for real-time inference on edge devices like the Jetson Orin Nano.
## III Proposed Approach
### BiTMedViT Architecture
The BiTMedViT architecture is based on the traditional ViT , parameterized by the number of attention heads , transformer layers , and patch embedding dimension . The post-attention Feed-Forward Networks (FFNs) employ an expansion factor of 4, yielding an FFN dimension of E_{\text{ff}}=4\times E_{d}. In this work, we fix the configuration to , , and E_{d}=512, providing a balanced trade-off between model capacity for medical image analysis and a reduced overall parameter count. Prior to deployment on the Orin Nano GPU platform, it is essential to ensure the model is sufficiently compact to meet stringent memory constraints before hardware-specific optimizations are applied. As illustrated in Fig. , a parameter breakdown of the conventional MHSA architecture under this configuration shows that the FFN layers and key–value projections contribute the largest share of parameters. This motivates our focus on compressing these components to reduce overall memory footprint.
#### Attention Layer Compression
Transformer architectures rely heavily on the attention mechanism, which computes context-aware representations through learned input feature linear projections. This mechanism is formally expressed as  
| Attention⁡=softmax⁡​V,\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_{k}}}\right)V,\vskip-4.30554pt  |  
| --- |  
where the queries (), keys (), and values () are projections of the input (). Moreover, the total embedding dimension in Equation is divided into multiple heads , each with separate , , and projection weights. While this increases parallelism and expressivity, the total number of parameters remains constant, as the projections are simply partitioned into smaller subspaces. MQA modifies this structure by sharing the key and value projections across all heads. Let X\\!\in\\!\mathbb{R}^{N\times D} be the patch embeddings, with heads of size so . The attention output for MHSA can then be computed using Equation with the projection weights of Q,K and V in a single head being  
| ,,W_{h}^{Q},W_{h}^{K},W_{h}^{V}\in\mathbb{R}^{D\times d_{h}}\vskip-4.30554pt  |  
| --- |  
where , and are computed as ,,Q_{h}=XW_{h}^{Q},\ K_{h}=XW_{h}^{K},\ V_{h}=XW_{h}^{V}. Within each head we further compute =Attn⁡Y_{h}=\mathrm{Attn}(Q_{h},K_{h},V_{h}), then Y=Concath​​Y=\mathrm{Concat}_{h}(Y_{h})W^{O}. The number of parameters in this case is, 3Dd_{h}H.
MQA on the other hand, maintains the per–head queries but shares the key and value projections across all the heads. Equation in this case can be rewritten as  
| ,,.W_{h}^{Q}\in\mathbb{R}^{D\times d_{h}},\qquad W^{K}_{\mathrm{sh}},W^{V}_{\mathrm{sh}}\in\mathbb{R}^{D\times d_{h}}.\vskip-4.30554pt  |  
| --- |  
where, W^{K}_{\mathrm{sh}} and W^{V}_{\mathrm{sh}} are the shared projections. We then compute for , and as, Q_{h}=XW_{h}^{Q} across all heads, while ,=∈K_{\mathrm{sh}}=XW^{K}_{\mathrm{sh}},\ V_{\mathrm{sh}}=XW^{V}_{\mathrm{sh}}\in\mathbb{R}^{N\times d_{h}} is shared. The attention within each head is now =Attn⁡Y_{h}=\mathrm{Attn}(Q_{h},K_{\mathrm{sh}},V_{\mathrm{sh}}) followed by Y=Concath​​Y=\mathrm{Concat}_{h}(Y_{h})W^{O}.
From Equations and we can conclude that by replacing the MHSA layers by MQA layer the number of parameters for the self-attention reduces by,  
| \displaystyle\underbrace{|W^{K}|+|W^{V}|}_{\text{MHSA}}  | =⟶MQA=\displaystyle=2Dd_{h}H\;\longrightarrow\;\underbrace{|W^{K}_{\mathrm{sh}}|+|W^{V}_{\mathrm{sh}}|}_{\text{MQA}}=2Dd_{h}  |  
| --- | --- |  
| ⇒​reduction for ​K,V.\displaystyle\Rightarrow\quad\mathbf{1/H}\ \text{reduction for }K,V.  |  
We adopt this approach in ViTs and evaluate its effectiveness in a representative experiment shown in the lower right portion of Figure using the training pipeline defined within Section . Our results align with those of , demonstrating that given a pretrained checkpoint or a high-capacity teacher, MQA can approach the accuracy of its MHSA counterpart while reducing overall parameter count.
#### Ternary Quantization
To aggressively compress the FFN layers found within BiTMedViT, we adapt the BitLinear layers from which computes W_{2}A_{8} output activations given 2-bit weights () and int8 activations (). We modify this computation to operate over the full precision weight matrix and patch embedding matrix during training defined over the range W\in\mathbb{R}^{k\times n},\;A\in\mathbb{R}^{m\times k}. The quantized counterparts are then computed using absmean quantization for and absmax quantization for formally defined as  
| =RoundClip⁡,β=​\displaystyle W_{2}=\mathrm{RoundClip}\Big(\frac{W}{\beta+\epsilon},-1,1\Big),\quad\beta=\frac{1}{kn}\textstyle\sum_{i=1,j=1}^{k,n}|W_{ij}|  |  
| --- |  
and  
| =Clip⁡(,−128,127),\displaystyle A_{8}=\mathrm{Clip}\Big(\frac{A}{\gamma+\epsilon},-128,127\Big),\quad\gamma={\max(|x|)\over{127}}  |  
| --- |  
Where represents a small floating point number. We modify \gamma\in\mathbb{R}^{m} to be a vector of length determined by the the number of patch embeddings , while \beta\in\mathbb{R} remains as a single scalar representing the mean value over the entire weight matrix . The scale factors \gamma,\beta are maintained within 16-bit precision during inference to ensure precise de-quantization for corresponding layers computed as O=W_{2}A_{8}\times\gamma\times\beta.
Fig. 3: Bit Packing and custom hardware deployment strategy of BiTMedViT within the GPU of the Jetson Orin Nano. Full precision weights are statically quantized to 2-bit, packed into 32-bit column major words and activations dynamically quantized to int8 rowmajor 128-bit words converting the model from its Pytorch to the custom cuda implementation merged within TensorRT. CUDA Kernel for Blocked Matrix–Matrix Multiplication with 2-bit Weight Unpacking
Weights (2-bit packed), patch input activations (int8), per-patch scale factors , weight scale factor , block sizes , 
(dequantized, bfloat16) stored in global memory 
shared_mem_scale \leftarrow(\gamma\times\beta)
W_{\text{tile}}\leftarrow weight start addr (index ) 
P_{\text{tile}}\leftarrow activation start addr (index ) 
for to step tile_size do
SyncLoad: DRAM registers (W_{\text{packed}})
AsyncLoad: DRAM shared_mem (P_{\text{tile}})
int8_weights Unpack(W_{\text{packed}}) shared_mem 
frag_B WMMA.Load(int8_weights) 
for to do
frag_P WMMA.Load(P_{\text{tile}}[m]) 
accum[m] WMMA.MMA(frag_P, frag_B) 
DRAM Dequantize(accum, shared_mem_scale) 
### Knowledge Distillation
Since the ternary quantized bitlinear layers require extensive training from scratch and MQA demonstrates effectiveness primarily when adapted from a pretrained model, we utilize MedViTv2 a state-of-the-art high-performing medical image classification ViT, achieving strong accuracy across medical imaging benchmarks. We train BiTMedViT by minimizing a composite loss function  
| =++\mathcal{L}_{\text{total}}=\lambda_{\text{cls}}\mathcal{L}_{\text{CE}}+\lambda_{\text{logits}}\mathcal{L}_{\text{KD}}+\lambda_{\text{feat}}\mathcal{L}_{\text{feat}}\vskip-4.30554pt  |  
| --- |  
where \mathcal{L}_{\text{CE}} is the cross-entropy loss for classification, \mathcal{L}_{\text{KD}} denotes the Kullback–Leibler divergence aligning the student’s logits with the teacher’s, and \mathcal{L}_{\text{feat}} encourages alignment between intermediate feature representations. To facilitate this process, a trainable projection layer is incorporated during training to align the student’s feature dimensions with those of the teacher, in effect defining BiTMedViT to match the same patch size as MedViTv2.
### Model Optimization and Hardware Deployment
BiTMedViT utilizes both ternary and full precision layers, necessitating a deployment strategy that optimizes for mixed precision within the GPU of the Orin Nano consisting of a custom CUDA kernel integrated within TensorRT. To design our optimized kernel, The Jetson Orin Nano GPU is organized into a Graphics Processing Cluster (GPC), which contains four Texture Processing Clusters (TPCs), each with two Streaming Multiprocessors (SMs). Each SM includes four Tensor Cores capable of performing Warp Matrix Multiply (WMMA) operations at a minimum int8 precision, of varying dimensions. To maximize memory and compute bandwidth, operations are performed at the per-warp level, where a warp consists of 32 threads, with a maximum read transaction of int4 (128 bits) per thread. Each SM communicates with the 8GB external DRAM through a hierarchical memory system: a 192KB L1 cache per SM, a 4MB L2 cache shared among all SMs, and a 4MB system cache. Since DRAM reads introduce the highest latency, designing highly optimized kernels that perform compact, coalesced data accesses is essential to reduce L2 cache misses and maximize throughput.
#### Weight Packing Strategy and WMMA Compatibility
To optimize inference of BiTMedViT, we redesign the weight packing scheme from the original BitNet kernel to align with operations for maximizing onboard Tensor Core utilization. Weights are arranged into matrices of size 8\times 32\times 16 (M N K). Within this layout, weights are packed into column major 32\times 16 fragments, where each 32-bit memory word encodes sixteen () 2-bit weight values with activations quantized to 8-bit integers grouped into 128-bit (int4) fragments, allowing sixteen activation values to be loaded per memory transaction. as shown in Figure . This compact packing reduces memory traffic and allows efficient unpacking via optimized bit masking and shifting during inference. In addition the 32 outputs of per input patch enables coalesced and consecutive 16-bit (BF16) output write-backs during de-quantization.
#### CUDA Kernel Integration
As outlined within Figure Our custom CUDA kernel parallelizes computation using a two-dimensional grid of size , where corresponds to output channels and to input patches divided among each of the onboard SMs. Each thread block contains multiple 32-thread warps, with threads accessing distinct output elements acting on the same activation fragment. To minimize runtime overhead, decoded weights are unpacked once per inference and stored in shared memory within each thread block. Activations, which have lower temporal reuse, are asynchronously loaded into shared memory, skipping past the L1 cache maintaining a continuous data streaming pipeline and enabling weight decoding during activation loading. Algorithm outlines this functionality and the overall per-block kernel execution.
Fig. 4: Latency and memory read traffic comparing the PyTorch BF16 kernel to our optimized W2A8 implementation for two varying workloads. Results on the left show the latency and memory read transactions for the K-N layer and results on the right show the N-K layer within the BiTMedViT FF network. Two power modes, 15W and Super (25W) mode are used to determine performance under varying clock speeds.
Using the Nvidia Nsight Compute and Nsight Systems profiling tools, we extensively evaluate the efficiency of the custom kernel implementation as well as overall inference time for the full model architecture. Profiling the linear layers in BiTMedViT as shown in Figure shows a 4.9 to 39 reduction in weight traffic (bytes moved). This shifts most accesses to on-chip memory and minimizes DRAM transactions, aligning with the power/latency gains.
#### TensorRT Integration for Efficient Deployment
We deploy BiTMedViT as an end-to-end solution by integrating our custom CUDA kernel into NVIDIA TensorRT. Since TensorRT doesn’t natively support quantized/compressed BitLinear layers, the plugin registers the layer, declares I/O shapes, and specifies supported precisions enabling building of engines with the new operation. On the Orin Nano, TensorRT had limited support for dynamic INT8 activations and would upcast them to FP16 at the plugin boundary. To accommodate this, we added kernel variants that accept BF16 activations and scales, and due to the limited BF16 support on the Orin Nano, FP16 activations and scales.
TABLE I: Performance of BiTMedViT across MedMNIST2D datasets in comparison with state-of-the-art classification models. The teacher model MedViTv2-L and BiTMedViT are highlighted. Validation accuracy is reported for fair comparison, while test accuracy for each dataset is indicated with a .  
| PathMNIST  | ChestMNIST  | DermaMNIST  | OCTMNIST  | PneumoniaMNIST  | RetinaMNIST |  
| --- | --- | --- | --- | --- | --- |  
| ACC  | AUC  | ACC  | AUC  | ACC  | AUC  | ACC  | AUC  | ACC  | AUC  | ACC  | AUC |  
| ResNet50 (224)   | 89.2  | 98.9  | 94.8  | 77.3  | 73.1  | 91.2  | 77.6  | 95.8  | 88.4  | 96.2  | 51.1  | 71.6 |  
| MedMamba-S   | 95.5  | 99.7  | -  | -  | 75.8  | 92.4  | 92.9  | 99.1  | 93.6  | 97.6  | 54.5  | 71.8 |  
| MedMamba-B   | 96.4  | 99.9  | -  | -  | 75.7  | 92.5  | 92.7  | 99.6  | 92.5  | 97.3  | 55.3  | 71.5 |  
| MedViTv2-S   | 96.5  | 99.8  | 96.4  | 80.3  | 79.2  | 94.6  | 94.2  | 99.4  | 96.5  | 99.6  | 56.2  | 78.0 |  
| MedViTv2-B   | 97.0  | 99.9  | 96.4  | 81.5  | 80.8  | 94.8  | 94.4  | 99.6  | 96.9  | 99.6  | 57.5  | 78.3 |  
 |  
| BitMedViT (ours)  |  
| BreastMNIST  | BloodMNIST  | TissueMNIST  | OrganAMNIST  | OrganCMNIST  | OrganSMNIST |  
| ACC  | AUC  | ACC  | AUC  | ACC  | AUC  | ACC  | AUC  | ACC  | AUC  | ACC  | AUC |  
| ResNet50 (224)   | 84.2  | 86.6  | 95.0  | 99.7  | 68.0  | 93.2  | 94.7  | 99.8  | 91.1  | 99.3  | 78.5  | 97.5 |  
| MedMamba-S   | 85.3  | 80.6  | 98.4  | 99.9  | -  | -  | 95.9  | 99.9  | 94.4  | 99.7  | 83.3  | 98.4 |  
| MedMamba-B   | 89.1  | 84.9  | 98.3  | 99.9  | -  | -  | 96.4  | 99.9  | 94.3  | 99.8  | 83.4  | 98.3 |  
| MedViTv2-S   | 89.5  | 94.7  | 98.5  | 99.9  | 70.5  | 93.9  | 96.6  | 99.9  | 95.0  | 99.8  | 83.9  | 98.6 |  
| MedViTv2-B   | 90.4  | 94.9  | 98.5  | 99.9  | 71.1  | 94.2  | 96.9  | 99.9  | 95.3  | 99.8  | 84.4  | 98.7 |  
 |  
| BitMedViT (ours)  |  
Fig. 5: Visualized GradCam for the MedViT-v2-L teacher and BiTMedViT across four representative datasets found within MedMNIST. Green boxes represent the manually placed relevant regions for diagnosis.
## IV Results
### BiTMedViT Accuracy Benchmarking
MedMNIST2D serves as the dataset used to evaluate the performance of BiTMedViT for medical image classification spanning 12 2D datasets of varying modalities and complexity. Using the pretrained MedViTv2-L teacher, we apply the distillation strategy outlined in Section across each dataset. Figure presents attention maps generated using GradCam for four representative images, which, while exhibiting increased noise, demonstrate stronger focus on diseased regions. These results highlight the ability of BiTMedViT to attend to clinically relevant regions despite aggresive reduction in parameters and weight expressivity.
Furthermore, we compare overall accuracy and ROC AUC against state-of-the-art medical image classification models, as shown in Table . Across the MedMNIST benchmark, BiTMedViT achieves competitive accuracy relative to leading models, attaining perfect or near-perfect AUC scores on multiple datasets, including PathMNIST, BloodMNIST, and OrganCMNIST. Notably, even after applying aggressive compression strategies, BiTMedViT maintains performance close to the teacher, with only a 3% decrease in average validation accuracy (86% vs. 89%) and a similar reduction on the test set (82% vs. 85%), demonstrating effective model compression while preserving competitive performance.
### Hardware Deployment
Fig. 6: Jetson Orin Nano Power Versus Time for the Ternary Precision (TP) BitMedVit(Ours) and Full precision (FP) MedVitV2-L teacher. BiTMedViT achieves nearly a 1.14\times reduction in power while achieving nearly a reduction in latency for processing 10 batches of 50 inferences.  TABLE II: Hardware Comparison Against state of the art medical image classification models deployed within the Jetson Orin Nano. Values with * are recomputed with our metrics  
| Work  | Precision  |   
 |   
 | (GOPs / sec) |  
| --- |  
|   
 |  
| --- | --- |  
|   
 | MedViTv2-L (Baseline)  |  
| --- |  
| ViT  | Float32  | 117.26  | 447.71  | 12.9  | 4.25  | 366.63  | 35.3  | 8.31 |  
| MedMambaLite-ST   | Float32  | 0.63  | 2.4  | 0.15  | 2.7  | 13.03  | 11.84  | *4.39 |  
|   
 | BiTMedViT (ours) |  
| --- |  
 |  
We deploy BiTMedViT within the Orin Nano and compare against (i) full-precision MedViTv2-L baseline and (ii) MedMambaLite. with hardare results summarized in Table . BiTMedViT achieves 16.88 ms latency per inference versus 366.63 ms for the MedViTv2-L baseline (21.7 faster). This corresponds to a 683.06 GOPs/sec throughput, 19.4 higher than the baseline. When compared to MedMambaLite, our implementation is 57.7\times better in terms of throughput while being \approx 42\times more energy efficient. Figure shows the power trace for executing an identical workload on the teacher and BiTMedViT on the Orin Nano platform: 50 inferences followed by 3 seconds of idle time, repeated 10 times. Compared to the teacher, BiTMedViT achieves a reduction in total inference time and a 1.14\times lower peak power consumption.
## Conclusion
In this work, we presented BiTMedViT, a ternary-quantized ViT for efficient, real-time medical image classification on the edge. By integrating Multi-Query Attention, knowledge distillation, and hardware-aware CUDA & TensorRT optimization, BitMedViT achieves 86% accuracy on MedMNIST only 3% below its teacher while reducing parameters by 13.6\times, model size by , and memory transfers by , at a 92% L2 cache hit rate and performing at 183.62 GOPs/J, 22 that of MedViTv2-L and that of MedMambaLite-ST . These results demonstrate that extreme-precision quantization, combined with architectural and deployment co-design, enables SOTA ViT performance within the strict compute and memory constraints of clinical edge devices.
Future work will explore mixed-precision training techniques to dynamically adjust bit precisions between layers, alongside adaptive quantization based on input modality or task complexity. In addition, expanding BiTMedViT to CPU-only and specialized low-power devices will further enhance its deployability and accessibility across diverse clinical environments and devices
## References
  * Romina Aalishah et al.  Mambalitesr: Image super-resolution with low-rank mamba using knowledge distillation.  In _2025 26th International Symposium on Quality Electronic Design (ISQED)_ , pages 1–8. IEEE, 2025a. 
  * Romina Aalishah et al.  Medmambalite: Hardware-aware mamba for medical image classification.  _arXiv preprint arXiv:2508.05049 [Accepted in IEEE BioCAS 2025]_ , 2025b. 
  * Joshua Ainslie et al.  Gqa: Training generalized multi-query transformer models from multi-head checkpoints.  _arXiv preprint arXiv:2305.13245_ , 2023. 
  * Al Ameen et al.  Security and privacy issues in wireless sensor networks for healthcare applications.  _Journal of medical systems_ , 36(1):93–101, 2012. 
  * Rohan Anil et al.  Palm 2 technical report.  _arXiv preprint arXiv:2305.10403_ , 2023. 
  * Chao Chen et al.  A review of convolutional neural network based methods for medical image classification.  _Computers in biology and medicine_ , 185:109507, 2025. 
  * Alexey Dosovitskiy et al.  An image is worth 16x16 words: Transformers for image recognition at scale.  _arXiv preprint arXiv:2010.11929_ , 2020. 
  * Hao Guan et al.  Federated learning for medical image analysis: A survey.  _Pattern recognition_ , 151:110424, 2024. 
  * Kaiming He et al.  Deep residual learning for image recognition.  In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 770–778, 2016. 
  * Morteza Hosseini et al.  Cyclic sparsely connected architectures for compact deep convolutional neural networks.  _IEEE Transactions on Very Large Scale Integration (VLSI) Systems_ , 29(10):1757–1770, 2021. 
  * Uttej Kallakuri et al.  Resource-aware saliency-guided differentiable pruning for deep neural networks.  In _Proceedings of the Great Lakes Symposium on VLSI 2024_ , pages 694–699, 2024. 
  * Uttej Kallakuri et al.  Enabling on-device medical ai assistants via input-driven saliency adaptation.  _arXiv preprint arXiv:2506.11105 [Accepted in IEEE BioCAS 2025]_ , 2025a. 
  * Uttej Kallakuri et al.  Magrip: Magnitude and gradient-informed pruning for task-agnostic large language models.  _ACM Trans. Embed. Comput. Syst._ , September 2025b.  ISSN 1539-9087. 
  * Hee E Kim et al.  Transfer learning for medical image classification: a literature review.  _BMC medical imaging_ , 22(1):69, 2022. 
  * Jangho Kim et al.  Qkd: Quantization-aware knowledge distillation.  _arXiv preprint arXiv:1911.12491_ , 2019. 
  * Nikita Kitaev et al.  Reformer: The efficient transformer.  _arXiv preprint arXiv:2001.04451_ , 2020. 
  * Shuming Ma et al.  The era of 1-bit llms: All large language models are in 1.58 bits.  _arXiv preprint arXiv:2402.17764_ , 1(4), 2024. 
  * Nitheesh Kumar Manjunath et al.  An energy efficient edgeai autoencoder accelerator for reinforcement learning.  _IEEE Open Journal of Circuits and Systems_ , 2:182–195, 2021. 
  * Omid Nejati Manzari et al.  Medvit: a robust vision transformer for generalized medical image classification.  _Computers in biology and medicine_ , 157:106791, 2023. 
  * Omid Nejati Manzari et al.  Medical image classification with kan-integrated transformers and dilated neighborhood attention.  _arXiv preprint arXiv:2502.13693_ , 2025. 
  * Mason Marks and Claudia E Haupt.  Ai chatbots, health privacy, and challenges to hipaa compliance.  _Jama_ , 330(4):309–310, 2023. 
  * Arnab Neelim Mazumder and Tinoosh Mohsenin.  Reg-tunev2: A hardware-aware and multiobjective regression-based fine-tuning approach for deep neural networks on embedded platforms.  _IEEE Micro_ , 43(6):74–83, 2023. 
  * Bitnet gpu inference kernels, 2025.  Accessed: 2025-08-13. 
  * NVIDIA Corporation.  NVIDIA Nsight Compute.  Accessed: 2025-07-30. 
  * NVIDIA Corporation.  NVIDIA Nsight Systems.  Accessed: 2025-07-30. 
  * NVIDIA Corporation.  NVIDIA TensorRT: High-performance deep learning inference optimizer and runtime.  Accessed: 2025‑07‑30. 
  * Sangjoon Park et al.  Self-evolving vision transformer for chest x-ray diagnosis through knowledge distillation.  _Nature communications_ , 13(1):3848, 2022. 
  * Adam Paszke et al.  Pytorch: An imperative style, high-performance deep learning library.  _Advances in neural information processing systems_ , 32, 2019. 
  * Hasib-Al Rashid et al.  Tinyvqa: Compact multimodal deep neural network for visual question answering on resource-constrained devices. 
  * Hasib-Al Rashid et al.  Hac-m-dnn: Hardware aware compression of sustainable multimodal deep neural networks for efficient tinyml deployment.  In _2025 IEEE Conference on Technologies for Sustainability (SusTech)_ , pages 1–7. IEEE, 2025. 
  * Md Eshmam Rayed et al.  Deep learning for medical image segmentation: State-of-the-art advancements and challenges.  _Informatics in medicine unlocked_ , 47:101504, 2024. 
  * Ramprasaath R Selvaraju et al.  Grad-cam: Visual explanations from deep networks via gradient-based localization.  In _Proceedings of the IEEE international conference on computer vision_ , pages 618–626, 2017. 
  * Aynur Sevinc et al.  A distillation approach to transformer-based medical image classification with limited data.  _Diagnostics_ , 15(7):929, 2025. 
  * Hugo Touvron et al.  Llama: Open and efficient foundation language models.  _arXiv preprint arXiv:2302.13971_ , 2023. 
  * Jinheng Wang et al.  Bitnet. cpp: Efficient edge inference for ternary llms.  _arXiv preprint arXiv:2502.11880_ , 2025. 
  * Sheng Xu et al.  Tervit: An efficient ternary vision transformer.  _arXiv preprint arXiv:2201.08050_ , 2022. 
  * Jiancheng Yang et al.  Medmnist classification decathlon: A lightweight automl benchmark for medical image analysis.  In _IEEE 18th International Symposium on Biomedical Imaging (ISBI)_ , pages 191–195, 2021. 
  * Jiancheng Yang et al.  Medmnist v2-a large-scale lightweight benchmark for 2d and 3d biomedical image classification.  _Scientific Data_ , 10(1):41, 2023. 
  * Chenyang Yin et al.  Tereffic: Highly efficient ternary llm inference on fpga.  _arXiv preprint arXiv:2502.16473_ , 2025. 
  * Zhengqing Yuan et al.  Vit-1.58 b: Mobile vision transformers in the 1-bit era.  _arXiv preprint arXiv:2406.18051_ , 2024. 
  * Yubiao Yue et al.  Medmamba: Vision mamba for medical image classification.  _arXiv preprint arXiv:2403.03849_ , 2024. 
  * Yichi Zhang et al.  Segment anything model for medical image segmentation: Current applications and future directions.  _Computers in Biology and Medicine_ , 171:108238, 2024. 
  * Quantized feature distillation for network quantization.  In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 37, pages 11452–11460, 2023. 
  * Lianghui Zhu et al.  Vision mamba: Efficient visual representation learning with bidirectional state space model.  _arXiv preprint arXiv:2401.09417_ , 2024.