# Creating a Real-Time License Plate Detection and Recognition App
Feb 25, 2021 
## AI-Generated Summary
  * NVIDIA [TAO Toolkit](https://developer.nvidia.com/tao-toolkit) provides production-quality pretrained models for automatic license plate recognition, including License Plate Detection (LPD) and License Plate Recognition (LPR) models available on NGC.
  * Fine-tuning the LPD model on the OpenALPR dataset using TAO Toolkit achieves 82.28% average precision, significantly outperforming training from scratch which reaches only 53.11% mAP with the same epochs and batch size.
  * Fine-tuning the LPR model with a pretrained ResNet18 backbone reaches 90.90% validation accuracy in 24 epochs, while training from scratch shows 0% validation accuracy even after 1000 epochs.
  * Trained models export to encrypted .etlt format for deployment with the [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk), which optimizes video decoding, preprocessing, and inference for real-time multi-stream analytics.
  * A reference DeepStream application demonstrates end-to-end ALPR pipelines achieving up to 1447 FPS on T4 GPUs with 14 concurrent 1080p streams using TrafficCamNet, LPD, and LPR models.


### Next Steps
  * Download the [LPD model](https://ngc.nvidia.com/catalog/models/nvidia:tao_lpdnet) from NGC to begin fine-tuning.
  * Download the [LPR model](https://ngc.nvidia.com/catalog/models/nvidia:tao_lprnet) from NGC for license plate recognition training.
  * Explore the [NVIDIA-AI-IOT/deepstream_lpr_app reference application](https://github.com/NVIDIA-AI-IOT/deepstream_lpr_app) for deployment guidance.


Powered by NVIDIA Nemotron. AI-generated content may summarize information incompletely. Verify important information. [Learn more](https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/)
Automatic license plate recognition (ALPR) on stationary to fast-moving vehicles is one of the common intelligent video analytics applications for smart cities. Some of the common use cases include parking assistance systems, automated toll booths, vehicle registration and identification for delivery and logistics at ports, and medical supply transporting warehouses. Being able to do this in real time is key to servicing these markets to their full potential. Traditional techniques rely on specialized cameras and processing hardware, which is both expensive to deploy and difficult to maintain.
_Figure 1. Cars with bounding boxes._
The pipeline for ALPR involves detecting vehicles in the frame using an object detection deep learning model, localizing the license plate using a license plate detection model, and then finally recognizing the characters on the license plate. Optical character recognition (OCR) using deep neural networks is a popular technique to recognize characters in any language.
In this post, we show you how to use production-quality AI models such as [License Plate Detection (LPD)](https://ngc.nvidia.com/models/nvidia:tao_lpdnet) and [License Plate Recognition (LPR)](https://ngc.nvidia.com/models/nvidia:tao:lprnet) models in conjunction with the NVIDIA [TAO Toolkit](https://developer.nvidia.com/tao-toolkit). Ready-to-use models allow you to quickly lift off your ALPR project. The resulting TAO-optimized models can be readily deployed using the DeepStream SDK.
_Video. Real-time license plate recognition._
To get started with creating and deploying highly accurate, pretrained models from TAO Toolkit, you need the following resources:
  * [TrafficCamNet](https://ngc.nvidia.com/catalog/models/nvidia:tao:trafficcamnet) or [DashCamNet](https://ngc.nvidia.com/catalog/models/nvidia:tao:dashcamnet) model from NGC to detect vehicles
  * [License plate detection (LPD) model](https://ngc.nvidia.com/models/nvidia:tao_lpdnet) to detect license plates
  * [License plate recognition (LPR) model](https://ngc.nvidia.com/models/nvidia:tao_lprnet) to translate the image to text


All the pretrained models are free and readily available on [NGC](http://ngc.nvidia.com). TAO Toolkit provides two LPD models and two LPR models: one set trained on US license plates and another trained on license plates in China. For more information, see the LPD and LPR model cards.
_Figure 2. ALPR application pipeline_ _for a real-time car license plate recognition application._
You use TAO Toolkit through the `tao-launcher` interface for training. To run the TAO Toolkit launcher, map the `~/tao-experiments` directory on the local machine to the Docker container using the `~/.tao_mounts.json` file. For more information, see [TAO Toolkit Launcher](https://docs.nvidia.com/tao/tao-toolkit/text/tao_launcher.html).
Install the TAO Toolkit launcher:

```
pip3 install nvidia-pyindex
pip3 install nvidia-tao
```

Create the `~/.tao_mounts.json` file and add the following content inside:

```
{
    "Mounts": [
        {
            "source": "/home/<username>/tao-experiments",
            "destination": "/workspace/tao-experiments"
        },
        {
            "source": "/home/<username>/openalpr",
            "destination": "/workspace/openalpr"
        }

    ]
}
```

Mount the path `/home/<username>/tao-experiments` on the host machine to be the path `/workspace/tao-experiments` inside the container. You also mount the path `/home/<username>/openalpr` on the host machine to be the path `/workspace/openalpr` inside the container.
## License plate detection 
In this section, we walk you through how to take the pretrained US-based LPD model from NGC and fine-tune the model using the OpenALPR dataset.
### Dataset 
Use the [OpenALPR benchmark](https://github.com/openalpr/benchmarks) as your experimental dataset. You take the LPD pretrained model from NGC and fine-tune it on the OpenALPR dataset.
### Algorithm introduction 
The LPD model is based on the Detectnet_v2 network from TAO Toolkit. The training algorithm optimizes the network to minimize the localization and confidence loss for the objects. 
The training is carried out in two phases. In the first phase, the network is trained with regularization to facilitate pruning. Following the first phase, you prune the network removing channels whose kernel norms are below the pruning threshold. In the second phase, the pruned network is retrained. Regularization is not included during the second phase.
### Training the LPD model 
Set up your [NGC](https://ngc.nvidia.com/) account and [install the TAO Toolkit launcher](https://docs.nvidia.com/tao/tao-toolkit/text/tao_launcher.html). To fine-tune the LPD model, download the LPD notebook from NGC. Then, [download the NGC LPD pretrained model](https://ngc.nvidia.com/models/nvidia:tao:lprnet) (usa_unpruned.tlt).
#### Prepare the dataset
First, sync the OpenALPR benchmark:

```
$ git clone https://github.com/openalpr/benchmarks benchmarks
```

Next, run the following command to download the dataset and resize images/labels. [Download lpd_prepare_data.py](https://github.com/NVIDIA-AI-IOT/deepstream_tlt_apps/blob/release/tlt3.0/misc/dev_blog/LPDR/lpd/lpd_prepare_data.py):

```
$ python lpd_prepare_data.py --input_dir benchmarks/endtoend/us --output_dir  lpd  --target_width  640 --target_height 480
```

Split the data into two parts: 80% for the training set and 20% for the validation set. Run the following command to split the dataset randomly and generate tfrecords. This command uses a spec file called [SPECS_tfrecord.txt](https://github.com/NVIDIA-AI-IOT/deepstream_tlt_apps/blob/release/tlt3.0/misc/dev_blog/LPDR/lpd/SPECS_tfrecord.txt).

```
$ tao detectnet_v2 dataset_convert -d /workspace/openalpr/SPECS_tfrecord.txt -o /workspace/openalpr/lpd_tfrecord/lpd
```

#### Configure the spec file
After you prepare the dataset, configure the parameters for training by [downloading the training spec](https://github.com/NVIDIA-AI-IOT/deepstream_tlt_apps/blob/release/tlt3.0/misc/dev_blog/LPDR/lpd/SPECS_train.txt). You are specifying the NGC pretrained model for LPD using the `pretrained_model_file` parameter in the spec file. Set the batch-size to 4 and run 120 epochs for training. The training model is evaluated with the validation set every 10 epochs.
#### Training
Run the following command to start fine-tuning on the OpenALPR data:

```
$ tao detectnet_v2 train -e /workspace/openalpr/SPECS_train.txt -r /workspace/openalpr/exp_unpruned -k nvidia_tlt
```

After the training completes, you see the following log that shows the average precision (AP) on the validation set:

```
class name      average precision (in %)
------------  --------------------------
lpd                     82.2808
```

After training, you can also prune your trained model to reduce the size of the model. Pruning is not shown in this post. For more information, see [Pruning the model](https://docs.nvidia.com/tao/tao-toolkit/text/pruning_model.html) or [Training with Custom Pretrained Models Using the NVIDIA Transfer Learning Toolkit](https://developer.nvidia.com/blog/training-custom-pretrained-models-using-tlt/).
#### Exporting the model
After training, export the model for deployment. The format for deployment is `.etlt` or encrypted TAO Toolkit. You encrypt the exported model with a key and use the key to decrypt the model during deployment.
To run inference using INT8 precision, you can also generate an INT8 calibration table in the model export step. The encrypted TAO Toolkit file can be directly consumed in the DeepStream SDK.
To export the LPD model in INT8, use the following command. This command first calibrates the model for INT8 using calibration images specified by the `--cal_image_dir` option. The encryption key for this model is specified by the `-k` option. This can be any string. The exported .etlt file and calibration cache is specified by the `-o` and the `--cal_cache_file` option, respectively. To learn more about all the options with model export, see the [TAO Toolkit DetectNet_v2 documentation](https://docs.nvidia.com/tao/tao-toolkit/text/object_detection/detectnet_v2.html).

```
$ tao detectnet_v2 export -m /workspace/openalpr/exp_unpruned/weights/model.tlt -o /workspace/openalpr/export/unpruned_model.etlt --cal_cache_file /workspace/openalpr/export/calibration.bin -e /workspace/openalpr/SPECS_train.txt -k nvidia_tao --cal_image_dir /workspace/openalpr/lpd/data/image --data_type int8 --batch_size 4 --batches 10 –-engine_file /workspace/openalpr/export/unpruned_int8.trt
```

### Accuracy of the trained LPD model 
The pretrained model provides a great starting point for training and fine-tuning on your own dataset. For comparison, we have trained two models: one trained using the LPD pretrained model and the second trained from scratch. The following table shows the mean average precision (mAP) comparison of the two models. By using the pretrained model, you can reach your target accuracy much faster with a smaller dataset. If you were to train from scratch, you would need a much larger dataset and would need to run it with longer to achieve similar accuracy.
You could use the following command in TAO Toolkit Docker to run an evaluation on the validation dataset specified in the experiments config file:

```
$ tao detectnet_v2 evaluate -m /workspace/openalpr/exp_unpruned/weights/model.tlt -k nvidia_tao -e /workspace/openalpr/SPECS_train.txt
```
  
| Model  | Epochs  | batch-size  | mAP  |  
| --- | --- | --- | --- |  
| LPD: Training from scratch  | 120  | 4  | 53.11%  |  
| LPD: Fine-tuning a pretrained model  | 120  | 4  | 82.28%  |  
_Table 1. Accuracy of using the pretrained model vs. training from scratch._
## License plate recognition 
In this section, we go into the details of the LPR model training. NVIDIA provides LPRNet models trained on US license plates and Chinese license plates. You can find the details of these models in the [model card](https://ngc.nvidia.com/models/nvidia:tao_lprnet). You use LPRNet trained on US license plates as the starting point for fine-tuning in the following section.
### Dataset 
You train and evaluate the LPRNet on the OpenALPR US images dataset as well. Split it to 80% (177 images) for training and 20% (44 images) for validation.
### Algorithm introduction 
For the license plate recognition task, you predict the characters in sequence of a license plate image. Just like other [computer vision](https://developer.nvidia.com/computer-vision) tasks, you first extract the image features. Take advantage of widely used DNN architecture, such as ResNet 10/18, to be the backbone of LPRNet. The original stride of the ResNet network is 32 but to make it more applicable to the small spatial size of the license plate image, tune the stride from 32 to 4. Then, feed the image feature into a classifier. Unlike the normal image classification task, in which the model only gives a single class ID for one image, the LPRNet model produces a sequence of class IDs. The image feature is divided into slices along the horizontal dimension and each slice is assigned a character ID in the prediction.
Finally, use the connectionist temporal classification (CTC) loss to train this sequence classifier. The training algorithm optimizes the network to minimize the CTC loss between the ground truth characters sequence of a license plate and predicted characters sequence.
In general, LPRNet is a sequence classification model with a tuned ResNet backbone. It takes the image as network input and produces sequence output. Then, the license plate is decoded from the sequence output using a CTC decoder based on a greedy decoding method.
_Figure 3. LPR model architecture._
### Training the LPR model 
Training LPRNet using TAO Toolkit requires no code development from your side. You prepare a dataset, set the experiment config, and then run the command.
#### Prepare the data
You process data in the `/home/<username>/tao-experiments/` path of the local machine and use the mapped path in Docker for `tao-launcher`. First, clone the OpenALPR benchmark from [openalpr/benchmarks](https://github.com/openalpr/benchmarks):

```
$ git clone https://github.com/openalpr/benchmarks benchmarks
```

Next, preprocess the downloaded dataset and split it into train/val using the [preprocess_openalpr_benchmark.py script](https://github.com/NVIDIA-AI-IOT/deepstream_tlt_apps/blob/release/tlt3.0/misc/dev_blog/LPDR/lpr/preprocess_openalpr_benchmark.py). 

```
$ python preprocess_openalpr_benchmark.py --input_dir=./benchmarks/endtoend/us --output_dir=./data/openalpr
```

After preprocessing, the OpenALPR dataset is in the format that TAO Toolkit requires. Each cropped license plate image has a corresponding label text file that contains the ground truth of the license plate image. You also create a `characters_list.txt` file that is a dictionary of all the [characters found in the US license plates](https://github.com/NVIDIA-AI-IOT/deepstream_tlt_apps/blob/release/tlt3.0/misc/dev_blog/LPDR/lpr/us_lp_characters.txt).
#### Experiments config
The experiments config file defines the hyperparameters for LPRNet model’s architecture, training, and evaluation. Download a [sample LPR training config](https://github.com/NVIDIA-AI-IOT/deepstream_tlt_apps/blob/release/tlt3.0/misc/dev_blog/LPDR/lpr/tutorial_spec.txt) file and place it in the `/home/`<username>`/tao-experiments/lprnet` path. Use this config for fine-tuning on US LPRNet.
In this config, you define a LPRNet model with a tuned ResNet18 backbone, which is your baseline. Train the model for 24 epochs with batch size 32, L2 regularization of 0.0005, and a `soft_start_annealing_schedule` to apply a variable learning rate during training. For more information about the parameters in the experiment config file, see the [TAO Toolkit User Guide](https://docs.nvidia.com/tao/tao-toolkit/text/character_recognition/lprnet.html).
We also provide a [spec](https://github.com/NVIDIA-AI-IOT/deepstream_tlt_apps/blob/release/tlt3.0/misc/dev_blog/LPDR/lpr/tutorial_spec_scratch.txt) file to train from scratch. Compared with fine-tuning config, you must enlarge the epoch number and learning rate. Though this is not a recommended way for training, we provided it for comparison.
#### Training
When the dataset and experiment spec are ready, start your training in TAO Toolkit. Use the following command to train a LPRNet with a single GPU and the US LPRNet model as pretrained weights:

```
$ tao lprnet train -e /workspace/tao-experiments/lprnet/tutorial_spec.txt -r /workspace/tao-experiments/lprnet/ -k nvidia_tao -m /workspace/tao-experiments/lprnet/us_lprnet_baseline18_trainable.tlt
```

TAO Toolkit also supports multi-GPU training (data parallelism) and [automatic mixed precision (AMP)](https://developer.nvidia.com/automatic-mixed-precision). To boost the training speed, you could run multi-GPU with option `--gpus <num_gpus>` and mixed precision training with option `--use_amp`. The training log, which includes accuracy on validation dataset, training loss, and learning rate, is saved in .csv format in the `<results_dir>` directory. The following code example shows the training log with pretrained weights:

```
epoch,accuracy,loss,lr
0,nan,1.085993747589952,1e-05
1,nan,0.9726232198503731,1e-05
2,nan,0.9452087508756563,1e-05
3,nan,0.7897920507495686,1e-05
4,0.8409090909090909,0.5753771635772145,1e-05               
…….
```

#### Exporting the model
To deploy the LPR model in DeepStream or other applications, export it to the .etlt format. Currently, LPR only supports FP32 and FP16 precision. Compared with LPD’s model export command, LPR is much simpler:

```
$ tao lprnet export -m /workspace/tao-experiments/lprnet/weights/lprnet_epoch-24.tlt -k nvidia_tao -e /workspace/tao-experiments/lprnet/tutorial_spec.txt
```

The output .etlt model is saved in the same directory as the trained .tlt model.
### Accuracy of the trained LPR model 
The evaluation metric of LPR is the accuracy of license plate recognition. A recognition is regarded as accurate if all the characters and the sequence in the license plate are correct. You can use the following command in TAO Toolkit Docker to run an evaluation on the validation dataset specified in the experiments config file:

```
$ tao lprnet evaluate -m /workspace/tao-experiments/lprnet/weights/lprnet_epoch-24.tlt -k nvidia_tao -e /workspace/tao-experiments/lprnet/tutorial_spec.txt
```

The following table shows the accuracy comparison of the model trained from scratch and the model trained with the LPRNet pretrained model.  
| Model  | Epochs  | Train Accuracy  | Val Accuracy  |  
| --- | --- | --- | --- |  
| baseline18_unpruned_from_scratch  | 100  | 0%  | 0%  |  
| baseline18_unpruned_from_pretrained  | 24  | 98.87%  | 90.90%  |  
_Table 2. Accuracy of using the pretrained model vs. training from scratch._
With the pretrained model, you can reach high accuracy with a small number of epochs. Conversely, when training from scratch, your model hasn’t even begun to converge with a 4x increase in the number of epochs. This means that you must increase the dataset significantly, which increases the training time and cost.
## Deploying LPD and LPR using the DeepStream SDK 
In this section, we walk you through the steps to deploy the LPD and LPR models in DeepStream. We have provided a [sample DeepStream application](https://github.com/NVIDIA-AI-IOT/ds-lpr-sample). The LPD/LPR sample application builds a pipeline for multiple video streams input and infer the batched videos with cascading models to detect cars and their license plates and to recognize characters.
_Figure 4. DeepStream pipeline of LPD/LPR sample application._
The source code for the sample application is constructed in two parts:
  * `lpr-test-sample—`The main application of the LPD/LPR sample, which constructs the Gstreamer pipeline with DeepStream plugins.
  * `nvinfer_custom_lpr_parser—`The customized DeepStream `nvinfer` plugin classifier parser library for LPR model. The default DeepStream `nvinfer` classifier can only support confidence parsing and get labels from the label file which is configured by `labelfile-path` parameter in the `nvinfer` configuration file. Because the LPR model outputs the `argmax` and confidence with two layers, the customized output parsing function is needed to parse the LPR output layers and generate correct labels for car plate strings. The customized parser function should look-up the designated dictionary to find characters with `argmax` values and confidence values, and then combine the characters into car plate string as the label.


### Download and prepare the models 
For this application, you need three models from TAO Toolkit:
  * TrafficCamNet detects vehicles.
  * LPD detects license plates.
  * LPR recognizes the characters.


All models can be downloaded from NVIDIA NGC. Alternatively, if you followed the training steps in the earlier two sections, you could also use your trained LPD and LPR model instead.
Download the TrafficCamNet model:

```
mkdir -p /opt/nvidia/deepstream/deepstream-5.0/samples/models/tao_pretrained_models/trafficcamnet
 cd /opt/nvidia/deepstream/deepstream-5.0/samples/models/tao_pretrained_models/trafficcamnet
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/trafficcamnet/versions/pruned_v1.0/files/trafficnet_int8.txt
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/trafficcamnet/versions/pruned_v1.0/files/resnet18_trafficcamnet_pruned.etlt
```

Download the LPD model:

```
mkdir -p /opt/nvidia/deepstream/deepstream-5.0/samples/models/LP/LPD
cd /opt/nvidia/deepstream/deepstream-5.0/samples/models/LP/LPD
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/lpdnet/versions/pruned_v1.0/files/usa_pruned.etlt
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/lpdnet/versions/pruned_v1.0/files/usa_lpd_cal.bin
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/lpdnet/versions/pruned_v1.0/files/usa_lpd_label.txt
```

Download the LPR model:

```
mkdir -p /opt/nvidia/deepstream/deepstream-5.0/samples/models/LP/LPR
cd /opt/nvidia/deepstream/deepstream-5.0/samples/models/LP/LPR
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/lprnet/versions/deployable_v1.0/files/us_lprnet_baseline18_deployable.etlt
#create an empty label file
echo > labels_us.txt
```

With DeepStreamSDK 5.x, the `gst-nvinfer` plugin cannot automatically generate TensorRT engine from the ONNX format from TAO Toolkit. The LPR model is exported in encrypted ONNX format from TAO Toolkit, and it’s a limitation for the LPR model. The LPD model is in legacy encrypted UFF format and automatically works with DeepStream. The engine files for the LPR model should be generated using the `tao-converter tool`. Download the latest `tao-converter` for your appropriate hardware and CUDA or cuDNN version from the [TAO Toolkit getting started ](https://developer.nvidia.com/tao-toolkit-get-started)page.  
| **Platform**  | **Compute**  |  
| --- | --- |  
| x86 + GPU  | [CUDA 10.2 / cuDNN 8.0 / TensorRT 7.1](https://developer.nvidia.com/cuda102-cudnn80-trt71)  |  
| x86 + GPU  | [CUDA 10.2 / cuDNN 8.0 / TensorRT 7.2](https://developer.nvidia.com/cuda102-cudnn80-trt72)  |  
| x86 + GPU  | [CUDA 11.0 / cuDNN 8.0 / TensorRT 7.1](https://developer.nvidia.com/cuda110-cudnn80-trt71)  |  
| x86 + GPU  | [CUDA 11.0 / cuDNN 8.0 / TensorRT 7.2](https://developer.nvidia.com/cuda110-cudnn80-trt72)  |  
| Jetson  |  
| Jetson  |  
_Table 1. Download locations by platform and hardware._
Convert the encrypted LPR ONNX model to a TAO Toolkit engine:

```
tao-converter -k nvidia_tlt -p image_input,1x3x48x96,4x3x48x96,16x3x48x96 ./us_lprnet_baseline18_deployable.etlt -t fp16 -e /opt/nvidia/deepstream/deepstream-5.0/samples/models/LP/LPR/lpr_us_onnx_b16.engine
```

### Build and run the sample application 
Download the sample code from the [NVIDIA-AI-IOT/deepstream_lpr_app](https://github.com/NVIDIA-AI-IOT/deepstream_lpr_app) GitHub repo and build the application. 
Copy the folder of `nvinfer_custom_lpr_parser` to the board and build the code:

```
cd  nvinfer_custom_lpr_parser
make
```

Copy the generated `libnvdsinfer_custom_impl_lpr.so` file to the `/opt/nvidia/deepstream/deepstream-5.0/lib/` directory.
Copy the folder of `lpr-test-sample` to your device and build the code. The sample application `lpt-test-app` is generated.

```
cd lpr-test-sample
make
```

Modify the `nvinfer` configuration files for TrafficCamNet, LPD and LPR with the actual model path and names. The config file for TrafficCamNet is provided in DeepStream SDK under the following path:

```
/opt/nvidia/deepstream/deepstream-5.0/samples/models/tao_pretrained_models/trafficcamnet.txt
```

The sample `lpd_config.txt` and `lpr_config_sgie_us.txt` files can be found and . Note the `parse-classifier-func-name` and `custom-lib-path`. This uses the new `nvinfer` LPR library from step 1.
Prepare the dictionary file for the OCR according to the trained TAO Toolkit LPR model. The dictionary file name should be `dict.txt`. Create `dict.txt` by using the US version.

```
$ cp dict_us.txt dict.txt
```

Run the sample application.

```
lpr-test-app [language mode:1-us 2-chinese]
  [sink mode:1-output as 264 stream file 2-no output 3-display on screen]
  [ROI enable:0-disable ROI 1-enable ROI]
  [input mp4 file path and name] [input mp4 file path and name] ... [input mp4 file path and name]
  [output 264 file path and name]
```

For example:

```
$ lpr-test-app 1 3 0 file1.mp4 file2.mp4 output.264
```

## Performance 
The following table just shows the inference throughput in frames per seconds (FPS) of the US LPD pruned model, which is trained on a proprietary dataset with over 45,000 US car images. The performance varies with different input size, pruning ratio, devices, etc. The performance is for the pruned version of the model that is available on NGC and not on the model trained in earlier sections.  
| Device  | Input Size(CHW)  | Precision  | Batch Size  | FPS  |  
| --- | --- | --- | --- | --- |  
| Jetson Nano  | 3x480x640  | FP16  | 1  | 66  |  
| Jetson NX  | 3x480x640  | INT8  | 1  | 461  |  
| Jetson Xavier  | 3x480x640  | INT8  | 1  | 913  |  
| T4  | 3x480x640  | INT8  | 1  | 2748  |  
_Table 3. Inference performance of license plate detection._
### LPR standalone performance 
The following table shows the inference performance of the LPR trained on US license plates on different devices. We profiled the model inference with the `trtexec` command of TensorRT.  
| Device  | Input Size  | Precision  | Batch Size  | FPS  |  
| --- | --- | --- | --- | --- |  
| Jetson Nano  | 3x48x96  | FP16  | 32  | 16  |  
| Jetson NX  | 3x48x96  | FP16  | 32  | 600  |  
| Jetson Xavier  | 3x48x96  | FP16  | 64  | 1021  |  
| T4  | 3x48x96  | FP16  | 128  | 3821  |  
_Table 4. Inference performance of license plate recognition._
### Sample application performance 
The full pipeline of this sample application runs three different DNN models. You use pretrained TrafficCamNet in TAO Toolkit for car detection. LPD and LPR are pretrained with the NVIDIA training dataset of US license plates.
The following test is done with 1080p (1920×1080) resolution videos with the sample LPR application. The following table shows the end-to-end performance of processing the entire video analytic pipeline with three DNN models, starting from ingesting video data to rendering the metadata on the frames. The data is collected on different devices.  
| Device  | Number of streams  | Batch Size  | Total FPS  |  
| --- | --- | --- | --- |  
| Jetson Nano  | 1  | 1  | 9.2  |  
| Jetson NX  | 3  | 3  | 80.31  |  
| Jetson Xavier  | 5  | 5  | 146.43  |  
| T4  | 14  | 14  | 447.15  |  
_Table 5. End-to-end inference of ALPR application using DeepStream._
## Summary 
In this post, we introduced an end-to-end AI solution for automatic license plate recognition. This solution covers all the aspects of developing an intelligent video analysis pipeline: training deep neural network models with TAO Toolkit to deploying the trained models in DeepStream SDK.
For training, you don’t need the expertise to build your own DNN and optimize the model. TAO Toolkit offers a simplified way to train your model: All you have to do is prepare the dataset and set the config files. Besides, you can take advantage of the highly accurate pretrained models in TAO Toolkit instead of random initialization.
For deployment, DeepStream optimizes the system resources for video decoding, image preprocessing and inference, providing you with highest channel density for real-time video analytics. You can quickly deploy your trained models into a multi-stream video analytics pipeline with DeepStream with minimal effort.
Start your next AI project with NVIDIA pretrained models and train using TAO Toolkit.
  * [NVIDIA-AI-IOT/deepstream_lpr_app reference application](https://github.com/NVIDIA-AI-IOT/deepstream_lpr_app)


For more information, see the following resources:
  * Use the [TAO Toolkit Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/intelligent-video-analytics/transfer-learning-toolkit/17) or [DeepStream Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/intelligent-video-analytics/deepstream-sdk/15) for questions or feedback


[Computer Vision / Video Analytics](https://developer.nvidia.com/blog/category/computer-vision/) | [Data Science](https://developer.nvidia.com/blog/category/data-science/) | [Robotics](https://developer.nvidia.com/blog/category/robotics/) | [Smart Cities / Spaces](https://developer.nvidia.com/blog/recent-posts/?industry=Smart+Cities+%2F+Spaces) | [Metropolis](https://developer.nvidia.com/blog/recent-posts/?products=Metropolis) | [TAO Toolkit](https://developer.nvidia.com/blog/recent-posts/?products=TAO+Toolkit) | [Tutorial](https://developer.nvidia.com/blog/recent-posts/?content_types=Tutorial) | [featured](https://developer.nvidia.com/blog/tag/featured/) | [Pre-Trained / Foundation Models](https://developer.nvidia.com/blog/tag/pre-trained-foundation-models/)
##  About the Authors 
**About Yue Zhu** Yue Zhu is a senior system software engineer at NVIDIA, focusing on perception algorithms development for automotive driving, intelligent video analytics, and robotics. He received a B.S and M.S in computer science from University of Electronic Science and Technology of China. 
**About Morgan Huang** Morgan Huang is a senior software testing engineer at NVIDIA, focusing on accuracy or performance issue analysis and optimization. He oversees the NVIDIA TLT forum.He holds a master’s degree in electrical engineering from Beijing Jiaotong University, China. 
**About Fei Chen** Fei Chen is a senior system software engineer at NVIDIA, focusing on DeepStream and multimedia software for intelligent video analytics. 
## Notable Replies
  1. The whole process is too complicated. Too many dependencies.
  2. Could you share the algorithm or implementation of the LPRNet?
  3. I am trying to “Convert the encrypted LPR ONNX model to a TLT engine”, however, in running the command:
> ./tlt-converter -k nvidia_tlt -p image_input,1x3x48x96,4x3x48x96,16x3x48x96 ./us_lprnet_baseline18_deployable.etltunpruned.etlt -t fp16 -e /opt/nvidia/deepstream/deepstream-5.1/samples/models/LP/LPR/lpr_us_onnx_b16.engine
I am receiving the error: `Error: no input dimensions given`
Do you happen to know the dimensions we should be using for this? cc/ [@Morganh](https://forums.developer.nvidia.com/u/morganh) (the github repo seems to have the same issues as noted [here](https://github.com/NVIDIA-AI-IOT/deepstream_lpr_app/issues)
  4. [@bbb](https://forums.developer.nvidia.com/u/bbb) Can you run following command $ ./tlt-converter -h
Did you download the latest version of tlt-converter ? See [Overview — TAO Toolkit 3.22.05 documentation](https://docs.nvidia.com/metropolis/TLT/tlt-user-guide/text/overview.html#id1) or [Overview — TAO Toolkit 3.22.05 documentation](https://docs.nvidia.com/metropolis/TLT/tlt-user-guide/text/overview.html#id2)


###  Continue the discussion at [forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/creating-a-real-time-license-plate-detection-and-recognition-app/169445)
35 more replies
#### Participants
NVIDIA uses cookies to improve your experience on our web site. We and our third-party partners also use cookies and other tools to collect and record information you provide as well as information about your interactions with our websites for performance improvement, analytics, and to assist in marketing efforts. By clicking "Accept All", you consent to our use of cookies and other tools as described in our [Cookie Policy](https://www.nvidia.com/en-us/about-nvidia/cookie-policy/). You can manage your cookie settings by clicking on "Manage Settings." By continuing to use this site or by clicking one of the buttons below, you agree to our [Terms of Service](https://www.nvidia.com/en-us/about-nvidia/terms-of-service/) (which contains important waivers). Please see our [Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) for more information on our privacy practices.
We have detected the Global Privacy Control (GPC) signal and recorded your rejection of all optional cookies on this site for this browser. You can manage your cookie settings by clicking on "Manage Settings". Please see our [Cookie Policy](https://www.nvidia.com/en-us/about-nvidia/cookie-policy/) for more information. To opt out of non-cookie personal information "sales" / "sharing" for targeted advertising purposes, please visit the [NVIDIA Preference Center](https://www.nvidia.com/en-us/about-nvidia/privacy-center/). Please see our [Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) for more information on our privacy practices.
We have detected the Global Privacy Control Signal (GPC) and have opted you out of all optional cookies on this browser. You can manage your cookie settings by clicking on "Manage Settings". Please see our [Cookie Policy](https://www.nvidia.com/en-us/about-nvidia/cookie-policy/) for more information. We have also opted you out of "sharing"/"sales" of personal information outside of cookies. You can manage these settings in the NVIDIA [NVIDIA Preference Center](https://www.nvidia.com/en-us/privacy-center/). Please see our [Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) for more information.
We have detected the Global Privacy Control Signal (GPC) and have opted you out of all optional cookies on this browser. You can manage your cookie settings by clicking on "Manage Settings". Please see our [Cookie Policy](https://www.nvidia.com/en-us/about-nvidia/cookie-policy/) for more information. We have also opted you out of "sharing"/"sales" of personal information outside of cookies which overrides at least one of your previous settings. You can manage them in the [NVIDIA Preference Center](https://www.nvidia.com/en-us/privacy-center/). Please see our [Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) for more information.
Manage Settings
Reject Optional Accept All
Cookie Settings
We and our third-party partners (including social media, advertising, and analytics partners) use cookies and other tracking technologies to collect, store, monitor, and process certain information about you when you visit our website. The information collected might relate to you, your preferences, or your device. We use that information to make the site work, analyze performance and traffic on our website, provide a more personalized web experience, and assist in our marketing efforts.Under certain privacy laws, you have the right to direct us not to "sell" or "share" your personal information for targeted advertising. To opt-out of the "sale" and "sharing" of personal information through cookies, you must opt-out of optional cookies using the toggles below. To opt out of the "sale" and "sharing" of data collected by other means (e.g., online forms) you must also update your data sharing preferences through the [NVIDIA Preference Center](https://www.nvidia.com/en-us/about-nvidia/privacy-center/).Click on the different category headings below to find out more and change the settings according to your preference. You cannot opt out of Required Cookies as they are deployed to ensure the proper functioning of our website (such as prompting the cookie banner and remembering your settings, etc.). By clicking "Save and Accept" or "Decline All" at the bottom, you consent to the use of cookies and other tools as described in our [Cookie Policy](https://www.nvidia.com/en-us/about-nvidia/cookie-policy/) in accordance with your settings and accept our [Terms of Service](https://www.nvidia.com/en-us/about-nvidia/terms-of-service/) (which contains important waivers). For more information about our privacy practices, please see our [Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/).
Required Cookies
Always Active
These cookies enable core functionality such as security, network management, and accessibility. These cookies are required for the site to function and cannot be turned off.
Cookies Details
Performance Cookies
Performance Cookies
These cookies are used to provide quantitative measures of our website visitors, such as the number of times you visit, time on page, your mouse movements, scrolling, clicks and keystroke activity on the websites; other browsing, search, or product research behavior; and what brought you to our site. These cookies may store a unique ID so that our system will remember you when you return. Information collected with these cookies is used to measure and find ways to improve website performance.
Cookies Details
Personalization Cookies
Personalization Cookies
These cookies collect data about how you have interacted with our website to help us improve your web experience, such as which pages you have visited. These cookies may store a unique ID so that our system will remember you when you return. They may be set by us or by third party providers whose services we have added to our pages. These cookies enable us to provide enhanced website functionality and personalization as well as make the marketing messages we send to you more relevant to your interests. If you do not allow these cookies, then some or all of these services may not function properly.
Cookies Details
Advertising Cookies
Advertising Cookies
These cookies record your visit to our websites, the pages you have visited and the links you have followed to influence the advertisements that you see on other websites. These cookies and the information they collect may be managed by other companies, including our advertising partners, and may be used to build a profile of your interests and show you relevant advertising on other sites. We and our advertising partners will use this information to make our websites and the advertising displayed on it, more relevant to your interests.
Cookies Details
Back Button
Cookie List
Search Icon
Filter Icon
Clear
  * checkbox label label


Consent Leg.Interest
checkbox label label
checkbox label label
checkbox label label
Decline All Save and Accept
