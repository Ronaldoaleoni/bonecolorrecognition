# Bonedetector

This project was developed using Python 3.10.16 and includes libraries focused on computer vision, deep learning, and numerical data processing.

It is part of the Automated Bone Recognition pipeline and performs the detection and segmentation of bones using Mask R-CNN implemented with Detectron2.

## Libraries Used

- **Python**: 3.10.16
- **cudatoolkit**: 11.3
- **conda install -c nvidia cudnn**: 8.2.1
- **pytorch**: 1.12.0
- **torchvision**: 0.13.0
- **torchaudio**: 0.12.0
- **cython**:  3.1.0
- **pycocotools**: 2.0.8
- **Detectron2**: 0.6
- **OpenCV (cv2)**: 4.11.0
- **NumPy**: 1.26.4
- **Matplotlib:** 3.10.0
- **wheel** 0.45.1
- **pandas** 2.2.3
- **Random**
- **Math**
- **git**
- **git clone https://github.com/facebookresearch/detectron2.git**

 
## Installation

 conda create --name bonecolorecog python=3.10.16

 conda activate bonecolorecog

 cd Desktop

 cd bonecolorrecognition

 conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch -c nvidia

 conda install -c nvidia cudnn=8.2.1

 python -m pip install -r requirements.txt

 conda install git

 git clone https://github.com/facebookresearch/detectron2.git

 cd detectron2

 python -m pip install .

 (If the installation fails, you may need the C++ compiler to build C/CUDA extensions) https://visualstudio.microsoft.com/visual-cpp-build-tools/ and download build tools

 python -m pip install "Pillow<10"
 
 ### **import dependencies and versions**

 **print(torch.__version__)**

     1.12.1      

 **print(torch.cuda.is_available())**

     True    

 **print(torch.version.cuda)** 

     11.3 (PyTorch's CUDA version)     

 **print(torch.backends.cudnn.version())**  

     8200 (cuDNN 8.2) or something near

 **print("detectron2:", detectron2.__version__)**

     0.6

 **print(numpy.__version__)**

     1.26.4

 **print(cv2.__version__)**
 
     4.11.0

## Dataset Setup

The dataset must be in COCO format:

 - [assets](assets/)
     - [images](assets/images/)              # All test or validation images (.jpg, .png)
     - [annotations](assets/annotations.json)      # COCO-style annotations

 - Register the dataset in the script by editing:

     annotation_file_path_test = "<your-path-here>/assets/annotations.json"

     image_directory_path_test = "<your-path-here>/assets/images"

 - Running the Script

## Model Inference and Visualization

 - The first section:

     register_coco_instances(...)

     predictor = DefaultPredictor(test_cfg)


 - creates a predictor and saves visualized detections with bounding boxes and masks into:

     ./test_set_predictions/


 - Each output image shows the predicted bone regions overlayed on the original photo.

## Bone Cropping and Export

The second part of the script automatically:

 - Reads all images from input_dir.

 - Runs inference to get instance masks.

 - Crops each detected bone with optional padding (padding_percentage = 0.1).

 - Resizes and centers each bone onto a 1024×1024 black canvas.

 - Saves each bone as a numbered file:

     0001.png

     0002.png

     0003.png

     ...


 - Modify these variables as needed:

     input_dir = "<your-path-here>/images"

     output_dir = "<your-path-here>/output"

     output_size = 1024

     padding_percentage = 0.1


 - Each file will appear in output_dir:

     ✅ Saved 0001.png

     ✅ Saved 0002.png

     ...

## Citation

Developed by Leoni et al. (2025).

Part of the From Visual Perception to Quantitative Approach: Automated Bone Recognition, Color Clustering, and Munsell Matching for Objective Taphonomic Analysis project.