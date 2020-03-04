# How To Use NNCASE(V0.2.0)

git clone https://github.com/sipeed/Maix_Toolbox.git

cd ./Maix_Toolbox/

mkdir -p ncc

cd ./ncc

wget https://github.com/kendryte/nncase/releases/download/v0.2.0-beta2/ncc_linux_x86_64.tar.xz

(wget https://github.com/kendryte/nncase/releases/download/v0.1.0-rc5/ncc-linux-x86_64.tar.xz)

tar -Jxf ncc_-_linux_x86_64.tar.xz
rm ncc_linux_x86_64.tar.xz

cd ./..

./ncc/ncc compile ~/Segmentation4M5StickV/VOC2012/TestNet_VOC2012_npz.tflite ~/Segmentation4M5StickV/VOC2012/TestNet_VOC2012_npz.kmodel -i tflite -o kmodel --dataset ~/Segmentation4M5StickV/VOC2012/data/JPEGImages_Sample/


  1. Import graph...  
  2. Optimize Pass 1...  
  3. Optimize Pass 2...  
  4. Quantize...  
    4.1. Add quantization checkpoints...  
    4.2. Get activation ranges...  
    Plan buffers...  
    Run calibration...  
    [==================================================] 100% 7.117s
    4.5. Quantize graph...  
  5. Lowering...  
  6. Generate code...  
    Plan buffers...  
    Emit code...  
  Main memory usage: 3543040 B  

  SUMMARY  
  INPUTS  
  0	input_1	1x3x128x160  
  OUTPUTS  
  0	Identity	1x21x128x160  


# How To Use NNCASE(V0.1.0)  

mkdir -p ncc

cd ./ncc

(wget https://github.com/kendryte/nncase/releases/download/v0.1.0-rc5/ncc-linux-x86_64.tar.xz)

tar -Jxf ncc-linux-x86_64.tar.xz
rm ncc-linux-x86_64.tar.xz

cd ./..

./ncc/ncc ~/Segmentation4M5StickV/VOC2012/Model_V0_1.tflite ~/Segmentation4M5StickV/VOC2012/Model_V0_1.kmodel -i tflite -o kmodel --dataset ~/Segmentation4M5StickV/VOC2012/data/JPEGImages_Sample/
