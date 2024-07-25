# FacemaskAI

FacemaskAI is an algorithm that receives a picture of a person and outputs whether or they have a facemask or not.

I developed this AI in order to solve the problem of disease and viral spread. Facemasks provide a physical barrier between pathogens and people, effectively preventing more infection. Since the beginning of the 2020 global coronavirus pandemic, masks have become very present in our lives. Many times they were mandatory to enter places. Since the end of the pandemic, mask usage has decreased, but they are still required in other areas, such as hospitals, doctors offices, and pedratrician units, amongst others. FacemaskAI will indicate whether a person is following the mask guidelines; this making a safer enviorment.

# The Algorithm

The algorithm works as a classification neural network I used transfer learning to retrain the resnet-18 based imagnet classifier. I ran training on over 12000 images total with two classes mask and nomask on the Nvidia Jetson Nano. After training it for 3 hours it achieved epoch 9 and had a 96.7500% val accuracy. It will output the image labeled with what class it belongs to and its certainty.


![themasktest](https://github.com/user-attachments/assets/1f5584bf-464a-49b6-bad4-4b24298f52d9)

This image shows the output of the classification of the FacemaskAI, correctly classifying the person wearing a mask into the mask class with an 84.42% certainty.

# Running this project

1. Navigate to the /home/nvidia/jetson-inference/python/training/classification
2. Set bash enviorment variables "NET=models/mask_nomask" and "DATASET = data/mask_nomask"
3. Run the command "imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/nomask/(imageofmaskfromtestdata.png) (nameofwhereitwillbesaved.png)"
4. The file will now appeare in /home/nvidia/jetson-inference/python/training/classification
