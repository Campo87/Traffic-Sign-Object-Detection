# Intro-ML-Semester-Project
This is a semester project for the "Introduction to Machine Learning" graduate class I took in Fall 2021. I choose traffic sign object detection for my project, specifically Stop, Do-not-enter, Yield, and Speed limit signs. TF2 object detection libraries were used for training and testing the model. The dataset was manually collected from Google Maps, and bounding boxes were generated using ImgLib. Training setup and TF2 installation scripts were based on [this](https://github.com/nicknochnack/TFODCourse/blob/main/2.%20Training%20and%20Detection.ipynb) GitHub tutorial. All training and evaluation was done using Google Collab cloud services.

### Performance
|              Model               | Recall |   mAP  | Steps |
|----------------------------------|--------|--------|-------|
| SSD MobileNet V2 FPNLite 320x320 | 0.8625 | 0.8552 |  25k  |
| SSD MobileNet V2 FPNLite 640x640 | 0.8625 | 0.9163 |  25k  |
| EfficientDet D0 512x512          | 0.8687 | 0.8552 |  7k   |
| EfficientDet D1 640x640          | 0.8552 | 0.9050 |  7k   |

Improvements: More data!
