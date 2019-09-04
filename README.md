# xiaocar

This project attempts to recreate Nvidia's End-to-End Learning [Paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) using a Raspberry Pi Robot.

The robot was driven inside a living romm using a PS4 controller while it captured images of the environment, and recorded the Left/Right Motors' PWM control signals.

The model's layers were as follows:

- 64x48x3 Input shape
- 3x3x32 2D Convolution
- 1x1 Padding
- ReLu Activation
- 0.25 Dropout
- 3x3x64 2D Convolution
- 1x1 Padding
- RelU Activation
- 0.25 Dropout
- 512 FC Layer
- 64 FC Layer

A mean_squared_error loss with an Adam optimizer were used to regress the images to the Left/Right moto control signals.

Training was done an AWS GPU using Keras with a Tensorflow backend.

Inference was done using a custom compiled TensorFlow directly on the Raspberry Pi 3.


![XiaoCar](demo/demo.gif)

