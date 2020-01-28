# Efficient Saliency and FastCAM 

FastCAM creates a saliency map using SMOE Scale saliency maps as described in our paper on 
[ArXiv:1911.11293](https://arxiv.org/abs/1911.11293). We obtain a highly significant speed-up by replacing
the Guided Backprop component typically used alongside GradCAM with our SMOE Scale saliency map.
Additionally, the expected accuracy of the saliency map is increased slightly. Thus, **FastCAM is three orders of magnitude faster and a little bit more accurate than GradCAM+Guided Backprop with SmoothGrad.** 

![Example output images](https://raw.githubusercontent.com/LLNL/fastcam/master/mdimg/fast-cam.ILSVRC2012_val_00049934.jpg)
                         
## Installation

The FastCAM package runs on **Python 3.x**. The package should run on **Python 2.x**. However, since 
the end of product life for 2.x has been announced, we will not actively support it going forward. 
All extra requirements are available through *pip* installation. On *IBM Power* based architecture, 
some packages may have to be hand installed, but it's totally doable. 

The primary functionality is demonstrated using a **Jupyter notebook**. By following it, you should be
able to see how to use FastCAM on your own deep network. 

### Required Packages

When you run the installation, these packages should automatically install for you. 

	numpy
	jupyter
	torch
	torchvision
	opencv-python
	pytorch_gradcam

### Install and Run the Demo

![Get Ready for EXCITEMENT](https://steemitimages.com/p/DVAkPJXe6RxaMiozqQxRKBpPCPSqM5k9eEaBqfuGYnq1rZoVgJfgBwH61WPbdCwxa7N5TvBS59Jxtv?format=match&mode=fit&width=640)

These are our recommended installation steps:

	git clone git@github.com:LLNL/fastcam.git 
	cd fastcam 
	python3 -m venv venv3 
	source venv3/bin/activate
	pip install -r requirements.txt
	
You may need to add the path of your libraries because of a bug in PyTorch GradCAM. So, for instance, do this:

	export PYTHONPATH=$PYTHONPATH:/path/to/my/python/lib/python3.7/site-packages/

See notes below for more details. 

Next you will need to start the jupyter notebook:

	jupyter notebook
	
It should start the jupyter web service and create a notebook instance in your browser. You can then click on

	demo_fast-cam.ipynb
	
To run the notebook, click on the double arrow (fast forward) button at the top of the web page. 

![Example output images](https://raw.githubusercontent.com/LLNL/fastcam/master/mdimg/option.jpg)

### Installation Notes

**1. You don't need a GPU**

Because FastCAM is ... well ... fast, you can install and run the demo on a five-year-old MacBook without GPU support. You just need to make sure you have enough RAM to run a forward pass of ResNet 50.  

**2. Pillow Version Issue**

If you get:

	cannot import name ‘PILLOW_VERSION’
	
This is a known weird issue between Pillow and Torchvision, install an older version as such:

	pip install pillow=6.2.1
	
**3. PyTorch GradCAM Path Issue**

The library does not seem to set the python path for you. You may have to set it manually. For example in Bash,
we can set it as such:

	export PYTHONPATH=$PYTHONPATH:/path/to/my/python/lib/python3.7/site-packages/
	
If you want to know where that is, try:

	which python
	
you will see:

	/path/to/my/python/bin/python


