# Efficient Saliency and FastCAM 

FastCAM creates a saliency map using SMOE Scale saliency maps as described in our paper on 
[ArXiv:1911.11293](https://arxiv.org/abs/1911.11293). We obtain a highly significant speed-up by replacing
the Guided Backprop component typically used alongside GradCAM with our SMOE Scale saliency map.
Additionally, the expected accuracy of the saliency map is increased slightly. Thus, **FastCAM is three orders of magnitude faster and a little bit more accurate than GradCAM+Guided Backprop with SmoothGrad.** 

![Example output images](https://raw.githubusercontent.com/LLNL/fastcam/master/mdimg/fast-cam.ILSVRC2012_val_00049934.jpg)
 
## Performance
FastCAM is not only fast, but it is more accurate than most methods. The following is a list or ROAR/KAR scores for different methods along with notes about performance:



| Method | KAR |	 ROAR |	SUM | Speed | Resolution |
| --- | --- | --- | --- | --- |
| Integrated Grad *Sundararajan et al.*  	|	3.62|	-3.58|	0.03 | Slow | Fine|
| Gradient *Simonyan et al.* 			|	3.57|	-3.54|	0.04 | Medium | Fine|
| Guided Backprop *Springenberg et al.*  	|	3.60|	-3.57|	0.04 | Medium | Fine|
| GradCAM++ *Chattopadhyay et al.* 		| 	3.64|	-2.27|	1.37 | Fast | Coarse|
| Trunc Normal Ent + Layer Weights [1,1,1,1,1]|	3.61|	-2.47|	1.14| Fast| Fine|
| SMOE Scale + Layer Weights [1,1,1,1,1]		|	3.62|	-2.46|	1.15| Fast| Fine|
| SMOE Scale + Layer Weights [1,2,3,4,5]		|	3.62|	-2.34|	1.28| Fast| Fine|
| Normal Std + Prior Layer Weights			|	3.61|	-2.32|	1.29| Fast| Fine|
| Trunc Normal Ent + Prior Layer Weights		|	3.61|	-2.31|	1.30| Fast| Fine|
| SMOE Scale + Prior Layer Weights			|	3.61|	-2.31|	1.30| Fast| Fine|
| Integrated Grad *-w-* SmoothGrad Sq. 	|	3.56|	-2.68|	0.88| Slow| Fine|
| Guided Backprop *-w-* SmoothGrad Sq. 	|	3.49|	-2.33|	1.16| Slow| Fine|
| Gradient *-w-* SmoothGrad Sq. 			|	3.52|	-2.12|	1.41| Slow| Fine|
| SMOE Scale + Prior Wts. *-w-* GradCAM++		| 	3.66|	-2.22| 	1.44| Fast| Fine|

 
                         
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
	jupyterlab
	notebook
	torch
	torchvision
	opencv-python
	pytorch_gradcam

### Install and Run the Demo!

![Get Ready for EXCITEMENT](https://steemitimages.com/p/DVAkPJXe6RxaMiozqQxRKBpPCPSqM5k9eEaBqfuGYnq1rZoVgJfgBwH61WPbdCwxa7N5TvBS59Jxtv?format=match&mode=fit&width=640)

This will run our [Jupyter Notebook](https://github.com/LLNL/fastcam/blob/master/demo_fast-cam.ipynb) on your local computer.

These are our recommended installation steps:

	git clone git@github.com:LLNL/fastcam.git 
	cd fastcam 
	python3 -m venv venv3 
	source venv3/bin/activate
	pip install -r requirements.txt

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
	
## Contact

Questions, concerns and friendly banter can be addressed to: 

T. Nathan Mundhenk [mundhenk1@llnl.gov](mundhenk1@llnl.gov)

## License

FastCAM is distributed under the [BSD 3-Clause License](https://github.com/LLNL/fastcam/blob/master/LICENSE).

LLNL-CODE-802426



