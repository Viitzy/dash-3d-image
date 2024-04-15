# Partitioning 3D images with `Dash` and `scikit-image`

# requirements Linux
sudo apt install python3-pip
sudo apt-get install python3-tk
sudo apt-get install python3-pil python3-pil.imagetk

    
1. Package to build our iDISF python module: 
	pip3 install scikit-build cmake
2. Install Tkinter libraries for python3 to run the interface: 
	sudo apt-get install python3-tk
	python3 -m pip install git+https://github.com/RedFantom/ttkthemes
3. Install other common python libraries: 
	pip3 install -r requirements.txt
4. para rodar iDISF Original
    cd IDISF/python3/; python3 -m pip install . ; cd .. ; cd ..


# Compiling and cleaning
In IDISF/
To compile all files: make
For removing all generated files from source: make clean


## Running

A recommended command line is

```bash
LOAD_SUPERPIXEL=assets/BraTS19_2013_10_1_flair_superpixels.npz.gz \
PYTHONPATH=plotly-common \
python app2.py
```

Then you can navigate to the displayed link in your browser. You can also run
without specifying `LOAD_SUPERPIXEL`, in which case the segmentation will happen
when the app loads.

To generate a new superpixel file, you can specify a path to the environment
variable `SAVE_SUPERPIXEL` in which case the app will run, compute the
superpixels, save them, and exit.

