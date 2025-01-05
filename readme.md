# To reproduce results


1. **make sure you have python installed beforehand**

1. Open terminal and ```git clone``` the repository

1.  Download dataset from Kaggle [Here!](https://www.kaggle.com/datasets/validmodel/grocery-store-dataset)

1. Unzip and place in "in"

1. use the terminal ```cd``` into grocerynet folder

1. write ```pip install -r requirements.txt```

1. run ```python src/dataset_merger.py``` to merge the dataset categories into a new compiled folder , in the ```/in``` folder

1. then, run the resnet.py script to train a model. The script takes arguments in the command line. Example of running the script:

```python src/resnet.py -e 30```

this will fine-tune a resnet model on the data over 30 epochs, and save the model in the ```/out``` folder

# visualize the training emissions

1. go into the generated model folder inside the ```/out``` folder, and go into the ```/emissions``` folder. e.g. ```/out/30 epochs/emissions```, and take ```emissions_base``` file. For example ```emissions_base_cdf802aa-d24b-4803-b5b8-ba0a567d098b.csv```

1. generate a new folder inside the ```/in``` folder, called ```emissions```, and put the ```.csv``` file in there.

1. Do this for as many emission files as you desire!! (rename the files for a better visualization, for example rename the csv to ```30 epochs emissions.csv```)

1. run the visualization script with ```python src/emission_vis.py```

1. enjoy

# running inference on a model and collect emission data

1. make sure you have a trained model

1. go into the script and change the epoch count to whatever the epoch count is of the model you want to run. for exmaple, it is curerntly set to 30, so it will look for a model in the ```/30 epohs``` folder.

1. put a test image in the ```/out``` folder, and change the test_image variable in the script to the name of your test image.

1. ```python src/inference.py```

1. emission results will be saved in a separate folder inside teh existing emission folder for that model.

# running the user study interface

1. run ```python src/find_test_images.py```, which will generate a new folder with images in the out/folder

1. use ```streamlit run src/st_interface.py```

1. complete the user study by filling in your name and rating all the image predictions. Results will be saved in ```/out``` folder.

1. run ```python src/user_vis.py``` to generate a visualization of the user results.

