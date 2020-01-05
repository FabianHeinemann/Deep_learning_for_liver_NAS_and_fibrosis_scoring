""" 
Copyright (c) 2019, Fabian Heinemann, Gerald Birk, Birgit Stierstorfer; Boehringer Ingelheim Pharma GmbH & Co KG
All rights reserved.
 
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials # provided with the distribution.
 
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import sys
import argparse
import yaml
from cnn_utils import cnn_utils
	
def main(args):
	# Open the configuration YAML file
	# given as command line argument
	with open(args.config, "r") as file:		
		# Get arguments from YAML		
		config_yaml = yaml.load(file)										
		model_path = config_yaml["model"]["model_path"]		
		model_file_name = config_yaml["model"]["model_file_name"]				
		ground_truth_path = config_yaml["tiles"]["ground_truth_path"]
		max_epochs = int(float(config_yaml["settings"]["max_epochs"]))
		do_val_split =  config_yaml["settings"]["do_val_split"]
		val_fraction =  config_yaml["settings"]["val_fraction"]
		generate_confusion_matrix = config_yaml["settings"]["generate_confusion_matrix"]						
		
		# Print arguments
		print("\n-----------------------------\n")		
		print("Model path: \t\t%s\n" % (model_path))		
		print("Model file name: \t%s\n" % (model_file_name))
		print("Ground truth path: \t%s\n" % (ground_truth_path))	
		print("Max epochs:\t\t%d\n" % (max_epochs))
		print("do_val_split: \t\t%s\n" % (do_val_split))
		print("val_fraction:  \t\t%s\n" % (val_fraction))
		print("Gen conf_matrix: \t%s\n" % (generate_confusion_matrix))				
		print("-----------------------------")
		
		# Create a cnn_utils object (handles tiled data and the CNN)					
		cnn_utils_obj = cnn_utils(model_path = model_path, model_file_name = model_file_name, tile_path = ground_truth_path, results_path = model_path)
					
		# Split off validation data (if desired by do_val_split keyword in config yaml)
		if (do_val_split):			
			ret = cnn_utils_obj.split_validation_data(val_fraction = val_fraction)
			if (ret == False):
				# There is data in val which needs to be manually moved to train
				sys.exit()
		# Prepare image data generators
		cnn_utils_obj.prepare_image_data_generators()

		# Set class weights
		cnn_utils_obj.set_class_weights()					

		# Initialize the CNN
		print("\n")
		print("Initializing CNN")
		cnn_utils_obj.initialize_model(load_pretrained_model = False)
		
		# Train
		print("Training CNN")
		cnn_utils_obj.train_model(n_epochs = max_epochs)
		cnn_utils_obj.save_learning_curves()
			
		# Generate confusion matrix
		if (generate_confusion_matrix is True):
			# Save non-normalized confusion matrix
			cnn_utils_obj.generate_and_save_confusion_matrix(normalize = False)
			
			# Save normalized confusion matrix
			cnn_utils_obj.generate_and_save_confusion_matrix(normalize = True)			
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config_file', action="store", dest="config", help="Filename of config file (*.yaml)", required = True)	
	args = parser.parse_args()
	main(args)
