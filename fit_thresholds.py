""" 
Copyright (c) 2019, Fabian Heinemann, Gerald Birk, Birgit Stierstorfer; Boehringer Ingelheim Boehringer Ingelheim Pharma GmbH & Co KG
All rights reserved.
 
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials # provided with the distribution.
 
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
import json
import collections
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Rectangle
from cnn_utils import cnn_utils
from sklearn import metrics


def get_monte_carlo_search_thresholds(score_str, merged_summary, cnn_utils_obj):
	""" Monte carlo fit to new thresholds
	
	Arguments:
		score_str: (str) score to optimize
		merged_summary: (Pandas Dataframe) with merged table
		cnn_utils_obj: (CNN utils object)
		
	Returns:
		thresholds_best: (dictionary) optimized thresholds
	
	"""	
	# Thresholds
	thresholds = {}
	thresholds_best = {}
	
	compensate_error_for_class_imbalance = True	
	optimization_target = "squared_error" 

	# In case of steatosis, we optimize for the cv_based thresholds
	if (score_str == "steatosis_score"):
		score_str_tmp = "steatosis_score_cv_based"
		# Otherwise for the expert sub scores
	else:
		score_str_tmp = score_str

	# Sd for random numbers to try out thresholds.
	sigma = 0.15

	# Scores
	score_range = merged_summary[score_str_tmp].unique()

	# Start values for thresholds
	thresholds_best[0] = np.min(merged_summary["average_weighted_class"])-0.001
	for score in sorted(score_range)[1:]:
		# Take 90th percentile as threshold starting value
		thresholds_best[int(score)] = np.percentile(merged_summary[merged_summary[score_str_tmp] == int(score)]["average_weighted_class"],90)

	thresholds_max = max(merged_summary[score_str_tmp])

	# Start values for error metrics to optimize
	min_err = np.inf
	max_acc = 0
	max_weighted_acc = 0

	class_imbalance_factors = {}

	if (compensate_error_for_class_imbalance):	
		for score in score_range:
			class_imbalance_factors[int(score)] = merged_summary.shape[0] / merged_summary[merged_summary[score_str_tmp] == int(score)].shape[0]
		print("Compensating for class imbalance with factors: ", class_imbalance_factors)
	else:
		for score in score_range:
			class_imbalance_factors[int(score)] = 1

	print("Optimizing %s" % optimization_target)

	#
	# Monte Carlo search for best split
	#
	# Rule: randomly try thresholds and minimize quadratic error

	max_iterations = 2500

	for i in range(0,max_iterations):
		current_err = 0	
		current_weighted_acc = 0
		current_acc = 0

		# Generate new thresholds
		thresholds_unsorted = {}
		for j in range(0,len(score_range)):		

			# First
			if (j == 0):
				thresholds_unsorted[j] = thresholds_best[0]
			# Others
			else:
				while (True):
					thresholds_unsorted[j] = thresholds_best[j] + sigma*np.random.randn()			
					if (thresholds_unsorted[j] > thresholds_best[0] and thresholds_unsorted[j] < thresholds_max):
						break

		# Sort them
		for j in range(0,len(score_range)):
			thresholds[j] = sorted(thresholds_unsorted.values())[j]

		# Compute error sum
		for index, row in merged_summary.iterrows():
			# Path score
			pathologist_score = row[score_str_tmp]						

			# DL score
			dl_score = cnn_utils_obj.get_dl_score(thresholds, row["average_weighted_class"])		   

			current_err = current_err + class_imbalance_factors[int(pathologist_score)]*(dl_score-pathologist_score)*(dl_score-pathologist_score)

			if (dl_score == pathologist_score):			
				current_weighted_acc = current_weighted_acc + class_imbalance_factors[int(pathologist_score)]
				current_acc = current_acc + 1

		current_weighted_acc = current_weighted_acc / merged_summary.shape[0]
		current_acc = current_acc / merged_summary.shape[0]

		# Update best threshold if new minimal error was found
		# In case of squared_error, an empircal rule was found beneficial
		# That simultaneously the accuracy only allowed to decrease by up to eps%
		# This prevents overfocus on minority classes due to weighting
		eps = 0.02
		if ((current_err < min_err and optimization_target == "squared_error") and (current_acc*(1+eps) > max_acc or max_acc == 0) 
			or 
			(current_weighted_acc > max_weighted_acc and optimization_target == "accuracy")):	

			# Update metics optimas
			min_err = current_err
			max_acc = current_acc
			max_weighted_acc = current_weighted_acc			

			# Copy them
			# https://stackoverflow.com/questions/2465921/how-to-copy-a-dictionary-and-only-edit-the-copy
			thresholds_best = dict(thresholds)

			print("Iteration: %d, Quad. error: %.1f, Weighted accuracy score: %.2f, Acc: %.2f" %(i, min_err, max_weighted_acc, max_acc))

			print("{")
			for key, value in thresholds_best.items():
				print("%d: %.3f\t" % (key, value))
			print("}")
	
	thresholds = thresholds_best	
	return thresholds_best
		
	print("Done")
	
def generate_and_save_plot(score_str, fig_path, fig_file_name_str, merged_summary, thresholds):
	""" Saves plot of the continous weighted class average vs the pathologist score as png
	
	Arguments:
		score_str: (str) score to optimize
		fig_path: (str) Full path of figure to save
		fig_file_name_str: (str) Name of figure to save
		merged_summary: (Pandas Dataframe) with merged table		
		thresholds: (dict) Dict of thresholds
		
	Returns:
		saves a figure to fig_path
	
	"""	

	np.random.seed(55)	
	show_thresholds = True

	if (score_str == "steatosis_score"):
		score_str_tmp = "steatosis_score_cv_based"
	else:
		score_str_tmp = score_str

	# Y label
	score_label_str = score_str.replace("_", " ").capitalize()
	if (score_str != "fibrosis_score"):
		score_label_str = score_label_str.replace("score", "sub-score / sample")
		
	if (score_str != "steatosis_score"):
		score_label_str = score_label_str + "\n(Human expert)"        
	else:
		score_label_str = score_label_str + "\n(CV-based)" 

	keys = ["average_weighted_class"]
	col_str = {"inflammation_score": "#ff1f20", "ballooning_score" : "#32cd32", "steatosis_score" : "b", "fibrosis_score" : "#111111"}

	# Set font size
	# https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
	matplotlib.rcParams.update({'font.size': 14})

	for key in keys:    
		fig = plt.figure()
		ax = plt.subplot(111)        
		
		min_x = -np.min(merged_summary[key])
			
		if (score_str == "ballooning_score"):
			ax.set_xlim([min(merged_summary["average_weighted_class"]), 1])
		elif (score_str == "inflammation_score"):
			ax.set_xlim([min(merged_summary["average_weighted_class"]), 2])
		elif (score_str == "steatosis_score"):
			ax.set_xlim([min(merged_summary["average_weighted_class"]), 3])        
		elif (score_str == "fibrosis_score"):            
			ax.set_xlim([min(merged_summary["average_weighted_class"]), 4])
		ax.set_xscale('log')
		jitter = 0.06
			
		ax.set_ylim([-0.2, max(thresholds.keys()) + 0.4])             
		
		ax.scatter(merged_summary["average_weighted_class"], merged_summary[score_str_tmp] + np.random.normal(0,jitter,merged_summary.shape[0]) , c=col_str[score_str], alpha=0.4, s=20)
			
		if (key == "average_weighted_class"):        
			for score in thresholds:
					
				if (show_thresholds):
					ax.add_patch(Rectangle((min_x, -0.2), -min_x + thresholds[score], max(merged_summary[score_str_tmp]) + 0.6, alpha = 0.1, facecolor = "#aaaaaa"))            

					# Add text with deep learning score in plot    
					dl_score_str = score
					if (score < max(thresholds.keys())):
						ax.text(thresholds[score] + (thresholds[score+1] - thresholds[score])/2, score + 0.3, dl_score_str)
					else:                    
						ax.text(thresholds[score]*1.2, score + 0.2, dl_score_str)                    

		# Set y ticks to integers
		ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
		ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))        
			
		if (score_str == "ballooning_score"):
			title_str = "Balloning"
			plt.xlabel("Average ballooning class (0-1) / sample")
		elif (score_str == "inflammation_score"):
			title_str = "Inflammation"
			plt.xlabel("Average inflammation class (0-2) / sample")
		elif (score_str == "steatosis_score"):
			title_str = "Steatosis"
			plt.xlabel("Average steatosis class (0-3) / sample")
		elif (score_str == "fibrosis_score"):
			title_str = "Fibrosis"
			plt.xlabel("Average fibrosis class (0-4) / sample")        
			
		plt.ylabel(score_label_str)        
		plt.title(title_str)
		
		fig.savefig(fig_path + fig_file_name_str, bbox_inches='tight', dpi=600)


def create_merged_summary_table(model_base_path, path_scores_file_name, summary_result):
	""" Merge table with pathologist scores (ground truth) and classification summary
	
	Arguments:
		model_base_path: (str) Filename of model root path
		path_scores_file_name: (str) File name of pathologist ground truth scores
		summary_result: (Pandas DataFrame) Results from classifier for one of the four models
		
	Returns:
		Pandas Dataframe with merged table
	
	"""	
	pathologist_scores = pd.read_csv(model_base_path + path_scores_file_name, sep = ";")
	pathologist_scores = pathologist_scores.dropna()

	# Set dtypes of pathologist_scores and summary
	pathologist_scores = pathologist_scores.astype({"experiment" : str, "group": str, "animal": str})

	merged_summary = summary_result
	merged_summary["ballooning_score"] = np.nan
	merged_summary["inflammation_score"] = np.nan
	merged_summary["steatosis_score"] = np.nan
	merged_summary["fibrosis_score"] = np.nan
	merged_summary["steatosis_cv_percentage"] = np.nan
	merged_summary["steatosis_score_cv_based"] = np.nan

	# Loop over all unique experiments
	for index, row in merged_summary.iterrows():
		experiment = str(row["experiment"])
		group = str(row["group"])
		animal = str(row["animal"])

		boolean_condition_exp = pathologist_scores["experiment"] == experiment
		boolean_condition_group = pathologist_scores["group"] == group
		boolean_condition_animal = pathologist_scores["animal"] == animal		
		
		boolean_condition = boolean_condition_exp & boolean_condition_group & boolean_condition_animal			 

		if (boolean_condition.any() == True):			
			index_path_scores = pathologist_scores[boolean_condition].index[0]				

			ballooning_score = pathologist_scores.at[index_path_scores, "ballooning_score"]			
			inflammation_score = pathologist_scores.at[index_path_scores, "inflammation_score"]					
			steatosis_score = pathologist_scores.at[index_path_scores, "steatosis_score"]
			fibrosis_score = pathologist_scores.at[index_path_scores, "fibrosis_score"]
			steatosis_cv_percentage  = pathologist_scores.at[index_path_scores, "steatosis_cv_percentage"] 

			# Steatosis score based on CV determined steatosis values
			if (steatosis_cv_percentage < 5):
				steatosis_score_cv_based = 0
			elif (steatosis_cv_percentage >= 5 and steatosis_cv_percentage <= 33):
				steatosis_score_cv_based = 1
			elif (steatosis_cv_percentage > 33 and steatosis_cv_percentage <= 66):			
				steatosis_score_cv_based = 2
			elif (steatosis_cv_percentage > 66):
				steatosis_score_cv_based = 3

			merged_summary.at[index, "ballooning_score"] = ballooning_score
			merged_summary.at[index, "inflammation_score"] = inflammation_score		
			merged_summary.at[index, "steatosis_score"] = steatosis_score		
			merged_summary.at[index, "fibrosis_score"] = fibrosis_score		
			merged_summary.at[index, "steatosis_cv_percentage"] = steatosis_cv_percentage		 
			merged_summary.at[index, "steatosis_score_cv_based"] = steatosis_score_cv_based

	merged_summary = merged_summary.dropna(axis = 0, subset = ["ballooning_score", "inflammation_score", "steatosis_score_cv_based", "fibrosis_score"])
	return merged_summary

def main(args):
	# Open the configuration YAML file
	# given as command line argument
	with open(args.config, "r") as file:
		############################################################################################################################################
		# Get arguments from YAML		
		############################################################################################################################################
		config_yaml = yaml.load(file)
		
		# Model base path
		model_base_path = config_yaml["settings"]["model_base_path"]				
		
		# Score type (test or train)
		test = config_yaml["settings"]["test"]	
		
		# Fit new thresholds or load them
		fit_new_thresholds = config_yaml["settings"]["fit_new_thresholds"]
		
		# File name of full results
		full_results_name = config_yaml["settings"]["full_results_name"]	
		
		# Type of score must be one of 'ballooning_score', 'inflammation_score', 'steatosis_score' or 'fibrosis_score'
		score_str = config_yaml["settings"]["score_str"]		
		
		# Fit new thresholds only for train
		if (test == False):					
			# Use ground truth for train
			path_scores_file_name = "pathologist_scores.csv"
			
			# Name tag to add to image
			str_name_flag = ""
		else:
			# For test no thresholds can be fitted
			fit_new_thresholds = False
			
			# Use ground truth for test			
			path_scores_file_name = "pathologist_scores_test.csv"
			
			# Name tag to add to image
			str_name_flag = "_test"

		if (score_str == "ballooning_score"):
			score_name = "Ballooning_sub_score"
			model_path = model_base_path + "NAS/ballooning/model/"
		elif (score_str == "inflammation_score"):		
			score_name = "Inflammation_sub_score"
			model_path = model_base_path + "NAS/inflammation/model/"
		elif (score_str == "steatosis_score"):
			score_name = "Steatosis_sub_score"
			model_path = model_base_path + "NAS/steatosis/model/"
		elif (score_str == "fibrosis_score"):
			score_name = "Fibrosis_score"
			model_path = model_base_path + "fibrosis/model/"
		else:
			print("Score string must be one of \'ballooning_score\', \'inflammation_score\', \'steatosis_score\' or \'fibrosis_score\'")
			sys.exit()									 	
		
		############################################################################################################################################
		# Prepare data
		############################################################################################################################################
		cnn_utils_obj = cnn_utils(model_path = model_path, model_file_name = "", tile_path = "", results_path = model_path, list_of_classes = [])
		
		# Load table with full results
		classification_result = pd.read_csv(model_path + full_results_name, sep = None, engine = "python")
		
		# Create summary result
		summary_result = cnn_utils_obj.generate_summary_results(classification_result, score_name)
		summary_result = summary_result.astype({"experiment" : str, "group": str, "animal": str})

		# Merge table with pathologist scores (ground truth) and classification summary
		merged_summary = create_merged_summary_table(model_base_path, path_scores_file_name, summary_result)
		
		############################################################################################################################################
		# Fit thresholds
		############################################################################################################################################		
		
		thresholds_file_name = model_path + score_str + "_thresholds" + ".json"
		if (fit_new_thresholds == False):	
			# Load existing thresholds from json	
			with open(thresholds_file_name, "r") as read_file:
				thresholds = json.load(read_file)
				
				# Convert type of dict
				thresholds = {int(k):float(v) for k,v in thresholds.items()}
				print("Thresholds loaded.")
		else:
			# Determine new set of thresholds
			thresholds = get_monte_carlo_search_thresholds(score_str, merged_summary, cnn_utils_obj)			
			
			# Save thresholds as json			
			with open(thresholds_file_name, "w") as write_file:
				json.dump(thresholds, write_file)
				print("Thresholds saved to %s" % (thresholds_file_name))
				
		############################################################################################################################################
		# Generate plot of continous score vs threshold
		############################################################################################################################################		
		fig_file_name_str = "average_weighted_class" + full_results_name[:-4] + str_name_flag + ".png"
		generate_and_save_plot(score_str, model_path, fig_file_name_str, merged_summary, thresholds)
		print("Figure saved to %s " % (score_str + model_path))
		
		############################################################################################################################################
		# Compute error metrics and print them
		############################################################################################################################################		
		if (score_str == "steatosis_score"):
			score_str_tmp = "steatosis_score_cv_based"
		else:
			score_str_tmp = score_str
			
		# Create summary result with thresholds
		summary_result = cnn_utils_obj.generate_summary_results(classification_result, score_name, score_str + "_thresholds" + ".json")
		summary_result = summary_result.astype({"experiment" : str, "group": str, "animal": str})

		# Merge table with pathologist scores (ground truth) and classification summary
		merged_summary = create_merged_summary_table(model_base_path, path_scores_file_name, summary_result)

		mae = metrics.mean_absolute_error(merged_summary[score_str_tmp], merged_summary[score_name])
		print("Mean absolute error = %.2f" % (mae))		
		
		prec = metrics.precision_score(merged_summary[score_str_tmp], merged_summary[score_name], average = "weighted")
		print("Weighted precision = %.2f" % (prec))
		
		rec = metrics.recall_score(merged_summary[score_str_tmp], merged_summary[score_name], average = "weighted")
		print("Weighted recall = %.2f" % (rec))
		
		F1 = metrics.f1_score(merged_summary[score_str_tmp], merged_summary[score_name], average = "weighted")
		print("Weighted F1 = %.2f" % (F1))
		
		acc = metrics.accuracy_score(merged_summary[score_str_tmp], merged_summary[score_name])
		print("Accuracy = %.2f %%" % (acc*100))
		
		cohen_kappa = metrics.cohen_kappa_score(merged_summary[score_str_tmp], merged_summary[score_name])
		print("Cohen Kappa: %.2f" % (cohen_kappa))	
		
		print("N = %d" % (merged_summary.shape[0]))
			
		
if __name__ == "__main__":	
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config_file', action="store", dest="config", help="Filename of config file (*.yaml)", required = True)	
	args = parser.parse_args()
	main(args)