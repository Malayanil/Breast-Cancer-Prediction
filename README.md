# Breast Cancer Prediction

Welcome ! This is a generalised Read Me File for the Breast Cancer Prediction project achieved by implementation of Machine Learning in Python. 


Index :

	1. General Details and FAQs
	2. Explanation of the Code
	3. Graphs plotted in the Program (Images in 'dependency_png' folder and 'k9.png')


1. General Details and FAQs: 

	1.1 What is a Biopsy ?
	
	A biopsy is a medical procedure, during which a small sample of tissue is removed from a part of the body. The sample of tissue is then examined under the microscope to look for abnormal cells. Sometimes the sample is tested in other ways. For example, it may be tested with chemical reagents to help identify abnormal chemicals in the tissue sample. Sometimes tests are done on the sample to look for bacteria or other germs.


	1.2 What is Cancer ? Define Breast Cancer.
	
	Cancer is the uncontrolled growth of abnormal cells anywhere in a body. These abnormal cells are termed cancer cells, malignant cells, or tumor cells. These cells can infiltrate normal body tissues. Many cancers and the abnormal cells that compose the cancer tissue are further identified by the name of the tissue that the abnormal cells originated from (for example, breast cancer, lung cancer, colon cancer). Cancer is not confined to humans; animals and other living organisms can get cancer.
	Breast cancer starts in the cells of the breast as a group of cancer cells that can then invade surrounding tissues or spread (metastasize) to other areas of the body.Cancer begins in the cells which are the basic building blocks that make up tissue. Tissue is found in the breast and other parts of the body.  Sometimes, the process of cell growth goes wrong and new cells form when the body doesn’t need them and old or damaged cells do not die as they should.  When this occurs, a build up of cells often forms a mass of tissue called a lump, growth, or tumor.\
	Breast cancer occurs when malignant tumors develop in the breast.  These cells can spread by breaking away from the original tumor and entering blood vessels or lymph vessels, which branch into tissues throughout the body.\
	Breast cancer has been observed to have 246,000 new estimated cases, 18% of all cancer estimated cases, a year(2015-16) followed by 40,000 deaths.


	1.3 How is Biopsy related to detection of Cancer ?
	
	Biopsies are often important to diagnose cancer. A biopsy is commonly done if you have a lump or swelling of a part of the body where there is no apparent cause. In these situations often the only way to confirm whether the lump is a cancer is to take a biopsy and look at the cells under the microscope. Cancer cells look different to normal cells.

	
	1.4 What is the origin of the dataset ?
	
	This breast cancer database was obtained from the University of Wisconsin Hospitals, Madison from Dr. William H. Wolberg. He assessed biopsies of breast tumours for 699 patients up to 15 July 1992; each of nine attributes has been scored on a scale of 1 to 10, and the outcome is also known. There are 699 rows and 11 columns.\
   16 values are missing from the Bare Nuclei column. In order to avoid any hassle, these 16 records were deleted; else introduction of unruly data may affect the predictibility of the model. The 'Class' feature was converted to numericals for ease of computation. Class Benign is assumed as 0 and Malignant as 1.\
P. M. Murphy and D. W. Aha (1992). *biopsy.csv* , UCI Repository of machine learning databases. [Machine-readable data repository]. Irvine, CA: University of California, Department of Information and Computer Science.


	1.5 What are the terms used in the CSV File ?
	
   	Benign Cells are non-cancerous. Malignant Cells are cancerous. They are described as Class and are the values to be predicted.\
  	Clump Thickness: Benign cells tend to be grouped in monolayers, while cancerous cells are often grouped in multilayers. Hence thickness of the lump or tumour may throw some light on the type of cells.\
  	Uniformity of Cell Size/Shape: Cancer cells tend to vary in size and shape oddly compared to normal cells. That's why these parameters are valuable in determining whether the cells are cancerous or not.\
  	Marginal Adhesion: Normal cells tend to stick together. Cancer cells tends to loose this ability. So loss of adhesion is a sign of malignancy.\
  	Single Epithelial Cell Size: It is related to the uniformity which is mentioned. Epithelial cells that are significantly enlarged may be a malignant cell.\
  	Bare Nuclei: This is a term used for nuclei that is not surrounded by cytoplasm (the rest of the cell). Those are typically seen in Benign tumours.\
  	Bland Chromatin: Describes a uniform "texture" of the nucleus seen in benign cells. In cancer cells the chromatin tend to be more coarse.\
  	Normal Nucleoli: Nucleoli are small structures seen in the nucleus. In normal cells the nucleolus is usually very small if visible at all. In cancer cells the nucleoli become more prominent, and sometimes there are more of them.

							--End of Section One--
							
2. Explanation of the Code

	2.1 At first, we import the libraries needed for our Script.\
		> NumPy is needed for Array usage in the program and also conversion of DataFrames.\
		> MatPlotLib is needed for plotting Graphs between various data to find out their dependencies.\
		> Pandas is needed for it's utility of using DataFrames for proper visualization and storage of datasets from CSV Files.\
		> SKLearn is needed for it's Machine Learning algorithms for data analysis.
		
		
		import numpy as np
		import matplotlib.pyplot as plt
		from matplotlib.ticker import NullFormatter
		import pandas as pd
		import matplotlib.ticker as ticker
		from sklearn import preprocessing
		from sklearn.model_selection import train_test_split
		from sklearn.neighbors import KNeighborsClassifier
		from sklearn import metrics
		from sklearn.model_selection import KFold
		from sklearn.cross_validation import cross_val_score, cross_val_predict

	2.2 Heatmaps show the Co-Relation between the features themselves and the colorbar beside each of the heatmaps show the numerical scale of the Heatmap. The dataset has been divided into 5 random non-repetitive portions and the Heatmap is plotted with 10 random non-repetitive data points; since all of the data from the dataset cannot be fit into a Heatmap as it will get cluttered. Then finally, combining all of the 5 Heatmaps, one final Heatmap is plotted for representing the full dataset.
	
		im =ax.imshow(X[n:(n+10)])
		
		ax.set_xticks(np.arange(len(features)))
		ax.set_yticks(np.arange(len(features)))
		ax.set_xticklabels(features)
		ax.set_yticklabels(features)
		plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
		
		ax.set_title('Heatmap of Feature\'s CoRelations')
		plt.gca().invert_xaxis()
		fig.tight_layout()
		plt.colorbar(im)
		plt.show()
		
		# 'n' is the random number generated to gather the data points from the dataset.
		
	2.3 Now, we load the dataset into our DataFrame and then split and convert them into two different Arrays required for Training and Testing the dataset.
	
		
		df = pd.read_csv('biopsy.csv')
		X = df[['ClumpThickness', 'UniformityOfCellSize', 'UniformityOfCellShape', 'MarginalAdhesion', 'EpithelialCellSize', 'BareNuclei', 'BlandChromatin', 'NormalNucleoli', 'Mitoses']]
		
		X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
		Y = df['Class'].values
		X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=42)
		# We have chosen the features as concluded from the individual dependencies of features compared to the 'Class' feature.
		# The graphs have been shown in Section 3 of this file.
		
	2.4 Since this is a Machine Learning problem to classify and predict data into two categories, we use the K-Nearest Neighbour Model for classification.\
		> We find out the optimal value of "K" by plotting an accuracy graph. We then use that value of "K" for final prediction of the data. 
		
		
		for i  in nCount:

   		 	# Train Model and Predict
   		 	neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,Y_train)
    
    			Yhat = neigh.predict(X_test)

    			# Accuracy Evaluation
    			training_accuracy.append(metrics.accuracy_score(Y_train, neigh.predict(X_train)))
     			print("Train set Accuracy: ",training_accuracy)
    			test_accuracy.append(metrics.accuracy_score(Y_test, Yhat))
				print("Test set Accuracy: ", test_accuracy)
    
		plt.plot(nCount,training_accuracy, label ='Training Accuracy')
		plt.plot(nCount, test_accuracy, label ='Testing Accuracy')
		plt.xlabel('Number of Neighbours')
		plt.ylabel('Accuracy')
		plt.title('Convergence of Train and Test Graph')
		plt.legend()
		plt.show()	
		# The graph has been shown in Section 3 of this file.
		
		
	2.5 Accuracy is also tested upon the Train Set and the Test Set.


		print('Train set Accuracy: ', metrics.accuracy_score(Y_train, neigh.predict(X_train)))
		print('Test set Accuracy: ', metrics.accuracy_score(Y_test, Yhat))
		
		OUTPUT: 
		('Train set Accuracy: ', 0.9706959706959707)
		('Test set Accuracy: ', 0.9708029197080292)
		
	
	2.6 We then implement K-Fold Cross-Validation to stike off any over-fitting or under-fitting of data in our model and make it a suitable and generalized one.\
		> We also display the Mean Cross-Validation score at the end.

		# K-Fold Implementation
		kf = KFold(3, True)
		kf.get_n_splits(X)
		print(kf)
		
		OUTPUT:
		KFold(n_splits=3, random_state=None, shuffle=True)
		
		for train_index, test_index in kf.split(X):
    			# print('Train: ', train_index, 'Test: ', test_index)
    			X_train, X_test = X[train_index], X[test_index]
    			Y_train, Y_test = Y[train_index], Y[test_index]

		# Cross Validation
		kNN = KNeighborsClassifier(n_neighbors = 9)
		score = cross_val_score(kNN, X, Y, cv=7)
		print('\n')
		print('Cross Validation Mean Score: ',score.mean())
		
		OUTPUT:
		('Cross Validation Mean Score: ', 0.969370905411301)
				
				
	2.7 This marks the end of code, the code is present in the "script.py" file.
	
							--End of Section Two--
							

3. Graphs plotted in the Program (Images in 'dependency_png' folder and 'k9.png')

		3.1 Finding out the optimal "K" of KNN Algorithm, i.e. the proper number of neighbours for the classification. Explained in Section 2.3

![alt text](https://github.com/Malayanil/Breast-Cancer-Prediction/blob/master/k9.png)

	3.2 Graphs 
		3.2.1 ClumpThickness Histogram
		3.2.2 UniformityOfCellSize Histogram
		3.2.3 UniformityOfCellShape Histogram
		3.2.4 MarginalAdhesion Histogram
		3.2.5 EpithelialCellSize Histogram
		3.2.6 BareNuclei Histogram
		3.2.7 BlandChromatin Histogram
		3.2.8 NormalNucleoli Histogram
		3.2.9 Mitoses Histogram
		3.2.10 Class [0: Benign Cell Count 1: Malignant Cell Count]
		
![alt text](https://github.com/Malayanil/Breast-Cancer-Prediction/blob/master/dependency_pngs/combined.png)
		

	
	3.3 Heatmaps
	
![alt text](https://github.com/Malayanil/Breast-Cancer-Prediction/blob/master/heatmaps/combined.png)
	
	 
	

						--End of Section Three--
