# README #

### Vehicle Loan Default Prediction Task on Analytics Vidhya

Financial institutions incur significant losses due to the default of vehicle loans. The need for a better credit risk scoring model is also raised by these institutions. This warrants a study to estimate the determinants of vehicle loan default. The task here is to accurately predict the probability of loanee/borrower defaulting on a vehicle loan in the first EMI (Equated Monthly Instalments) on the due date. Following Information regarding the loan and loanee are provided in the datasets:  
* Loanee Information (Demographic data like age, Identity proof etc.). 
* Loan Information (Disbursal details, loan to value ratio etc.). 
* Bureau data & history (Bureau score, number of active accounts, the status of other loans, credit history etc.).   
Doing so will ensure that clients capable of repayment are not rejected and important determinants can be identified which can be
further used for minimising the default rates.  

### APPROACH

Followed the approaches mentioned below to score a **Private LB Score of 0.6716 AUC and Ranked 5th Overall**

* First tested out a Random Forest Model with no feature engineering and only label encoding for categorical variables to get a baseline score. The model was a 10 fold cross-validated and averaged model. This scored a CV of around 0.655 and Public LB of around 0.643. 
* EDA to see the distribution of the different variables and also their distribution in train and test to decide on a cross validation strategy. While the disbursal date column had values in the future in the test as compared to train, there did not seem to be large differences in the distribution of data between the train and test for most of the variables. Hence, I decided to go for a regular Stratified K-Fold CV Strategy. My CV seemed very high for most of my models but an increase in CV was also leading to an increase in the Public LB, so, I decided to go forward with this method. 
* Trained a Light GBM Model on the base features to test out its performance. Tuned the model manually and also tested out bayesian hyper parameter tuning. Bayesian method did not give improvements upon the manual method. CV around 0.675 and a Public LB of around 0.6595.
* For the feature engineering part, I used an iterative approach and validated CV score for each approach/iteration to decide on the which new features that I generated to use i.e. added a couple of new features and checked if they increased the CV to decide whether to keep them or not. Total feature count at the end was ~170 features.
	* Some features were generated outside the CV and some features inside the CV. Grouped and averaged features were generated inside the CV in order to avoid any leakage of test or validation data into the train data.
	* Some non-grouped features created were:  Age at the time of disbursal, Credit History length converted to days, Primary + Secondary amounts summed for their corresponding variables i.e. Primary Active Accounts + Secondary Active Accounts to get Total Active Accounts and so on, Disbursed Amount/Asset Cost, Asset Cost - Disbursed Amount etc.
	* Some averaged features created were: avg ltv, avg disbursed amount, avg primary current balance etc averaged across state_id, current_pincode, branch_id, supplier_id etc.
	* Some summed features created were: sum of primary overdue accounts, sum of primary active accounts, sum of number of enquiries etc summed across state_id, pincode, branch_id, supplier_id etc.
* Used Target Encoding on some features like manufacturer_id, state_id, DOB_year, Emplyoyment Type, CNS Score Description, Age Bins etc which also improved the CV by ~0.0005. 
* My Final Models consists of 3 CatBoost Models and an 4 LightGBM Models blended together using geometric mean and weighted blending. CatBoost with GPU and indices of categorical features turned out to be very fast and my single model with both the highest CV and Public LB. All my models were manually tuned except for one LightGBM Model which was tuned using bayesian tuning. 
* Other Things I tried: Experimented with Time Based CV strategy, Neural Networks, Polynomial Interaction of features + PCA, Oversampling etc but these did not show improvements in my CV score. 

**Note**: The file `Main.ipynb` contains all the codes that needs to be run in the correct order to obtain the final model
