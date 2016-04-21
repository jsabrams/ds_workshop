{\rtf1\ansi\ansicpg1252\cocoartf1404\cocoasubrtf460
{\fonttbl\f0\fnil\fcharset0 HelveticaNeue;\f1\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red36\green43\blue141;\red255\green255\blue255;
\red41\green101\blue168;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\sl280\partightenfactor0

\f0\fs28 \cf2 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Predicting default risk in the Lending Club loan data\
The Lending Club provides peer-to-peer loans, allowing individual lenders to invest in a portfolio of loans. The Lending Club provides it's own risk assessment and funds a loan at an internally selected interest rate. Lenders can then purchase part of the loan for their portfolio. To inform lender decisions, the Lending Club provides information about the borrower and makes historic loan data available. Here, I develop a series of models to predict default risk in the Lending Club's historic loan data.\
\pard\pardeftab720\sl340\qr\partightenfactor0

\f1 \cf3 \cb4 \strokec3 \
\pard\pardeftab720\sl400\partightenfactor0

\f0 \cf2 \cb1 \strokec2 \
\pard\pardeftab720\sl400\partightenfactor0
\cf2 0: Raw data files\
LoanStats3a.csv.zip, LoanStats3b.csv.zip, LoanStats3c.csv.zip, and LoanStats3d.csv.zip\
These are the raw csv data tables provided by the Lending Club. These files contain all funded loans from 2007 until the end of 2015.\
\pard\pardeftab720\sl340\qr\partightenfactor0

\f1 \cf3 \strokec3 \
\pard\pardeftab720\sl400\partightenfactor0

\f0 \cf2 \strokec2 1: Data cleaning and processing\
data_cleaning_processing.ipynb, utils.py, anon_avg_irs_data.pkl\
Data fields were cleaned/formatted and missing data were imputed as outlined in the data_cleaning_processing notebook. Charged off and defaulted loans are both considered to be "Default" for this analysis, whereas paid loans are considered "Paid." Loans in progress are not considered.\
Additionally, IRS tax data for listed zip codes were joined to the data table, as the data for individual zip codes was sparse. Zip code data were processed using the zip_code_processing notebook and the following files:\
09zpallnoagi.csv, 10zpallnoagi.csv, 11zpallnoagi.csv, 12zpallnoagi.csv, 13zpallnoagi.csv\
Additional zip code data are available at {\field{\*\fldinst{HYPERLINK "https://www.irs.gov/uac/SOI-Tax-Stats-Individual-Income-Tax-Statistics-ZIP-Code-Data-(SOI"}}{\fldrslt \cf5 \ul \ulc5 \strokec5 https://www.irs.gov/uac/SOI-Tax-Stats-Individual-Income-Tax-Statistics-ZIP-Code-Data-(SOI}})\
After this process, features were all normalized.\
Data were then randomly sampled and divided into training, validation, and test sets.\
\pard\pardeftab720\sl340\qr\partightenfactor0

\f1 \cf3 \strokec3 \
\pard\pardeftab720\sl400\partightenfactor0

\f0 \cf2 \strokec2 2: Training, validation, and test data\
data_files.zip\
These data are compressed in data_files as pandas dataframes. Models were trained on the training set with cross-validation and selected based on performance on the validation set. Final performance was evaluated on the test set.\
\pard\pardeftab720\sl340\qr\partightenfactor0

\f1 \cf3 \strokec3 \
\pard\pardeftab720\sl400\partightenfactor0

\f0 \cf2 \strokec2 3: Feature analysis\
lending_club_feature_analysis.ipynb, model_fitting.py\
On the training data, I evaluated the statistical relationship between the various features and the labels. I report effect sizes and tests of statistical significant with correction for multiple comparisons. I also report the correlations between the features. This facilitated my selection of a subset of features to be used for modeling.\
\pard\pardeftab720\sl340\qr\partightenfactor0

\f1 \cf3 \strokec3 \
\pard\pardeftab720\sl400\partightenfactor0

\f0 \cf2 \strokec2 4: Model selection\
lending_club_modeling.ipynb, model_fitting.py, loan_roi.py\
The following procedure was applied to several models: First, models were fit to the training data, one feature at a time, in order of effect size for the features. Next, performance evaluated on the validation set as the area under the ROC curve (AUC). Then, a number of features were selected and performance was evaluated as ROI for a hypothetical portfolio of loans with the model's estimates of default probability used as a rejection criterion. This performance was compared a model that rejected a random set of loans.\
Models were selected based on AUC from cross-validation and predicted ROI based on the validation set.\
\pard\pardeftab720\sl340\qr\partightenfactor0

\f1 \cf3 \strokec3 \
\pard\pardeftab720\sl400\partightenfactor0

\f0 \cf2 \strokec2 5: Model performance\
model_performance.ipynb, model_fitting.py, loan_roi.py\
A decision tree, an AdaBoost ensemble of trees, and a stacked ensemble learner were tested on the test set. Performance was measured as AUC on the test set and ROI on the test set. The three models were selected to represent the span between simpler and more complex models.\
Additionally, I presented the performance of the models when the loan amount and interest rate were included as features. This is to illustrate how high performance can get. However, I do not recommend using these features if the goal is to generate a risk model to set interest rates and loan amounts. The reason is that these features are based on the Lending Club's in-house risk-assessment tools, so these features limit our ability to objectively assess the risk of default outside of the Lending Club context.\
\pard\pardeftab720\sl340\qr\partightenfactor0

\f1 \cf3 \strokec3 \
\pard\pardeftab720\sl400\partightenfactor0

\f0 \cf2 \strokec2 6: Final prediction code\
prediction_workflow.ipynb, loan_prediction.py, train_ensemble.py, test_loans.pkl\
Two models are considered for prediction: a decision tree and a stacked ensemble learner. Unnormalized loan data can be found in test_loans (a pandas dataframe). A single loan or batch of loans can be passed to either predict_default_tree or predict_default_ensemble, and those functions will output a probability of default. Note that the classifier files are saved and uploaded for the decision tree. The classifier files for the stacked ensemble learner are too big to be uploaded to GitHub. As such, I have included train_ensemble.py. When run, that function will generate all of the files for the stacked ensemble learner.}