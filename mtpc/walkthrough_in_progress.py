import pandas as pd
from data_split import data_split, augment_data
from train_predict_utils import train_neural_ode, predict_using_trained_model
from evaluation_utils import merge_predictions
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

# general hyperparameters
BASE_RANDOM_SEED = 1329
TORCH_RANDOM_SEED = 1000  # they have different random seeds for splitting and for the neural network
SPLIT_FRAC = 0.2
OUTER_FOLDS = [1, 2, 3, 4, 5]  # indices of train/test splits
MODEL_REPLICATES = [1, 2, 3, 4, 5]  # indices of model replicates for the ensemble of neural ODEs

# hyperparemeters for the model, selected by grid search
# note: the paper was not clear WHICH hyperparameters were selected by grid search
LR = 0.00005  # this is the most important hyperparameter to tune
L2 = 0.1  # weight decay is a form of regularization. should be tuned
# ODE solver tolerance. From the Neural ODE paper:
"""
ODE solvers can approximately ensure that the output is within a given tolerance of the true solution. 
The time spent by the forward call is proportional to the number of function evaluations, 
so tuning the tolerance gives us a trade-off between accuracy and computational cost. 
Our framework allows the user to trade off speed for precision, 
but requires the user to choose an error tolerance on both the forward and reverse passes during training. 
For sequence modeling, the default value of 1.5e-8 was used. In the classification and density estimation experiments, 
we were able to reduce the tolerance to 1e-3 and 1e-5, respectively, without degrading performance.
In short, tol is the tolerance for accepting/rejecting an adaptive step.
"""
TOL = 1e-4
# number of epochs to train the model. the authors use early stopping so it's not crucial.
# just needs to be large enough so the val loss eventually increases
EPOCHS = 30
# all of these together decide the size of the neural network
HIDDEN_DIM = 128
LATENT_DIM = 6
HIDDEN_DIM = 128
ODE_HIDDEN_DIM = 16

"""
Example data has the following columns:
  - STUD - Study ID. Can be 1000, 2000, 3000.
  - PTNM - Patient number. Can be repeated between studies, but for example patient 1 in Study 1000 is not the same person as patient 1 in study 2000.
  - DSFQ - Dosage frequency is how often the dose is administred. Only 1 or 3.
  - AMT - Dosage amount. Can be 0 when measurements taken between doses.
  - TIME - Time since beginning of patient's treatment.
  - TFDS - Time since dose.
  - DV - Concentration measurement. 
"""
data_complete = pd.read_csv("ExampleData/sim_data.csv", na_values=".")

select_cols = ["STUD", "DSFQ", "PTNM", "CYCL", "AMT", "TIME", "TFDS", "DV"]
# According to authors: Patient data that have been marked with non-missing values in the "C" columns have been removed from the analysis
if "C" in data_complete.columns.values:
    data_complete = data_complete[data_complete.C.isnull()]
data_complete = data_complete[data_complete.CYCL < 100]  # cut off all dosing cycles greater than 100
data_complete = data_complete[select_cols]  # filter down to columns of interest
data_complete = data_complete.rename(
    columns={"DV": "PK_timeCourse"}
)  # DV is our variable of interest - anolyte concentration
data_complete["PTNM"] = data_complete["PTNM"].astype("int").map("{:05d}".format)
data_complete["ID"] = (
    data_complete["STUD"].astype("int").astype("str") + data_complete["PTNM"]
)  # concatenate study ID and patient ID for overall, unique ID

time_summary = (
    data_complete[["ID", "TIME"]].groupby("ID").max().reset_index()
)  # get max time since start of treatment per ID
# only keep patients who have measurements past initial measurements (TIME == 0)
selected_ptnms = time_summary[time_summary.TIME > 0].ID
data_complete = data_complete[data_complete.ID.isin(selected_ptnms)]

data_complete["AMT"] = data_complete["AMT"].fillna(0)  # replace missing values for dosage with 0s

# Set up round 1 measurement features.
# Round 1 measurements for each ID are always used as input features for the neural network to predict measurements after round 1.
# For weekly dosage IDs, round 1 is anything before end of week 1 (TIME <= 168), for every 3 week dosage IDs, anything before end of week 3 (TIME <= 604)
data_complete["PK_round1"] = data_complete["PK_timeCourse"]
data_complete.loc[(data_complete.DSFQ == 1) & (data_complete.TIME >= 168), "PK_round1"] = 0
data_complete.loc[(data_complete.DSFQ == 3) & (data_complete.TIME >= 504), "PK_round1"] = 0

# Missing PK measurement value handling
data_complete["PK_round1"] = data_complete["PK_round1"].fillna(0)  # round 1 missing values filled with 0
data_complete["PK_timeCourse"] = data_complete["PK_timeCourse"].fillna(
    -1
)  # all others filled with -1, used to find missing values during training

data_complete = data_complete[
    ~((data_complete.AMT == 0) & (data_complete.TIME == 0))
]  # drop all first patient rows with no dosage

# Some rows are duplicate pairs for PTNM and TIME combinations with different cycle (CYCL) values
# Set the first dosage amount of duplicated rows to the last dosage amount and keep only last row of duplicated rows
# This implementation may be an issue if patient number (PTNM) repeats across multiple studies (STUD)
data_complete.loc[
    data_complete[["PTNM", "TIME"]].duplicated(keep="last"), "AMT"  # all non-last duplicated rows
] = data_complete.loc[
    data_complete[["PTNM", "TIME"]].duplicated(keep="first"), "AMT"  # all non-first duplicated rows
].values
data = data_complete[~data_complete[["PTNM", "TIME"]].duplicated(keep="first")]


"""
The authors of this code are doing two things when it comes to splitting the data:

(1) They are doing 5 train/test splits. They repeat the entire training and test procedure 5 times.
(2) Within each of the 5 splits above, the are also doing model averaging by training 5 Neural ODE models
that differ in: (a) initial conditions, (b) random seeds and (c) which subset of the training data is used
for actual model training vs validation. These 5 models are then averaged together to get the final model
which is then applied to the test set.
"""
eval_results_all = {}
for fold in OUTER_FOLDS:
    for model in MODEL_REPLICATES:

        # first we split up the data into training/validation/test
        train, test = data_split(data, "PTNM", seed=BASE_RANDOM_SEED + fold, test_size=SPLIT_FRAC)
        train, validate = data_split(train, "PTNM", seed=BASE_RANDOM_SEED + fold + model, test_size=SPLIT_FRAC)

        """
        Adding the first cycle of treatment of the test set to the training set, as it will later be used
        during test to predict the rest of the test set and not for evaluation. As such, the authors
        think it is OK to add to training data and to maximize the total amount of training data.
        
        TODO: confirm that the test_add_to_train was NOT actually used in the final evaluation metrics. 
        
        Reasoning from the paper:
        "Additionally, the first cycle of the observation, PK_cycle1 is also available as predictive features for the models. 
        Using the information above, we sought to predict the PK dynamics after the first cycle, i.e., 
        after 168 hr for the Q1W data and after 504 hr for the Q3W data."
        """
        test_add_to_train = pd.concat(
            [test[(test.DSFQ == 1) & (test.TIME < 168)], test[(test.DSFQ == 3) & (test.TIME < 504)]], ignore_index=True
        )
        train = pd.concat([train, test_add_to_train], ignore_index=True)
        # i am not sure it makes sense to add this to the validation data?
        validate = pd.concat([validate, test_add_to_train], ignore_index=True)

        """
        They add extra data to the training set made out of existing training data. 
        Here is a description from the paper:

        "We applied augmentation to prevent overfitting.
        We applied timewise truncation to increase the number of training examples.
        For each training example, in addition to the original example, we also truncated
        the examples at 1008 hr, 1512 hr, and 2016 hr and generated and added
        a set of new examples to the training examples."
        """
        train = augment_data(train)

        # create and train the model
        # the best checkpoint will be saved
        train_neural_ode(
            TORCH_RANDOM_SEED + model + fold,
            train,
            validate,
            model,
            fold,
            LR,
            TOL,
            EPOCHS,
            L2,
            HIDDEN_DIM,
            LATENT_DIM,
            ODE_HIDDEN_DIM,
        )

        # predict on test using the best model saved
        # during train_neural_ode
        eval_results = predict_using_trained_model(
            test,
            model,
            fold,
            TOL,
            HIDDEN_DIM,
            LATENT_DIM,
            ODE_HIDDEN_DIM,
        )

        eval_results_all[(fold, model)] = eval_results


"""
Now we can compute evaluation metrics and summarize them
"""
r2_scores = []
rmses = []
pearsonrs = []
for fold in OUTER_FOLDS:
    # perform the ensembling
    evals_per_fold = [eval_results_all[(fold, m)] for m in MODEL_REPLICATES]
    predictions = merge_predictions(evals_per_fold, data)
    # evaluate various metrics
    y_true = predictions["labels"].values
    y_pred = predictions["pred_agg"].values
    rmses.append(mean_squared_error(y_true, y_pred, squared=False))
    r2_scores.append(r2_score(y_true, y_pred))
    pearsonrs.append(pearsonr(y_true, y_pred)[0])


df = pd.DataFrame({"R2": r2_scores, "RMSE": rmses, "Pearson R": pearsonrs})
df.index = OUTER_FOLDS
print(df)

summary_df = df.agg(["min", "max", "mean", "median"])
print(summary_df)

# TODO(anyone):
# add a lot of documentation everywhere
# convert to jupyter notebook
# have a discussion of the 3 types of metrics as per Kei's request
