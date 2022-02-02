import os
import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
from sklearn.metrics import mean_squared_error, r2_score

import utils
from model import Encoder, ODEFunc, Classifier
from data_parse import parse_tdm1
from datetime import datetime


log_path = "logs/" + f"{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.log"
utils.makedirs("logs/")
logger = utils.get_logger(logpath=log_path)


def load_model(ckpt_path, input_dim, hidden_dim, latent_dim, ode_hidden_dim, device="cpu"):
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")

    # choose whether to use a GPU if it is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create the parts of the model into which we load our checkpoint state
    encoder = Encoder(input_dim=input_dim, output_dim=2 * latent_dim, hidden_dim=hidden_dim)
    ode_func = ODEFunc(input_dim=latent_dim, hidden_dim=ode_hidden_dim)
    classifier = Classifier(latent_dim=latent_dim, output_dim=1)

    # load model checkpoint
    checkpt = torch.load(ckpt_path)

    # load checkpoint states into each part of model
    encoder_state = checkpt["encoder"]
    encoder.load_state_dict(encoder_state)
    encoder.to(device)

    ode_state = checkpt["ode"]
    ode_func.load_state_dict(ode_state)
    ode_func.to(device)

    classifier_state = checkpt["classifier"]
    classifier.load_state_dict(classifier_state)
    classifier.to(device)

    return encoder, ode_func, classifier


"""
TRAINING
"""


def train_neural_ode(
    random_seed, train, validate, model, fold, lr, tol, epochs, l2, hidden_dim, latent_dim, ode_hidden_dim
):
    # choose whether to use a GPU if it is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set various seeds for complete reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # the model checkpoints will be stored in this directory
    utils.makedirs(f"fold_{fold}")
    ckpt_path = os.path.join(f"fold_{fold}", f"fold_{fold}_model_{model}.ckpt")

    # for the logging we'll have this informative string
    input_cmd = f"--fold {fold} --model {model} --lr {lr} --tol {tol} --epochs {epochs} --l2 {l2} --hidden_dim {hidden_dim} --laten_dim {latent_dim}"

    tdm1_obj = parse_tdm1(device, train, validate, None, phase="train")
    input_dim = tdm1_obj["input_dim"]

    # put the model together
    encoder = Encoder(input_dim=input_dim, output_dim=2 * latent_dim, hidden_dim=hidden_dim).to(device)
    ode_func = ODEFunc(input_dim=latent_dim, hidden_dim=ode_hidden_dim).to(device)
    classifier = Classifier(latent_dim=latent_dim, output_dim=1).to(device)

    # make the logs
    logger.info(input_cmd)

    batches_per_epoch = tdm1_obj["n_train_batches"]
    criterion = nn.MSELoss().to(device=device)  # mean squared error loss
    params = list(encoder.parameters()) + list(ode_func.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=l2)  # most common neural network optimizer
    best_rmse = 0x7FFFFFFF  # initialize the best rmse to a very large number
    best_epochs = 0  # will be updated with the epoch number of the best rmse

    for epoch in range(1, epochs + 1):

        for _ in tqdm(range(batches_per_epoch), ascii=True):
            optimizer.zero_grad()
            # generate data batch
            ptnm, times, features, labels, cmax_time = tdm1_obj["train_dataloader"].__next__()
            # get predictions using current state of model
            preds = predict(encoder, ode_func, classifier, tol, latent_dim, ptnm, times, features, cmax_time, device)
            idx_not_nan = ~(torch.isnan(labels) | (labels == -1))
            preds = preds[idx_not_nan]
            labels = labels[idx_not_nan]

            # compute loss of predictions
            loss = torch.sqrt(criterion(preds, labels))

            # update model based on loss
            try:
                loss.backward()
            except RuntimeError:
                print(ptnm)
                print(times)
                continue
            optimizer.step()

        with torch.no_grad():
            # compute training loss on all batches
            train_res = compute_loss(
                encoder,
                ode_func,
                classifier,
                tol,
                latent_dim,
                tdm1_obj["train_dataloader"],
                tdm1_obj["n_train_batches"],
                device,
                phase="train",
            )

            # compute validation loss on all batches
            validation_res = compute_loss(
                encoder,
                ode_func,
                classifier,
                tol,
                latent_dim,
                tdm1_obj["val_dataloader"],
                tdm1_obj["n_val_batches"],
                device,
                phase="validate",
            )

            train_loss = train_res["loss"]
            validation_loss = validation_res["loss"]

            # save model if it beats previous best RMSE (initialized at very high value)
            if validation_loss < best_rmse:
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "ode": ode_func.state_dict(),
                        "classifier": classifier.state_dict(),
                    },
                    ckpt_path,
                )
                best_rmse = validation_loss
                best_epochs = epoch

            message = """
            Epoch {:04d} | Training loss {:.6f} | Training R2 {:.6f} | Validation loss {:.6f} | Validation R2 {:.6f}
            Best loss {:.6f} | Best epoch {:04d}
            """.format(
                epoch, train_loss, train_res["r2"], validation_loss, validation_res["r2"], best_rmse, best_epochs
            )
            logger.info(message)


"""
PREDICTION
"""


def sample_standard_gaussian(mu, sigma):
    device = torch.device("cpu")
    if mu.is_cuda:
        device = mu.get_device()

    d = torch.distributions.normal.Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.0]).to(device))
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()


def predict(encoder, ode_func, classifier, tol, latent_dim, ptnm, times, features, cmax_time, device="cpu"):
    # generate dosing information to integrate into the latent features that go into the ODE solver
    dosing = torch.zeros([features.size(0), features.size(1), latent_dim])
    dosing[:, :, 0] = features[:, :, -2]
    dosing = dosing.permute(1, 0, 2).to(device)

    # Get encoder output and sample latent features from a gaussian latent
    # distribution. Uses the first half of encoder output to estimate mean
    # and the second half to estimate variance. This technique is inspired
    # by a variational autoencoder, which learns distributions for latent
    # variables.
    # The authors never discuss and it is unclear to us why they chose this
    # method over learning latent variable values directly.
    encoder_out = encoder(features.to(device))
    qz0_mean, qz0_var = encoder_out[:, :latent_dim], encoder_out[:, latent_dim:]
    z0 = sample_standard_gaussian(qz0_mean, qz0_var)

    # add sampled latent variable value as initial state of solver
    solves = z0.unsqueeze(0).clone().to(device)
    try:
        # iterate through time intervals
        for idx, (time0, time1) in enumerate(zip(times[:-1], times[1:])):
            z0 += dosing[idx]  # add dosing information
            time_interval = torch.Tensor([time0 - time0, time1 - time0])  # compute time interval
            # use ODE solver to integrate dosing information and time interval
            sol = odeint(ode_func.to(device), z0.to(device), time_interval.to(device), rtol=tol, atol=tol)
            # feed output of ODE solver to next time point
            z0 = sol[-1].clone()
            # assemble ODE solver outputs for time intervals
            solves = torch.cat([solves, sol[-1:, :]], 0)
    except AssertionError:
        print(times)
        print(time0, time1, time_interval, ptnm)

    # decoder step, in which we feed the above ODE solver outputs per time
    # interval concatenated with time and PK reponse for first cycle
    preds = classifier(solves, cmax_time).permute(1, 0, 2)

    return preds


def merge_predictions(evals_per_fold, reference_data):
    cols = ["PTNM", "TIME", "preds", "labels"]
    left = evals_per_fold[0][cols]
    for right in evals_per_fold[1:]:
        left = left.merge(right[cols], on=["PTNM", "TIME", "labels"], how="left")
    preds = [col for col in left.columns.values if col.startswith("preds")]
    left["pred_agg"] = left[preds].agg("mean", axis=1)

    ref = reference_data[["PTNM", "DSFQ"]].drop_duplicates()
    # just making sure the two columns that we are joining on have the same type
    # or else there would be an error
    ref.loc[:, "PTNM"] = ref["PTNM"].astype(int)
    left.loc[:, "PTNM"] = left["PTNM"].astype(int)
    left = left.merge(ref, on="PTNM", how="left")
    # get rid of the first round of treatment
    left_q1w = left[(left.DSFQ == 1) & (left.TIME >= 168)]
    left_q3w = left[(left.DSFQ == 3) & (left.TIME >= 504)]
    return pd.concat([left_q1w, left_q3w], ignore_index=False)


def predict_using_trained_model(test, model, fold, tol, hidden_dim, latent_dim, ode_hidden_dim):
    """
    This method loads the best available model for the specified training and generates predictions.
    Intended to generate predictions on validation and test sets.
    """
    # choose whether to use a GPU if it is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # the model checkpoints and evaluation results will be stored in these directories
    ckpt_path = os.path.join(f"fold_{fold}", f"fold_{fold}_model_{model}.ckpt")
    eval_path = os.path.join(f"fold_{fold}", f"fold_{fold}_model_{model}.csv")

    # create the test data object
    tdm1_obj = parse_tdm1(device, None, None, test, phase="test")
    input_dim = tdm1_obj["input_dim"]

    # load best trained model
    encoder, ode_func, classifier = load_model(ckpt_path, input_dim, hidden_dim, latent_dim, ode_hidden_dim, device)

    ## Predict & Evaluate
    with torch.no_grad():
        test_res = compute_loss(
            encoder,
            ode_func,
            classifier,
            tol,
            latent_dim,
            tdm1_obj["test_dataloader"],
            tdm1_obj["n_test_batches"],
            device,
            phase="test",
        )

    # save evaluation results to a csv
    eval_results = pd.DataFrame(test_res).drop(columns="loss")
    eval_results.to_csv(eval_path, index=False)

    return eval_results


"""
EVALUATION 
"""


def compute_loss(encoder, ode_func, classifier, tol, latent_dim, dataloader, n_batches, device, phase):
    ptnms = []
    Times = torch.Tensor([]).to(device=device)
    predictions = torch.Tensor([]).to(device=device)
    ground_truth = torch.Tensor([]).to(device=device)

    for _ in range(n_batches):
        ptnm, times, features, labels, cmax_time = dataloader.__next__()
        preds = predict(encoder, ode_func, classifier, tol, latent_dim, ptnm, times, features, cmax_time, device)

        idx_not_nan = ~(torch.isnan(labels) | (labels == -1))
        preds = preds[idx_not_nan]
        labels = labels[idx_not_nan]

        if phase == "test":
            times = times[idx_not_nan.flatten()]
            ptnms += ptnm * len(times)
            Times = torch.cat(
                (Times, times * 24)
            )  # 24 refers to 4 time-dependent features and concatenated zero-padded round 1 features

        predictions = torch.cat((predictions, preds))
        ground_truth = torch.cat((ground_truth, labels))

    rmse_loss = mean_squared_error(ground_truth.cpu().numpy(), predictions.cpu().numpy(), squared=False)
    r2 = r2_score(ground_truth.cpu().numpy(), predictions.cpu().numpy())

    if phase == "test":
        return {
            "PTNM": ptnms,
            "TIME": Times.cpu().numpy(),
            "labels": ground_truth.cpu().tolist(),
            "preds": predictions.cpu().tolist(),
            "loss": rmse_loss,
            "r2": r2,
        }
    else:
        return {"labels": ground_truth.cpu().tolist(), "preds": predictions.cpu().tolist(), "loss": rmse_loss, "r2": r2}
