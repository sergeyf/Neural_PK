import os
import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import utils
from evaluation_utils import predict, compute_loss
from model import Encoder, ODEFunc, Classifier
from data_parse import parse_tdm1
from datetime import datetime


log_path = "logs/" + f"{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.log"
utils.makedirs("logs/")
logger = utils.get_logger(logpath=log_path)


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
    encoder = Encoder(input_dim=input_dim, output_dim=2 * latent_dim, hidden_dim=hidden_dim)
    ode_func = ODEFunc(input_dim=latent_dim, hidden_dim=ode_hidden_dim)
    classifier = Classifier(latent_dim=latent_dim, output_dim=1)

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
            preds = predict(encoder, ode_func, classifier, tol, latent_dim, ptnm, times, features, cmax_time)
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


def load_model(ckpt_path, input_dim, hidden_dim, latent_dim, ode_hidden_dim, device="cpu"):
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")

    # create the parts of the model into which we load our checkpoint state
    encoder = Encoder(input_dim=input_dim, output_dim=2 * latent_dim, hidden_dim=hidden_dim)
    ode_func = ODEFunc(input_dim=latent_dim, hidden_dim=ode_hidden_dim)
    classifier = Classifier(latent_dim=latent_dim, output_dim=1)

    checkpt = torch.load(ckpt_path)

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


def predict_using_trained_model(test, model, fold, tol, hidden_dim, latent_dim, ode_hidden_dim):
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
