import torch
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from train_predict_utils import predict


def merge_predictions(evals_per_fold, reference_data):
    cols = ["PTNM", "TIME", "preds", "labels"]
    left = evals_per_fold[0][cols]
    for right in evals_per_fold[1:]:
        left = left.merge(right[cols], on=["PTNM", "TIME", "labels"], how="left")
    preds = [col for col in left.columns.values if col.startswith("preds")]
    left["pred_agg"] = left[preds].agg("mean", axis=1)

    ref = reference_data[["PTNM", "DSFQ"]].drop_duplicates()
    left = left.merge(ref, on="PTNM", how="left")
    # get rid of the first round of treatment
    left_q1w = left[(left.DSFQ == 1) & (left.TIME >= 168)]
    left_q3w = left[(left.DSFQ == 3) & (left.TIME >= 504)]
    return pd.concat([left_q1w, left_q3w], ignore_index=False)


def compute_loss_on_train(criterion, labels, preds):
    idx_not_nan = ~(torch.isnan(labels) | (labels == -1))
    preds = preds[idx_not_nan]
    labels = labels[idx_not_nan]
    return torch.sqrt(criterion(preds, labels))


def compute_loss_on_test(encoder, ode_func, classifier, tol, latent_dim, dataloader, n_batches, device, phase):
    ptnms = []
    Times = torch.Tensor([]).to(device=device)
    predictions = torch.Tensor([]).to(device=device)
    ground_truth = torch.Tensor([]).to(device=device)

    for _ in range(n_batches):
        ptnm, times, features, labels, cmax_time = dataloader.__next__()
        preds = predict(encoder, ode_func, classifier, tol, latent_dim, ptnm, times, features, cmax_time)

        idx_not_nan = ~(torch.isnan(labels) | (labels == -1))
        preds = preds[idx_not_nan]
        labels = labels[idx_not_nan]

        if phase == "test":
            times = times[idx_not_nan.flatten()]
            ptnms += ptnm * len(times)
            Times = torch.cat((Times, times * 24))

        predictions = torch.cat((predictions, preds))
        ground_truth = torch.cat((ground_truth, labels))

    rmse_loss = mean_squared_error(ground_truth.cpu().numpy(), predictions.cpu().numpy(), squared=False)
    r2 = r2_score(ground_truth.cpu().numpy(), predictions.cpu().numpy())

    if phase == "test":
        return {
            "PTNM": ptnms,
            "TIME": Times,
            "labels": ground_truth.cpu().tolist(),
            "preds": predictions.cpu().tolist(),
            "loss": rmse_loss,
            "r2": r2,
        }
    else:
        return {"labels": ground_truth.cpu().tolist(), "preds": predictions.cpu().tolist(), "loss": rmse_loss, "r2": r2}
