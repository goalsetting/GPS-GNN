# ============================================================
# engine.py
# ============================================================

import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)


# ============================================================
# TRAIN
# ============================================================

def train(

    model,

    loader,

    optimizer,

    loss_fn,

    epochs,

    device,

    save

):

    best=1e9

    for ep in range(
        epochs
    ):

        model.train()

        total=0

        for batch in loader:

            batch=batch.to(
                device
            )

            H,\
            hs,\
            he,\
            expr=(

                model(
                    batch
                )

            )

            out=loss_fn(

                hs,

                he,

                batch.edge_s,

                batch.edge_e,

                expr

            )

            optimizer.zero_grad()

            out[
                "loss"
            ].backward()

            optimizer.step()

            total+=(
                out[
                    "loss"
                ].item()
            )

        total/=len(
            loader
        )

        print(

            f"epoch={ep}",

            total

        )

        if total<best:

            best=total

            torch.save({

                "epoch":ep,

                "model":

                model.state_dict()

            },

            save

            )


# ============================================================
# EVAL
# ============================================================

@torch.no_grad()

def evaluate(

    model,

    loader,

    device,

    n_cluster

):

    model.eval()

    emb=[]

    lab=[]

    for batch in loader:

        batch=batch.to(
            device
        )

        H,_,_,_=model(
            batch
        )

        emb.append(
            H.cpu()
        )

        if hasattr(

            batch,

            "label"

        ):

            lab.append(

                batch.label

            )

    emb=torch.cat(
        emb
    )

    pred=KMeans(

        n_cluster

    ).fit_predict(

        emb

    )

    result={}

    if len(lab):

        lab=torch.cat(
            lab
        )

        result[

            "ARI"

        ]=(

            adjusted_rand_score(

                lab,

                pred

            )

        )

        result[

            "NMI"

        ]=(

            normalized_mutual_info_score(

                lab,

                pred

            )

        )

    result[

        "SIL"

    ]=(

        silhouette_score(

            emb,

            pred

        )

    )

    return result


# ============================================================
# MAIN
# ============================================================

def run(

    model,

    loader,

    optimizer,

    loss_fn,

    epochs,

    device,

    save,

    cluster

):

    train(

        model,

        loader,

        optimizer,

        loss_fn,

        epochs,

        device,

        save

    )

    ckpt=torch.load(
        save
    )

    model.load_state_dict(

        ckpt[
            "model"
        ]

    )

    res=evaluate(

        model,

        loader,

        device,

        cluster

    )

    print(
        res
    )