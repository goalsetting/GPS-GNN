# =====================================
# trainer/pretrain.py
# =====================================

import torch


def pretrain(

    model,

    loader,

    optimizer,

    criterion,

    epochs,

    device,

    save_path

):

    model.train()

    best=1e9

    for ep in range(
        epochs
    ):

        total=0

        for batch in loader:

            edge=batch.edge_index.to(
                device
            )

            weight=batch.weight.to(
                device
            )

            expr=batch.expr.to(
                device
            )

            cid=batch.cell_id.to(
                device
            )

            gid=batch.gene_id.to(
                device
            )

            emb=model(

                edge,

                weight

            )

            pred_expr,\
            pred_edge=(
                model.decode(

                    cid,

                    gid,

                    emb

                )
            )

            loss=criterion(

                pred_expr,

                pred_edge,

                expr

            )

            optimizer.zero_grad()

            loss[
                "loss"
            ].backward()

            optimizer.step()

            total+=(
                loss[
                    "loss"
                ].item()
            )

        total/=len(
            loader
        )

        print(
            ep,
            total
        )

        if total<best:

            best=total

            torch.save({

                "model":
                model.state_dict(),

                "cell":
                model.cell_emb.weight,

                "gene":
                model.gene_emb.weight

            },

            save_path)

    return best