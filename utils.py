# Les fonctions utiles ici
import pandas as pd
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from model import PINNs
import torch
import time
from geometry import RectangleWithoutCylinder
import numpy as np


def write_csv(data, path, file_name):
    dossier = Path(path)
    df = pd.DataFrame(data)
    # Créer le dossier si il n'existe pas
    dossier.mkdir(parents=True, exist_ok=True)
    df.to_csv(path + file_name)


def read_csv(path):
    return pd.read_csv(path)


def charge_data(hyper_param, param_adim):
    """
    Charge the data of X_full, U_full with every points
    And X_train, U_train with less points
    """
    # La data
    # On adimensionne la data

    time_start_charge = time.time()
    f = 0.5 * (hyper_param["H"] / hyper_param["m"]) ** 0.5 / hyper_param['nb_period_plot']  # la fréquence de l'écoulement
    t_max = hyper_param['t_min'] + hyper_param['nb_period'] / f 
    df = pd.read_csv("data/" + hyper_param["file"])
    df_modified = df.loc[
        (df["Points:0"] >= hyper_param["x_min"])
        & (df["Points:0"] <= hyper_param["x_max"])
        & (df["Points:1"] >= hyper_param["y_min"])
        & (df["Points:1"] <= hyper_param["y_max"])
        & (df["Time"] > hyper_param["t_min"])
        & (df["Time"] < t_max)
        & (df["Points:2"] == 0.0)
        & (df["Points:0"] ** 2 + df["Points:1"] ** 2 > (0.025 / 2) ** 2),
        :,
    ].copy()

    # Adimensionnement
    x_full = torch.tensor(df_modified["Points:0"].to_numpy(), dtype=torch.float32) / param_adim["L"]
    y_full = torch.tensor(df_modified["Points:1"].to_numpy(), dtype=torch.float32) / param_adim["L"] 
    time_without_modulo = df_modified["Time"].to_numpy() - hyper_param['t_min']
    time_with_modulo = hyper_param['t_min'] + time_without_modulo % (1/f)
    t_full = torch.tensor(time_with_modulo, dtype=torch.float32) / (param_adim["L"] / param_adim["V"])
    u_full = torch.tensor(df_modified["Velocity:0"].to_numpy(), dtype=torch.float32) / param_adim["V"]
    v_full = torch.tensor(df_modified["Velocity:1"].to_numpy(), dtype=torch.float32) / param_adim["V"]
    p_full = torch.tensor(df_modified["Pressure"].to_numpy(), dtype=torch.float32) / ((param_adim["V"] ** 2) * param_adim["rho"])

    # les valeurs pour renormaliser ou dénormaliser
    mean_std = {
        "u_mean": u_full.mean(),
        "v_mean": v_full.mean(),
        "p_mean": p_full.mean(),
        "x_mean": x_full.mean(),
        "y_mean": y_full.mean(),
        "t_mean": t_full.mean(),
        "x_std": x_full.std(),
        "y_std": y_full.std(),
        "t_std": t_full.std(),
        "u_std": u_full.std(),
        "v_std": v_full.std(),
        "p_std": p_full.std(),
    }



    X_full = torch.zeros((0, 3))
    U_full = torch.zeros((0, 3))

    # Normalisation Z
    x_norm_full = (x_full - mean_std["x_mean"]) / mean_std["x_std"]
    y_norm_full = (y_full - mean_std["y_mean"]) / mean_std["y_std"]
    t_norm_full = (t_full - mean_std["t_mean"]) / mean_std["t_std"]
    p_norm_full = (p_full - mean_std["p_mean"]) / mean_std["p_std"]
    u_norm_full = (u_full - mean_std["u_mean"]) / mean_std["u_std"]
    v_norm_full = (v_full - mean_std["v_mean"]) / mean_std["v_std"]
    X_full = torch.stack((x_norm_full, y_norm_full, t_norm_full), dim=1)
    U_full = torch.stack((u_norm_full, v_norm_full, p_norm_full), dim=1)

    X_train = torch.zeros((0, 3))
    U_train = torch.zeros((0, 3))
    print("Starting X_train")

    print(f"Time:{(time.time()-time_start_charge):.3f}")
    for time_ in torch.unique(t_norm_full):
        # les points autour du cylindre dans un rayon de hyper_param['rayon_proche']
        masque = (
            ((x_full ** 2 + y_full ** 2) < ((hyper_param['rayon_close_cylinder'] / param_adim["L"]) ** 2))
            & (t_norm_full == time_)
        )
        indices = torch.randperm(len(x_norm_full[masque]))[
            : hyper_param["nb_points_close_cylinder"]
        ]

        new_x = torch.stack(
            (
                x_norm_full[masque][indices],
                y_norm_full[masque][indices],
                t_norm_full[masque][indices],
            ),
            dim=1,
        )
        new_y = torch.stack(
            (
                u_norm_full[masque][indices],
                v_norm_full[masque][indices],
                p_norm_full[masque][indices],
            ),
            dim=1,
        )
        X_train = torch.cat((X_train, new_x))
        U_train = torch.cat((U_train, new_y))


        # Les points avec 'latin hypercube sampling'
        masque = (t_norm_full == time_) 
        if x_norm_full[masque].size(0) > 0:
            indices = torch.randperm(x_norm_full[masque].size(0))[
                : hyper_param["nb_points"]
            ]
            new_x = torch.stack(
                (
                    x_norm_full[masque][indices],
                    y_norm_full[masque][indices],
                    t_norm_full[masque][indices],                ),
                dim=1,
            )
            new_y = torch.stack(
                (
                    u_norm_full[masque][indices],
                    v_norm_full[masque][indices],
                    p_norm_full[masque][indices],
                ),
                dim=1,
            )
            X_train = torch.cat((X_train, new_x))
            U_train = torch.cat((U_train, new_y))
    indices = torch.randperm(X_train.size(0))
    X_train = X_train[indices]
    U_train = U_train[indices]
    print("X_train OK")

    # les points du bord
    teta_int = torch.linspace(0, 2 * torch.pi, hyper_param["nb_points_border"])
    X_border = torch.empty((0, 3))
    x_ = (
        (((0.025 / 2) * torch.cos(teta_int)) / param_adim["L"]) - mean_std["x_mean"]
    ) / mean_std["x_std"]
    y_ = (
        (((0.025 / 2) * torch.sin(teta_int)) / param_adim["L"]) - mean_std["y_mean"]
    ) / mean_std["y_std"]

    for time_ in torch.unique(t_norm_full):
        new_x = torch.stack(
            (x_, y_, torch.ones_like(x_) * time_), dim=1
        )
        X_border = torch.cat((X_border, new_x))
    indices = torch.randperm(X_border.size(0))
    X_border = X_border[indices]
    print("X_border OK")

    teta_int_test = torch.linspace(0, 2 * torch.pi, 15)
    X_border_test = torch.zeros((0, 3))
    x_ = (
        (((0.025 / 2) * torch.cos(teta_int_test)) / param_adim["L"])
        - mean_std["x_mean"]
    ) / mean_std["x_std"]
    y_ = (
        (((0.025 / 2) * torch.sin(teta_int_test)) / param_adim["L"])
        - mean_std["y_mean"]
    ) / mean_std["y_std"]

    for time_ in torch.unique(t_norm_full):
        new_x = torch.stack(
            (x_, y_, torch.ones_like(x_) * time_), dim=1
        )
        X_border_test = torch.cat((X_border_test, new_x))

    # On charge le pde 
    # le domaine de résolution
    rectangle = RectangleWithoutCylinder(
        x_max=X_full[:, 0].max(),
        y_max=X_full[:, 1].max(),
        t_min=X_full[:, 2].min(),
        t_max=X_full[:, 2].max(),
        x_min=X_full[:, 0].min(),
        y_min=X_full[:, 1].min(),
        x_cyl=0,
        y_cyl=0,
        r_cyl=0.025/2,
        mean_std=mean_std,
        param_adim=param_adim,
    )

    X_pde = rectangle.generate_lhs(hyper_param["nb_points_pde"]).detach()
    print("X_pde OK")

    # Data test loading
    X_test_pde = rectangle.generate_lhs(hyper_param["n_pde_test"]).detach()
    points_coloc_test = np.random.choice(
        len(X_full), hyper_param["n_data_test"], replace=False
    )
    X_test_data = X_full[points_coloc_test]
    U_test_data = U_full[points_coloc_test]
    return X_train, U_train, X_full, U_full, X_border, X_border_test, mean_std, X_pde, X_test_pde, X_test_data, U_test_data


def init_model(f, hyper_param, device, folder_result):
    model = PINNs(hyper_param).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyper_param["lr_init"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=hyper_param["gamma_scheduler"]
    )
    loss = nn.MSELoss()
    # On regarde si notre modèle n'existe pas déjà
    if Path(folder_result + "/model_weights.pth").exists():
        # Charger l'état du modèle et de l'optimiseur
        checkpoint = torch.load(folder_result + "/model_weights.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        weights = checkpoint["weights"]
        print("\nModèle chargé\n", file=f)
        print("\nModèle chargé\n")
        csv_train = read_csv(folder_result + "/train_loss.csv")
        csv_test = read_csv(folder_result + "/test_loss.csv")
        train_loss = {
            "total": list(csv_train["total"]),
            "data": list(csv_train["data"]),
            "pde": list(csv_train["pde"]),
            "border": list(csv_train["border"]),
        }
        test_loss = {
            "total": list(csv_test["total"]),
            "data": list(csv_test["data"]),
            "pde": list(csv_test["pde"]),
            "border": list(csv_test["border"]),
        }
        print("\nLoss chargée\n", file=f)
        print("\nLoss chargée\n")
    else:
        print("Nouveau modèle\n", file=f)
        print("Nouveau modèle\n")
        train_loss = {"total": [], "data": [], "pde": [], "border": []}
        test_loss = {"total": [], "data": [], "pde": [], "border": []}
        weights = {
            "weight_data": hyper_param["weight_data"],
            "weight_pde": hyper_param["weight_pde"],
            "weight_border": hyper_param["weight_border"],
        }
    return model, optimizer, scheduler, loss, train_loss, test_loss, weights


if __name__ == "__main__":
    write_csv([[1, 2, 3], [4, 5, 6]], "ready_cluster/piche/test.csv")
