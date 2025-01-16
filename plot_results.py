import numpy as np
import torch
from model import PINNs
import pandas as pd
from utils import charge_data
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import read_csv


def plot_flow(
    x,
    y,
    t,
    norme_vitesse_data,
    norme_vitesse_predict,
    name_file,
    fps=7,
    title="Norme vitesse",
):
    # Créer une figure et des axes
    # Ajuster la taille de la figure
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Déterminer les valeurs min et max pour la colormap
    vmin = min(np.min(norme_vitesse_data), np.min(norme_vitesse_predict))
    vmax = max(np.max(norme_vitesse_data), np.max(norme_vitesse_predict))

    # Initialiser les cartes de chaleur
    indices = np.where(t == np.min(t))
    c1 = ax[0].tripcolor(
        x[indices],
        y[indices],
        norme_vitesse_data[indices],
        shading="gouraud",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )
    c2 = ax[1].tripcolor(
        x[indices],
        y[indices],
        norme_vitesse_predict[indices],
        shading="gouraud",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )

    ax[0].set_title("Données")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")

    ax[1].set_title("Prédictions")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")

    # Ajouter une barre de couleur
    cbar = fig.colorbar(c1, ax=ax, orientation="vertical", label=f"{title}")

    # Fonction d'initialisation
    def init():
        return c1, c2

    # Fonction d'animation
    def update(frame):
        print(frame)
        time = list(set(t))
        time.sort()
        indices = np.where(t == time[frame])

        # Mettre à jour la première carte de chaleur
        c1.set_array(norme_vitesse_data[indices].flatten())
        ax[0].set_title("Données")

        # Mettre à jour la deuxième carte de chaleur
        c2.set_array(norme_vitesse_predict[indices].flatten())
        ax[1].set_title("Prédictions")

        # Titre général
        plt.suptitle(f"{title} à t={time[frame]:.2f}", fontsize=16)

        return c1, c2

    # Créer l'animation
    ani = FuncAnimation(
        fig, update, frames=len(set(t)), init_func=init, blit=False, repeat=True
    )
    ani.save(name_file, writer="pillow", fps=fps)

    plt.show()  # Afficher la figure à la fin


# Exemple d'utilisation
# plot_flow(x, y, t, norme_vitesse_data, norme_vitesse_predict, 'animation.gif')


def plot_loss(file_save, file, title_graph):
    plt.clf()
    csv_train = read_csv(file + "/train_loss.csv")
    csv_test = read_csv(file + "/test_loss.csv")
    train_total = list(csv_train["total"])
    test_total = list(csv_test["total"])

    plt.figure(figsize=(10, 6))
    epochs = np.arange(len(train_total))
    plt.plot(epochs, train_total, label="Loss d'entraînement")
    plt.plot(epochs, test_total, label="Loss de test")

    # Configuration des axes logarithmiques
    plt.yscale("log")

    # Ajout de titres et légendes
    plt.title("Loss d'entraînement et de test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # Affichage du graphique
    plt.savefig(file_save + "/" + title_graph)


def plot_loss_decompose(file_save, file, title_graph):
    plt.clf()
    csv_train = read_csv(file + "/train_loss.csv")
    train_data = list(csv_train["data"])
    train_pde = list(csv_train["pde"])
    train_border = list(csv_train["border"])

    csv_test = read_csv(file + "/test_loss.csv")
    test_data = list(csv_test["data"])
    test_pde = list(csv_test["pde"])
    test_border = list(csv_test["border"])

    plt.figure(figsize=(10, 6))
    epochs = np.arange(len(train_data))
    plt.plot(epochs, train_pde, label="Loss pde train", color="blue")
    plt.plot(epochs, train_data, label="Loss data train", color="green")
    plt.plot(epochs, train_border, label="Loss border train", color="red")

    plt.plot(epochs, test_pde, label="Loss pde test",
             linestyle="--", color="cyan")
    plt.plot(epochs, test_data, label="Loss data test",
             linestyle="--", color="lime")
    plt.plot(
        epochs, test_border, label="Loss border test", linestyle="--", color="orange"
    )

    # Configuration des axes logarithmiques
    plt.yscale("log")

    # Ajout de titres et légendes
    plt.title("Loss d'entraînement et de test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # Affichage du graphique
    plt.savefig(file_save + "/" + title_graph)


def plot_points(X_train, X_border, mean_std, param_adim, file_save):
    plt.clf()
    masque = X_train[:, 2] == np.unique(X_train[:, 2])[0]
    x = (X_train[:, 0][masque] * mean_std["x_std"] + mean_std["x_mean"]) * param_adim[
        "L"
    ]
    y = (X_train[:, 1][masque] * mean_std["y_std"] + mean_std["y_mean"]) * param_adim[
        "L"
    ]
    x_border = (X_border[:, 0] * mean_std["x_std"] +
                mean_std["x_mean"]) * param_adim["L"]
    y_border = (X_border[:, 1] * mean_std["y_std"] +
                mean_std["y_mean"]) * param_adim["L"]
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(
        "Points de colocations pour un pas de temps (12*12 + 50 points + 25 points)"
    )
    plt.grid()
    plt.scatter(x, y, marker=".", color="blue", label="data")
    plt.scatter(x_border, y_border, marker=".", color="green", label="border")
    plt.legend()
    plt.axis("equal")
    plt.savefig(file_save)


def plot_cl(X_full, U_full, model, param_adim, mean_std, file_save, r=0.025 / 2):
    plt.clf()
    x_min = ((-0.02 / param_adim["L"]) -
             mean_std["x_mean"]) / mean_std["x_std"]
    x_max = ((0.02 / param_adim["L"]) - mean_std["x_mean"]) / mean_std["x_std"]
    y_min = ((-0.02 / param_adim["L"]) -
             mean_std["y_mean"]) / mean_std["y_std"]
    y_max = ((0.02 / param_adim["L"]) - mean_std["y_mean"]) / mean_std["y_std"]
    masque = (
        (X_full[:, 0] > x_min)
        & (X_full[:, 0] < x_max)
        & (X_full[:, 1] > y_min)
        & (X_full[:, 1] < y_max)
    )
    X_red = X_full[masque]
    U_red = U_full[masque]

    def P_base(theta, t_ad):
        """Trouve la pression la plus proche en theta et à t"""
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        x_ad = ((x / param_adim["L"]) - mean_std["x_mean"]) / mean_std["x_std"]
        y_ad = ((y / param_adim["L"]) - mean_std["y_mean"]) / mean_std["y_std"]
        distances = np.linalg.norm(
            X_red - np.array([x_ad, y_ad, t_ad]), axis=1)
        indice_proche = np.argmin(distances)
        P_dim = ((U_red[indice_proche][2] * mean_std["p_std"] + mean_std["p_mean"])) * (
            (param_adim["V"] ** 2) * param_adim["rho"]
        )
        return P_dim

    def P(theta, t_ad, model_use=model):
        """Donne la pression prédite par le modèle"""
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        x_ad = ((x / param_adim["L"]) - mean_std["x_mean"]) / mean_std["x_std"]
        y_ad = ((y / param_adim["L"]) - mean_std["y_mean"]) / mean_std["y_std"]
        P_adim = model_use(torch.tensor(
            [x_ad, y_ad, t_ad], dtype=torch.float32))[2]
        P = ((P_adim * mean_std["p_std"] + mean_std["p_mean"])) * (
            (param_adim["V"] ** 2) * param_adim["rho"]
        )
        return P.detach().item()

    def force_portance(t_ad, N_points=100, model_use=model):
        """Calcul la force de portance au temps t_ad"""
        d_theta = 2 * np.pi / N_points
        sum_integrale = 0
        for theta in np.linspace(0, 2 * np.pi, N_points):
            sum_integrale += -P(theta, t_ad, model_use) * \
                np.sin(theta) * r * d_theta
        return sum_integrale

    def force_portance_base(t_ad, N_points=100):
        """Calcul la force de portance au temps t_ad de base"""
        d_theta = 2 * np.pi / N_points
        sum_integrale = 0
        for theta in np.linspace(0, 2 * np.pi, N_points):
            sum_integrale += -P_base(theta, t_ad) * np.sin(theta) * r * d_theta
        return sum_integrale

    portance_base = np.array([force_portance_base(t_)
                             for t_ in np.unique(X_red[:, 2])])
    Cp_base = portance_base / \
        (0.5 * param_adim["rho"] * param_adim["V"] ** 2 * (2 * r))
    time = np.linspace(
        np.unique(X_red[:, 2]).min(), np.unique(X_red[:, 2]).max(), 300)
    portance = np.array([force_portance(t_) for t_ in time])
    Cp = portance / (0.5 * param_adim["rho"] * param_adim["V"] ** 2 * (2 * r))
    time_1 = (np.unique(X_red[:, 2]) * mean_std["t_std"] + mean_std["t_mean"]) * (
        param_adim["L"] / param_adim["V"]
    )
    time_dim = (time * mean_std["t_std"] + mean_std["t_mean"]) * (
        param_adim["L"] / param_adim["V"]
    )
    plt.plot(time_1, Cp_base, color="r", marker="o",
             label="données", linestyle="")
    plt.plot(time_dim, Cp, color="b", label="prédictions")
    plt.grid()
    plt.xlabel("t")
    plt.ylabel("Cl")
    plt.title("Coefficient de portance en fonction du temps")
    plt.legend()
    plt.savefig(file_save)


def plot_results(file_name, param_adim, epoch="", title_loss="loss_graph"):
    """Plot les vidéos pression, vitesse, la loss, la courbe de Cp, à un temps les points de coloc"""
    # On charge le modele et les data
    with open("results/" + file_name + "/hyper_param.json", "r") as file:
        hyper_param = json.load(file)
    model = PINNs(hyper_param)
    checkpoint = torch.load(
        "results/" + file_name + "/" + epoch + "/" + "model_weights.pth",
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    X_train, U_train, X_full, U_full, X_border, X_border_test, mean_std = charge_data(
        hyper_param, param_adim
    )
    X_pred = torch.tensor(X_full, dtype=torch.float32)
    U_pred = model(X_pred)
    model.load_state_dict(checkpoint["model_state_dict"])
    x_norm_pred, y_norm_pred, t_norm_pred = (
        X_pred.detach().numpy()[:, 0],
        X_pred.detach().numpy()[:, 1],
        X_pred.detach().numpy()[:, 2],
    )
    u_norm_pred, v_norm_pred, p_norm_pred = (
        U_pred.detach().numpy()[:, 0],
        U_pred.detach().numpy()[:, 1],
        U_pred.detach().numpy()[:, 2],
    )
    x_pred = (x_norm_pred * mean_std["x_std"] +
              mean_std["x_mean"]) * param_adim["L"]
    y_pred = (y_norm_pred * mean_std["y_std"] +
              mean_std["y_mean"]) * param_adim["L"]
    t_pred = (t_norm_pred * mean_std["t_std"] + mean_std["t_mean"]) * (
        param_adim["L"] / param_adim["V"]
    )
    u_pred = (u_norm_pred * mean_std["u_std"] +
              mean_std["u_mean"]) * param_adim["V"]
    v_pred = (v_norm_pred * mean_std["v_std"] +
              mean_std["v_mean"]) * param_adim["V"]
    p_pred = (p_norm_pred * mean_std["p_std"] + mean_std["p_mean"]) * (
        (param_adim["V"] ** 2) * param_adim["rho"]
    )
    norme_vitesse_pred = np.sqrt(u_pred**2 + v_pred**2)
    x_data = (X_full[:, 0] * mean_std["x_std"] +
              mean_std["x_mean"]) * param_adim["L"]
    y_data = (X_full[:, 1] * mean_std["y_std"] +
              mean_std["y_mean"]) * param_adim["L"]
    t_data = (X_full[:, 2] * mean_std["t_std"] + mean_std["t_mean"]) * (
        param_adim["L"] / param_adim["V"]
    )
    u_data = (U_full[:, 0] * mean_std["u_std"] +
              mean_std["u_mean"]) * param_adim["V"]
    v_data = (U_full[:, 1] * mean_std["v_std"] +
              mean_std["v_mean"]) * param_adim["V"]
    p_data = (U_full[:, 2] * mean_std["p_std"] + mean_std["p_mean"]) * (
        (param_adim["V"] ** 2) * param_adim["rho"]
    )
    norme_vitesse_data = np.sqrt(u_data**2 + v_data**2)
    print("OK chargement de données")

    # On plot les vidéos
    plot_flow(
        x_data,
        y_data,
        t_data,
        p_data,
        p_pred,
        name_file="results/" + "/" + file_name + "/" + "result_pression.gif",
        title="Pression",
        fps=5,
    )
    plot_flow(
        x_data,
        y_data,
        t_data,
        norme_vitesse_data,
        norme_vitesse_pred,
        name_file="results/" + "/" + file_name + "/" + "result_vitesse.gif",
        title="Vitesse",
        fps=5,
    )
    print("OK les vidéos")

    # On plot la loss
    plot_loss(
        file_save="results/" + file_name,
        file="results/" + file_name + "/" + epoch,
        title_graph=title_loss,
    )
    print("OK le plot de la loss")

    # On plot la loss decompose
    plot_loss_decompose(
        file_save="results/" + file_name,
        file="results/" + file_name + "/" + epoch,
        title_graph="loss_decompose",
    )

    # On plot les points
    plot_points(
        X_train,
        X_border,
        mean_std,
        param_adim,
        file_save="results/" + file_name + "/points_coloc",
    )
    print("OK pour les points de coloc")

    # On plot le Cl
    plot_cl(
        X_full,
        U_full,
        model,
        param_adim,
        mean_std,
        file_save="results/" + file_name + "/courbe_cl",
    )
