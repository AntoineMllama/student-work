import matplotlib.pyplot as plt
import numpy as np

from display import get_png_path


def plot_histogram(ax, times):
    ax.hist(times, bins=20, edgecolor="k", alpha=0.7)
    ax.set_title("Distribution des temps de séjour")
    ax.set_xlabel("Temps de séjour (s)")
    ax.set_ylabel("Fréquence")


def plot_waiting_times(ax, waiting_times):
    ax.hist(waiting_times, bins=20, edgecolor="k", alpha=0.7, color="orange")
    ax.set_title("Distribution des temps d'attente")
    ax.set_xlabel("Temps d'attente (s)")
    ax.set_ylabel("Fréquence")


def plot_server_utilization(ax, utilization_time, utilization):
    ax.plot(utilization_time, utilization, label="Utilisation")
    ax.axhline(y=1, color="r", linestyle="--", label="Saturation")
    ax.set_title("Taux d'utilisation des serveurs")
    ax.set_xlabel("Temps")
    ax.set_ylabel("Utilisation")
    ax.legend()
    ax.grid(True)


def plot_queue_length(ax, queue_time, queue_length):
    ax.plot(queue_time, queue_length, label="Longueur des files", color="green")
    ax.set_title("Longueur des files d'attente")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Longueur")
    ax.legend()
    ax.grid(True)


def plot_nb_arrivals_departure(ax, arrivals, departures):
    ax.plot(
        arrivals,
        range(len(arrivals)),
        label="Nombre des arrivées cumulées A(t)",
        color="blue",
    )
    ax.plot(
        departures,
        range(len(departures)),
        label="Nombre de départs cumulées D(t)",
        color="red",
    )
    ax.set_title("Nombre des arrivées et des départs depuis le début de la simulation")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Nombre de push")
    ax.legend()
    ax.grid(True)


def compute(arrivals, end_time):
    arrivals = np.floor(arrivals).astype(int)
    unique_arrivals, nb = np.unique(arrivals, return_counts=True)
    full_time = np.arange(1, end_time + 2)
    full_nb = np.zeros(len(full_time))
    full_nb[unique_arrivals] = nb
    return full_time, np.cumsum(full_nb)


def plot_mean_rate(ax, arrivals, departures, end_time):
    full_time, full_nb_arrivals = compute(arrivals, end_time)
    mean_rate_arrivals = full_nb_arrivals / full_time
    ax.plot(
        full_time,
        mean_rate_arrivals,
        label="Taux moyen des arrivées",
        color="green",
        linestyle="--",
    )
    full_time, full_nb_departures = compute(departures, end_time)
    mean_rate_departures = full_nb_departures / full_time
    ax.plot(
        full_time,
        mean_rate_departures,
        label="Taux moyen des départs",
        color="red",
        linestyle="--",
    )
    ax.set_title("Taux moyen des arrivées et des départs")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Nombre de push")
    ax.legend()
    ax.grid(True)


def plot_combined(history, simulation_name: str, description=None):
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    plot_histogram(axes[0, 0], history.get("spent_time", []))
    plot_waiting_times(axes[0, 1], history.get("test_waiting_time", []))
    plot_server_utilization(
        axes[1, 0],
        history.get("exec_utilization_time", []),
        history.get("exec_utilization", []),
    )
    plot_queue_length(
        axes[1, 1],
        history.get("exec_queue_length_time", []),
        history.get("exec_queue_length", []),
    )
    plot_nb_arrivals_departure(
        axes[2, 0],
        sorted(history.get("arrival_time", [])),
        sorted(history.get("departure_time", [])),
    )
    plot_mean_rate(
        axes[2, 1],
        sorted(history.get("arrival_time", [])),
        sorted(history.get("departure_time", [])),
        max(history.get("exec_utilization_time", [])),
    )
    fig.suptitle(
        f"Analyse des métriques de la simulation {simulation_name}",
        fontsize=16,
        fontweight="bold",
    )
    if description:
        plt.figtext(
            0.5,
            0.93,
            description,
            horizontalalignment="center",
            fontsize=12,
        )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    filepath = get_png_path(simulation_name)
    plt.savefig(filepath)
    plt.close()
