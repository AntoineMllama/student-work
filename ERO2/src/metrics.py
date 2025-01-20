import numpy as np


def compute_metrics(data):
    if len(data) == 0:
        return 0, 0, 0
    return np.mean(data), np.max(data), np.min(data)


def print_metrics(history):
    avg_time, max_time, min_time = compute_metrics(history.get("spent_time", []))
    avg_wait, max_wait, min_wait = compute_metrics(history.get("test_waiting_time", []))
    avg_util, max_util, _ = compute_metrics(history.get("exec_utilization", []))
    avg_queue, max_queue, _ = compute_metrics(history.get("exec_queue_length", []))

    exec_step_avg, exec_step_max, exec_step_min = compute_metrics(
        history.get("exec_step_time", [])
    )
    send_step_avg, send_step_max, send_step_min = compute_metrics(
        history.get("send_step_time", [])
    )
    step_avg = (exec_step_avg, send_step_avg)
    step_max = (exec_step_max, send_step_max)
    step_min = (exec_step_min, send_step_min)

    print("\n--- Résultats de la Simulation ---")
    print(f"Temps de séjour moyen : {avg_time:.2f} unités de temps")
    print(f"Temps de séjour maximum : {max_time:.2f} unités de temps")
    print(f"Temps de séjour minimum : {min_time:.2f} unités de temps")
    print(f"Nombre total de tags traités : {len(history['spent_time'])}")

    print("\n--- Temps d'attente ---")
    print(f"Temps d'attente moyen : {avg_wait:.2f} unités de temps")
    print(f"Temps d'attente maximum : {max_wait:.2f} unités de temps")
    print(f"Temps d'attente minimum : {min_wait:.2f} unités de temps")

    print("\n--- Utilisation des serveurs ---")
    print(f"Taux d'utilisation moyen : {avg_util:.2%}")
    print(f"Taux d'utilisation maximum : {max_util:.2%}")

    print("\n--- Longueur des files d'attente ---")
    print(f"Longueur moyenne des files : {avg_queue:.2f}")
    print(f"Longueur maximum des files : {max_queue:.2f}")

    print("\n--- Temps par étape ---")
    step_names = ["Exécution des tests", "Envoi des résultats"]
    for i, name in enumerate(step_names):
        print(
            f"{name} - Moyenne : {step_avg[i]:.2f}s, Maximum : {step_max[i]:.2f}s,"
            f" Minimum : {step_min[i]:.2f}s"
        )
    print("------------------------------------\n")
