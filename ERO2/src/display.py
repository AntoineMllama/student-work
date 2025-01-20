import os
import webbrowser

from metrics import compute_metrics

_METRICS_DIR = "metrics"

_METRICS_PNG_DIR = f"{_METRICS_DIR}/png"
_METRICS_HTML_DIR = f"{_METRICS_DIR}/html"


def get_png_path(simulation_name: str):
    os.makedirs(_METRICS_PNG_DIR, exist_ok=True)
    return os.path.join(_METRICS_PNG_DIR, f"{simulation_name}_metrics.png")


def get_html_path(simulation_name: str):
    os.makedirs(_METRICS_HTML_DIR, exist_ok=True)
    return os.path.join(_METRICS_HTML_DIR, f"{simulation_name}_metrics.html")


def _generate_time_per_step(step_avg, step_max, step_min):
    steps = []
    for i, name in enumerate(["Exécution des tests", "Envoi des résultats"]):
        step = (
            f"<p>{name} - "
            f"Moyenne : {step_avg[i]:.2f} s, "
            f"Maximum : {step_max[i]:.2f} s, "
            f"Minimum : {step_min[i]:.2f} s</p>"
        )
        steps.append(step)

    return f"""
        <div class="metric-group">
            {''.join(steps)}
        </div>"""


def generate_metrics_html(history):
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

    metrics_html = f"""
    <div class="metrics-section">
        <h3>Résultats de la Simulation</h3>
        <div class="metric-group">
            <p>Temps de séjour moyen : {avg_time:.2f} unités de temps</p>
            <p>Temps de séjour maximum : {max_time:.2f} unités de temps</p>
            <p>Temps de séjour minimum : {min_time:.2f} unités de temps</p>
            <p>Nombre total de tags traités : {len(history['spent_time'])}</p>
        </div>

        <h3>Temps d'attente</h3>
        <div class="metric-group">
            <p>Temps d'attente moyen : {avg_wait:.2f} unités de temps</p>
            <p>Temps d'attente maximum : {max_wait:.2f} unités de temps</p>
            <p>Temps d'attente minimum : {min_wait:.2f} unités de temps</p>
        </div>

        <h3>Utilisation des serveurs</h3>
        <div class="metric-group">
            <p>Taux d'utilisation moyen : {avg_util:.2%}</p>
            <p>Taux d'utilisation maximum : {max_util:.2%}</p>
        </div>

        <h3>Longueur des files d'attente</h3>
        <div class="metric-group">
            <p>Longueur moyenne des files : {avg_queue:.2f}</p>
            <p>Longueur maximum des files : {max_queue:.2f}</p>
        </div>

        <h3>Temps par étape</h3>
        {_generate_time_per_step(step_avg, step_max, step_min)}
    </div>
    """
    return metrics_html


def generate_plot_html(simulation_name, metrics):
    png_path = get_png_path(simulation_name)
    html_path = get_html_path(simulation_name)
    relative_png_path = os.path.relpath(png_path, os.path.dirname(html_path))

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simulation: {simulation_name}</title>
    <link rel="stylesheet" href="../../css/styles.css">
</head>
<body>
    <div class="plot">
        <img src="{relative_png_path}" alt="img {simulation_name}">
        {metrics}
    </div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def _generate_dtale_links(links, type):
    links_html = "\n".join(
        f'<li><a href="{link}" target="_blank">'
        f'{"Submissions details" if i == 0 else "Resources metrics"} {type.lower()}'
        "</a></li>"
        for i, link in enumerate(links)
    )
    return f"""
                <div class="{type.lower()} views">
                    <strong>{type} Views:</strong>
                    <ul>
                        {links_html}
                    </ul>
                </div>
    """


def display_all_simulations(simulations_name, dtale_links, history):
    html_path = get_html_path("all_simulations")

    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>All Simulations</title>
    <link rel="stylesheet" href="../../css/styles.css">
</head>
<body>
    <h1>Simulations Results</h1>
        <div class="simulation">
            <h2>Simulation files</h2>
            <div class="links">
                <ul>
    """
    for simul_name in simulations_name:
        path = f"file://{os.path.abspath(get_html_path(simul_name))}"
        html_content += f"""
        <li><a href="{path}" target="_blank">{simul_name}</a></li>
    """
    html_content += """
                </ul>
            </div>
        </div>
    """
    for simul_name in simulations_name:
        file_html_path = f"file://{os.path.abspath(get_html_path(simul_name))}"
        png_path = get_png_path(simul_name)
        png_path = os.path.relpath(png_path, os.path.dirname(html_path))
        metrics = generate_metrics_html(history[simul_name].history)
        generate_plot_html(simul_name, metrics)
        html_content += f"""
    <div class="simulation">
        <h2>Simulation: <a href="{file_html_path}" target="_blank">{simul_name}</a></h2>
        """
        if dtale_links and simul_name in dtale_links:
            links = dtale_links[simul_name]
            html_content += f"""
        <div class="links">
            {_generate_dtale_links(links[0], "Data")}
            {_generate_dtale_links(links[1], "Chart")}
        </div>
        <img src="{png_path}" alt="img {simul_name}">
        {metrics}
    </div>
</body>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    webbrowser.open(f"file://{os.path.abspath(html_path)}")
