import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def get_timing_results():
    results = {}

    # for epsilon in [0.01]:
    for epsilon in [0.01, 0.001, 0.0001]:
        filename = f"results/results_{epsilon}_[2, 3, 4, 5, 6]_2_3_days.json"

        with open(filename) as f:
            data = json.load(f)
            results[epsilon] = {"E2E": [], "GRB": []}

            for num_digits in ["2", "3", "4", "5", "6"]:
                if epsilon == 0.01 and (num_digits == "5" or num_digits == "6"):
                    continue
                e2e_times = [log["runtime"] for log in data[num_digits]["E2E"].values()]
                grb_times = [log["runtime"] for log in data[num_digits]["GRB"].values()]

                results[epsilon]["E2E"].append(sum(e2e_times) / len(e2e_times))
                results[epsilon]["GRB"].append(sum(grb_times) / len(grb_times))

    return results


def plot_timing_results(timing_results, marabou_res):
    fig, ax = plt.subplots()
    # for epsilon in [0.01]:
    for epsilon in [0.01, 0.001, 0.0001]:
        x = [2, 3, 4] if epsilon == 0.01 else [2, 3, 4, 5, 6]
        ax.plot(
            x,
            timing_results[epsilon]["E2E"],
            label=f"E2E-A  (ε={epsilon})",
            linewidth=2,
            marker=(5, 0),
        )
    # for epsilon in [0.01]:
    for epsilon in [0.01, 0.001, 0.0001]:
        x = [2, 3, 4] if epsilon == 0.01 else [2, 3, 4, 5, 6]
        ax.plot(
            x,
            timing_results[epsilon]["GRB"],
            label=f"A+SLV (ε={epsilon})",
            linewidth=2,
            linestyle="dotted",
            marker="^",
        )

    marabou_solve_times = [x["solve_time"] for x in marabou_res["0.001"].values()]
    avg_solve_time = sum(marabou_solve_times) / len(marabou_solve_times)
    print("Marabou average solve time (ε=0.001):", round(avg_solve_time, 2))
    # ax.axline(xy1=(2, avg_solve_time), xy2=(6, avg_solve_time), label="Baseline 2 - CNN only", color="#57a0d3", linewidth=2, linestyle="dashed")

    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Number of MNIST digits")
    ax.set_ylabel("Time (log(s))")
    ax.legend()
    ax.grid(which="both")
    fig.tight_layout()
    plt.show()


def get_bound_results():
    filename = f"results/results_0.001_[2, 3, 4, 5, 6]_2_3_days.json"
    with open(filename) as f:
        data = json.load(f)
        results = {}
        for num_digits in ["2", "3", "4", "5"]:
            e2e_lb = [s["lower_bound"] for s in data[num_digits]["E2E"].values()]
            e2e_ub = [s["upper_bound"] for s in data[num_digits]["E2E"].values()]
            e2e_robust = [s["robust"] for s in data[num_digits]["E2E"].values()]
            grb_lb = [s["lower_bound"] for s in data[num_digits]["GRB"].values()]
            grb_ub = [s["upper_bound"] for s in data[num_digits]["GRB"].values()]
            grb_robust = [s["robust"] for s in data[num_digits]["GRB"].values()]

            results[num_digits] = {
                "avg_lower_e2e": sum(e2e_lb) / len(e2e_lb),
                "avg_upper_e2e": sum(e2e_ub) / len(e2e_ub),
                "robust_e2e": sum(e2e_robust) / len(e2e_robust),
                "avg_lower_grb": sum(grb_lb) / len(grb_lb),
                "avg_upper_grb": sum(grb_ub) / len(grb_ub),
                "robust_grb": sum(grb_robust) / len(grb_robust),
            }

    return results


def print_bound_results(bound_results):
    for digits, stats in bound_results.items():
        print(digits, end="   -   ")
        print(
            f"E2E: {stats['avg_lower_e2e']:.3f}-{stats['avg_upper_e2e']:.3f}, {round(stats['robust_e2e'] * 100, 2)}",
            end="   ",
        )
        print(
            f"GRB: {stats['avg_lower_grb']:.3f}-{stats['avg_upper_grb']:.3f}, {round(stats['robust_grb'] * 100, 2)}"
        )


if __name__ == "__main__":
    with open("results/marabou_mnist.json", "r") as f:
        marabou_res = json.load(f)

    timing_results = get_timing_results()
    plot_timing_results(timing_results, marabou_res)

    bound_results = get_bound_results()
    print_bound_results(bound_results)
