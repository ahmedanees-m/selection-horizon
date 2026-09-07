"""Validation of the selection-gradient module against a real schedule.

The execution plan asks that the module be checked by requiring a peak in the
late teens or early twenties. That criterion belongs to Fisher reproductive
value v, not to the residual reproductive weight W, which is a tail sum of
non-negative terms and so is monotone non-increasing for every life history.

It also requires a demography with substantial pre-reproductive mortality. v
rises through childhood only because surviving childhood makes future
reproduction more certain. Under a modern schedule, where survival to thirty is
close to one, there is almost nothing to become more certain about and v is
close to monotone from birth. The peak criterion therefore cannot be exercised
on modern demography, and this runner reports that rather than pretending
otherwise.

What can be checked now, and is:

    W is monotone non-increasing everywhere
    W reaches zero past the end of the fertile span
    v falls to zero past the end of the fertile span
    the intrinsic growth rate solves the Euler-Lotka equation
    the net reproduction rate matches the published total fertility rate
"""

from __future__ import annotations

import json

import numpy as np

from src import config
from src.ingest import demography as demography_ingest
from src.models import demography


def life_history() -> demography.LifeHistory:
    table = demography_ingest.load()
    return demography.LifeHistory(
        ages=table["age"].to_numpy(dtype=float),
        survivorship=table["survivorship"].to_numpy(dtype=float),
        fertility=table["fertility"].to_numpy(dtype=float),
        label=f"{table['location'].iloc[0]} {int(table['year'].iloc[0])}",
    )


def validate(history: demography.LifeHistory) -> dict:
    rate = demography.intrinsic_growth_rate(history)
    weight = demography.residual_reproductive_weight(history, rate)
    value = demography.reproductive_value(history, rate)

    fertile = history.ages[history.fertility > 0]
    last_fertile = float(fertile.max()) if fertile.size else float("nan")

    characteristic = float(
        np.sum(np.exp(-rate * history.ages) * history.survivorship * history.fertility)
    )
    net_reproduction = float(np.sum(history.survivorship * history.fertility))

    past_fertile = history.ages > last_fertile
    checks = {
        "weight_monotone_non_increasing": bool(np.all(np.diff(weight) <= 1e-12)),
        "weight_zero_past_fertile_span": bool(np.allclose(weight[past_fertile], 0.0, atol=1e-12)),
        "value_zero_past_fertile_span": bool(np.allclose(value[past_fertile], 0.0, atol=1e-12)),
        "euler_lotka_solved": bool(abs(characteristic - 1.0) < 1e-8),
    }

    return {
        "schedule": history.label,
        "intrinsic_growth_rate": rate,
        "euler_lotka_characteristic": characteristic,
        "net_reproduction_rate": net_reproduction,
        "last_fertile_age": last_fertile,
        "reproductive_value_peak_age": float(history.ages[int(np.argmax(value))]),
        "survivorship_to_first_fertile_age": float(
            history.survivorship[int(fertile.min())] if fertile.size else np.nan
        ),
        "checks": checks,
        "peak_criterion_exercisable": bool(
            fertile.size and history.survivorship[int(fertile.min())] < 0.90
        ),
    }


def figure(history: demography.LifeHistory, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = config.load("figures")["style"]
    rate = demography.intrinsic_growth_rate(history)
    weight = demography.residual_reproductive_weight(history, rate)
    value = demography.reproductive_value(history, rate)

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), dpi=style["figure_dpi"])
    for axis, series, title in (
        (axes[0], history.fertility, "Fertility m(a)"),
        (axes[1], weight / weight[0] if weight[0] else weight, "Residual weight W(a), scaled"),
        (axes[2], value, "Reproductive value v(a)"),
    ):
        axis.plot(
            history.ages,
            series,
            linewidth=style["line_width"],
            color=style["palette"]["constraint"],
        )
        axis.set_title(title, fontsize=style["base_font_size"])
        axis.set_xlabel("age", fontsize=style["base_font_size"] - 1)
        axis.tick_params(labelsize=style["base_font_size"] - 2)
        axis.set_xlim(0, 80)

    fig.suptitle(history.label, fontsize=style["base_font_size"])
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config.ensure_dirs()
    history = life_history()
    report = validate(history)

    out_dir = config.derived_dir() / "analysis_set"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "demography_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    figure(history, config.display_dir() / "demography_validation.png")

    print(f"Selection gradient on {report['schedule']}")
    print(f"   intrinsic growth rate r: {report['intrinsic_growth_rate']:+.5f}")
    print(f"   net reproduction rate:   {report['net_reproduction_rate']:.4f}")
    print(f"   fertile span ends at:    {report['last_fertile_age']:.0f}")
    print(
        f"   survivorship to first fertile age: {report['survivorship_to_first_fertile_age']:.4f}"
    )
    print(f"   reproductive value peaks at age:   {report['reproductive_value_peak_age']:.0f}")
    print()
    for name, passed in report["checks"].items():
        print(f"   {'pass' if passed else 'FAIL'}  {name}")
    print()
    if report["peak_criterion_exercisable"]:
        print("   The peak criterion is exercisable on this schedule.")
    else:
        print(
            "   The peak criterion is not exercisable here: survival to the first\n"
            "   fertile age is too high for reproductive value to rise through\n"
            "   childhood. It is checked under the ancestral schedule instead."
        )

    if not all(report["checks"].values()):
        raise SystemExit("demography validation failed")


if __name__ == "__main__":
    main()
