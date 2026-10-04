"""Run the Neumann training workflow with repository-local provenance."""
from pathlib import Path


def main(argv=None):
    from rl_tt_dgmres_neumann import experiment

    root = Path(__file__).resolve().parent
    provenance = root / "results" / "rl_tt_dgmres_neumann_d4_20261001" / "paper_provenance"
    sources = tuple(provenance / name for name in ("backend_cells.json", "ch05_setup.tex"))
    if not all(source.is_file() for source in sources):
        raise FileNotFoundError("Bundled Neumann construction provenance is missing.")
    experiment.PAPER_SOURCES = sources
    return experiment.main(argv)


if __name__ == "__main__":
    main()
