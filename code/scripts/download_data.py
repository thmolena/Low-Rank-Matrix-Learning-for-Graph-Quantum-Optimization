#!/usr/bin/env python3
"""Download and authenticate every Stanford Gset source in the study."""

from hqml_trotter_scheduling.experiment import SOURCES, download_source


def main() -> None:
    for name in SOURCES:
        print(download_source(name))


if __name__ == "__main__":
    main()
