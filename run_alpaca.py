#!/usr/bin/env python
"""
ALPACA: *A*utomated *L*ens-modelling *P*ipeline for *A*ccelerated TD*C*osmography *A*nalysis

Run script - executes the pipeline with settings from alpaca_config.py

Usage:
    python run_alpaca.py

To customize settings, edit alpaca_config.py
"""

import datetime

# Import configuration
import alpaca_config as cfg

# Import pipeline and output modules
from alpaca.pipeline import run_pipeline
from alpaca.output import print_full_summary


def main():
    """Run the ALPACA pipeline."""
    # Print timestamp
    dateandtime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(f"Current date and time: {dateandtime}")

    # Print configuration summary
    cfg.print_config_summary()

    # Build configuration object
    config = cfg.build_config()
    print("Configuration created.")

    # Run the pipeline
    results = run_pipeline(
        config=config,
        run_sampling=cfg.RUN_SAMPLING,
        sampler=cfg.SAMPLER,
        verbose=True,
    )

    # Print full results summary
    print_full_summary(
        results=results,
        sampler=cfg.SAMPLER,
        config=config,
        verbose=True,
    )

    return results


if __name__ == "__main__":
    main()
