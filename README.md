# SupeRR Selector Season 2 — Pressure Analysis

This repository contains the data engineering and analytical pipeline for the Rajasthan Royals "SupeRR Selector Season 2" hackathon.

## Project Objective

To build a data-driven system that measures how batters and bowlers perform in high-pressure match situations compared to their usual baseline performance, ultimately identifying the Top 10 most reliable players in each category.

## Methodology & Structural Plan

This project strictly adheres to a consensus-driven Master Structural Plan. Key features include:

1. **Binary Pressure Flags:** Asymmetrical definitions for chasing and setting pressure.
2. **Reconstructed Match State:** Ball-by-ball computation of RRR, balls remaining, and cumulative scores.
3. **Strict Baseline Comparison:** A player's baseline is calculated exclusively from non-pressure deliveries.
4. **Bayesian Shrinkage:** Handling small sample sizes statistically rather than through arbitrary cutoffs.

## Setup Instructions

1. Clone this repository.
2. Ensure you have Python 3.10+ installed.
3. Create a virtual environment: `python -m venv venv`
4. Activate the virtual environment.
5. Install dependencies: `pip install -r requirements.txt`
6. Place the official Unstop CSV data files into the `data/raw/` directory.

_Note: Raw and processed data files are explicitly ignored in version control via `.gitignore`._
