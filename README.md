# BIM-Master-Thesis

This repository contains all the source code to run the experiments, generate plots, as well as the LaTeX document source for the research project. This paper has been produced as part f the BMMTBIM BIM Master Thesis course at Erasmus University Rotterdam, The Netherlands.

**Author:** Codrin Socol _([744294cs@student.eur.nl](744294cs@student.eur.nl))_

**Supervisor:** Dr. Georgios Pierris 

**Co-Reader:** _To be determined_

**Master Programme:** Business Information Management

**University:** Rotterdam School of Management, Erasmus University Rotterdam

**Academic Year:** 2024-2025

# Table of Contents

TODO

# Getting Started

## Prerequisites

- Python 3.12
- `HFTBacktest` Python library
- pip - Python package manager

Required Packages can be installed using the following command:

```bash
pip install -r requirements.txt
```

## Data
This section provides details on how to obtain data required to run the backtesting agent and how to process it in order to be usable with the `HFTBacktest` library.
## Data source
This project requires cryptocurrency futures data from the [Deribit Exchange](https://www.deribit.com/futures/ETH-PERPETUAL), which can be purchased via [Tardis.dev](https://tardis.dev/deribit). Tardis.dev provides a variety of datasets for sale, but this project requires only two:
- ETH-PERPETUAL trades data _([trades](https://docs.tardis.dev/downloadable-csv-files#trades))_
- ETH-PERPETUAL orderbook data _([incremental_l2](https://docs.tardis.dev/downloadable-csv-files#incremental_book_l2))_

The data should be downloaded and placed in the `/data/daily-tardis` directory. This directory should include 2 subdirectories:
- `trades`: containing the trades data (one `csv.gz` file per day)
- `incremental_l2`: containing the orderbook data (one `csv.gz` file per day)

The file naming needs to adhere to the following format:
- `trades`: `deribit_trades_<date>_ETH-PERPETUAL.csv.gz`
- `incremental_l2`: `deribit_incremental_l2_<date>_ETH-PERPETUAL.csv.gz`
- _the date format should be `YYYY-MM-DD` (e.g. `2023-01-14`)_

## Data Preprocessing
The project uses utility functions from `HFTBacktest` library to convert Deribit data into `npz` file formats used by the backtesting library. The preprocessed data is persisted to files in the `/data/daily_processed` directory.
Moreover, daily end-of-day snapshots are generated and saved in the `/data/snapshots` directory. End-of-Day snapshots are used as starting points for the backtesting experiments. The snapshots are saved in `npz` format.

This research project focuses on ETH-Perpetual futures, with data available from January 2025. If the datasets exist in the `/data/daily-tardis` directory, with the formats mentioned above, the preprocessing stage can be executed by running the following command:

```bash
python3 preprocessing.py
```

Running this command will generate the following files:
- `/data/daily_processed/deribit_eth_perp_<date>.npz`
- `/data/snapshots/deribit_eth_perp_<date>_eod.npz`

These files can then directly be used to create a `HFTBacktest` agent instance.

## Alternative Data sources
The `HFTBacktest` library supports other data source, such as from Binance. This is out of scope for this research. However, if you want to test this agent with other data inputs, please follow the [Data Preparation tutorials](https://hftbacktest.readthedocs.io/en/latest/tutorials/Data%20Preparation.html) from the `HFTBacktest` library.