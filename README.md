# CSV Analysis Agent

A local, single-turn CSV analysis application. Upload a CSV file and ask a question; the agent inspects the data, generates Python code, runs it in an isolated Docker container, and displays the answer, execution evidence, and any tables or charts.

## Requirements

- Python 3.12–3.14 and [uv](https://docs.astral.sh/uv/)
- A running local Docker Engine
- An API key for a compatible model service
- Network access to install dependencies and reach the model service

## Setup

Clone the repository and install the application:

```bash
git clone https://github.com/XixianWasTaken/csv-analysis-agent.git
cd csv-analysis-agent
uv sync --locked
docker build -t csv-agent-python:0.1 docker
cp .env.example .env
chmod 600 .env
```

Edit `.env` and set the model endpoint, model name, protocol and API key. The template uses Spark's Anthropic-compatible endpoint and reads the key from `SPARK_API_KEY`. Never commit `.env` or API keys.

## Run

```bash
uv run streamlit run app.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit. Upload `examples/sales.csv` and ask: **What are total sales?** The example total is **45**.

The command-line interface is also available:

```bash
uv run python -m csv_agent --csv examples/sales.csv --question 'What are total sales?'
```

Model requests consume provider credits. Use `uv run python -m csv_agent --smoke` to check the configured model connection without running a Docker container.

## How it works

The application profiles the CSV, requests analysis instructions from the configured model, and executes generated code in a container. The interface shows the answer, generated code, execution feedback, request history and reported model usage. Failed calculations may be repaired within the configured task budget.

The execution container runs without network access, with a read-only root filesystem and resource limits. The CSV column names, inferred types, missing-value counts, row count and first five sample rows are sent to the model provider; the full CSV is made available to the execution container. Review the provider's data handling terms before analyzing sensitive data.

A completed task means that code ran and produced a valid result; it does not guarantee that the answer is correct. Review the calculation and results before relying on them. CSV files must be UTF-8 encoded.

## Configuration

`.env.example` lists model and execution settings. `MODEL_PROTOCOL` accepts `anthropic` or `openai`; `MODEL_ENDPOINT` is the full API request URL. Execution settings include time, memory, CPU, process and output limits, as well as task-wide model request and execution budgets. Restart the application after changing the configuration.

## Checks

```bash
uv run mypy
uv run ruff check src tests app.py docker/runner.py
uv run pytest -q
```

The full test suite requires the Docker image and access to a local Docker daemon. To run tests that do not require Docker:

```bash
uv run pytest -q -m 'not docker'
```

## Repository layout

- `app.py` — Streamlit application
- `src/csv_agent/` — analysis, model, execution and evaluation code
- `docker/` — isolated execution image and runner
- `tests/` — automated checks
- `examples/` — sample CSV
- `benchmarks/` and `scripts/` — evaluation and reproducibility tools

## License

Project code is licensed under the [MIT License](LICENSE).

The DABench scorer in `src/csv_agent/_vendor/dabench.py` and its upstream snapshot in `benchmarks/dabench/eval_closed_form.py.upstream` come from [InfiAgent/InfiAgent](https://github.com/InfiAgent/InfiAgent) and retain the [Apache License 2.0](src/csv_agent/_vendor/LICENSE). Their pinned source revision and checksums are recorded in `benchmarks/dabench/manifest.json`.
