# Markowitz Portfolio Optimizer
# Usage: just <command>

app:
    uv run streamlit run app.py

lint:
    uv run ruff check .

clean:
    rm -rf .venv __pycache__
