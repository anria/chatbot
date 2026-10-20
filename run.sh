curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
uv add python-dotenv
uv add openai
uv add rank_bm25
uv add sentence-transformers
uv add numpy
uv add pandas
uv run main.py
