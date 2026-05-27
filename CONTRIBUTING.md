# Contributing to Adaptive Prompt Compressor

First off, thank you for considering contributing! It's people like you that make the open-source community such a great place to learn, inspire, and create.

## 🛠️ Development Setup

1. **Clone and Install**
   ```bash
   git clone https://github.com/howwang0507/Adaptive-Prompt-Compressor.git
   cd Adaptive-Prompt-Compressor
   uv sync --all-extras
   ```

2. **Environment Variables**
   Copy `.env.example` to `.env` and add your API keys.

3. **Coding Standards**
   - Use `ruff` for linting.
   - Use `black` for formatting.
   - Follow Type Hinting for all new functions.

## 🧪 Testing

Run tests before submitting a PR:
```bash
uv run pytest tests/
```

## 📝 Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters.
3. You may merge the Pull Request in once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you.

## ⚖️ License
By contributing, you agree that your contributions will be licensed under its MIT License.
