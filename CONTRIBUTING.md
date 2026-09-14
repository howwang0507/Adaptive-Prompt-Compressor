# Contributing to Adaptive Prompt Compressor

Thank you for your interest in contributing to **Adaptive Prompt Compressor**! We welcome contributions from researchers, engineers, and developers across the open-source and AI communities.

This guide provides guidelines and setup instructions to help you get started quickly and effectively.

---

## 🧭 Code of Conduct

All contributors and maintainers are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md). Please report any unacceptable behavior to `howwang0507@example.com`.

---

## 🛠️ Local Development Setup

We use [`uv`](https://github.com/astral-sh/uv) as the fast package and project manager.

### 1. Clone the Repository
```bash
git clone https://github.com/howwang0507/Adaptive-Prompt-Compressor.git
cd Adaptive-Prompt-Compressor
```

### 2. Install Dependencies
Install all core and development dependencies in an isolated virtual environment:
```bash
uv sync --all-extras --dev
```

### 3. Configure Environment Variables (Optional)
If running live LLM benchmarks against OpenAI or Anthropic:
```bash
cp .env.example .env
# Edit .env and supply your OPENAI_API_KEY
```
*(Note: Offline simulation tests do NOT require any external API keys.)*

---

## 🧪 Testing & Code Quality

Before opening a pull request, ensure all linters and test suites pass locally:

### 1. Linting & Formatting
```bash
# Check code style with Ruff
uv run ruff check .

# Optional formatting check
uv run ruff format --check .
```

### 2. Run Test Suite
```bash
uv run pytest tests/ -v
```

### 3. Run Cookbooks / Examples
Verify that the example scripts run without errors:
```bash
uv run python examples/basic_usage.py
uv run python examples/openai_sdk_wrapper_demo.py
uv run python examples/openai_tool_calling_compression.py
```

---

## 🌿 Branching & Git Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

* `feat(...)`: A new feature or capability (e.g. `feat(integrations): add LangChain prompt compressor adapter`)
* `fix(...)`: A bug fix (e.g. `fix(openai): handle empty tool call lists gracefully`)
* `docs(...)`: Documentation changes or additions (e.g. `docs(readme): add cookbook architecture diagram`)
* `test(...)`: Adding or refactoring unit/integration tests
* `perf(...)`: Performance optimizations
* `chore(...)`: Tooling, dependency, or packaging updates

---

## 📝 Pull Request Workflow

1. **Fork and Branch**: Create a feature branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. **Make Changes**: Keep commits atomic and self-contained.
3. **Write Tests**: Add unit tests in `tests/` covering new features or edge cases.
4. **Submit PR**: Open a PR pointing to `main`. Fill in the PR template describing your motivation, implementation details, and verification steps.
5. **Review**: Maintainers will review the code and CI status. Once approved, it will be merged into `main`!

---

## ⚖️ License

By contributing code to Adaptive Prompt Compressor, you agree that your contributions are licensed under the [MIT License](LICENSE).
