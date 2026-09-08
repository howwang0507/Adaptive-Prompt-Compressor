# Security Policy

The **Adaptive Prompt Compressor** team takes security, user privacy, and production robustness seriously. Because prompt compression sits as middleware between client applications and Large Language Model (LLM) APIs, ensuring safety against prompt injection, data leakage, and unsafe code execution is a first-class priority.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.1.x   | :white_check_mark: |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Security Architecture & Defenses

1. **AST Syntax Verification Sandbox**:
   When technical code prompts are compressed, downstream code is validated via Python's Abstract Syntax Tree parser (`ast.parse`) in an isolated check before being returned or dispatched to LLM providers. Unsafe syntax triggers immediate auto-fallback to preserve logic.

2. **Zero In-Memory Credential Persistence**:
   Provider API keys (OpenAI, Gemini, Anthropic) are read strictly at initialization from environment variables (`OPENAI_API_KEY`, etc.) and are never stored in disk databases, telemetry logs, or exported state vectors.

3. **Out-of-Distribution (OOD) Guardrails**:
   A built-in statistical feature guard monitors prompt vectors. When input features deviate significantly from historical distributions (potential adversarial prompt stuffing or jailbreak vectors), the system engages conservative fallback routing.

## Reporting a Vulnerability

If you discover a potential security vulnerability in this repository, please **do NOT report it via public GitHub issues**.

Instead, please send an email to:
- **Project Maintainer**: `howwang0507@example.com`

Please include:
- A description of the vulnerability
- Steps to reproduce or proof-of-concept payload
- Potential impact on client or downstream LLM workflows

We will acknowledge receipt within **48 hours** and provide a patch timeline. We appreciate responsible disclosure and will credit contributors in our release notes.
