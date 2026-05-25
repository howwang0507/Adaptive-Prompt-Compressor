# Adaptive Prompt Compression via LinUCB 🧠

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://adaptive-prompt-compreappr-wanghao.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/howwang0507/Adaptive-Prompt-Compressor?style=social)](https://github.com/howwang0507/Adaptive-Prompt-Compressor/stargazers)

> **"Optimizing LLM efficiency where every token and every request counts. Learning adaptive strategies under extreme resource constraints."**

## 🌟 為什麼選擇這個專案？ (Why This Project?)
本專案專注於解決真實開發環境中的痛點：**API 配額限制與高昂成本**。
透過 **Contextual Bandit (LinUCB)**，系統不僅在優化 Token，更在挑戰 **極限樣本效率 (Sample Efficiency)**：
1. **資源受限學習**：專為每日請求數受限（如 Free-tier API）的環境設計。
2. **精準權衡**：在「成本節省」與「API 穩定性」之間尋找最優解。
3. **自適應路由**：自動識別高敏感內容（如 Code），防止因壓縮導致的系統崩潰。

---

## 📂 專案結構 (Project Structure)
本專案採用專業的模組化設計，結構如下：
- `src/`: 核心邏輯
  - `agent.py`: LinUCB 強化學習代理人實作。
  - `environment.py`: 真實 API 與離線模擬環境。
  - `utils.py`: 獎勵函數與資料處理工具。
- `tests/`: 單元測試案例，確保核心演算法之穩定性。
- `app.py`: Streamlit 互動式介面。

## 🧪 品質保證 (Engineering Quality)
- **單元測試**: 使用 `pytest` 確保 `LinUCB` 與 `Reward` 邏輯正確。
- **離線模擬**: 內建 `SimulatedEnvironment`，無需 API Key 即可觀察學習曲線。

## 🚀 快速啟動 (Quick Start)
*(此處保留原有的啟動指令)*

4. **開始實驗：** 程式啟動後，會在瀏覽器中開啟 Streamlit 介面。請在側邊欄輸入您的 API Key 並開始實驗。

## 實驗結果展示 (Results)
*(此處可放上 Streamlit 執行後的「平均獎勵收斂圖」和「自適應策略分佈圖」截圖。)*
- **收斂性分析**：LinUCB 能夠在少量的嘗試後快速收斂，找到平均獎勵較高的策略組合。
- **策略分佈**：觀察不同類別（Chat, Code, Translation等）所傾向選擇的 Arm，驗證系統是否具備對不同上下文的適應能力。
