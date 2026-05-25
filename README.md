# Adaptive Prompt Compression via LinUCB 🧠

## 研究動機 (Motivation)
隨著大型語言模型 (LLM) 的普及，API 呼叫成本與 Token 數量的限制成為實務應用上的挑戰。不同類型的文本（如程式碼、日常對話）對壓縮的容忍度不同。本專案旨在開發一種能夠根據文本特徵自動選擇最佳壓縮策略的系統，以在保證回答品質的同時，最大化減少 API 使用成本。

## 系統架構 (Architecture)
本專案採用 **Contextual Bandit (LinUCB)** 演算法進行動態策略選擇。系統定義了三種壓縮策略 (Arms)：
- **Arm 0**: 原文發送 (高品質、高成本，適用於對結構敏感的文本如程式碼)
- **Arm 1**: 基本去空白 (低風險壓縮，移除多餘空格與換行)
- **Arm 2**: 激進去停用詞 (高風險、高壓縮比，移除停用詞但保留標點以維持語義)

系統會即時提取輸入文本的特徵（長度、詞彙多樣性、代碼特徵等），並透過 LinUCB 演算法預測各策略的回報率，選擇最適合的壓縮方式。

## 快速啟動 (Quick Start)
請確保您的環境中已安裝 Python。

1. **複製專案：** (如果是從 GitHub clone)
   ```bash
   git clone <你的_github_repo_url>
   cd Adaptive-Prompt-Compressor
   ```

2. **取得 Google Gemini API Key：** 您需要一組 Gemini API Key 來執行實驗。

3. **一鍵啟動：**
   ```bash
   ./run.sh
   ```
   (若權限不足，請先執行 `chmod +x run.sh`)

4. **開始實驗：** 程式啟動後，會在瀏覽器中開啟 Streamlit 介面。請在側邊欄輸入您的 API Key 並開始實驗。

## 實驗結果展示 (Results)
*(此處可放上 Streamlit 執行後的「平均獎勵收斂圖」和「自適應策略分佈圖」截圖。)*
- **收斂性分析**：LinUCB 能夠在少量的嘗試後快速收斂，找到平均獎勵較高的策略組合。
- **策略分佈**：觀察不同類別（Chat, Code, Translation等）所傾向選擇的 Arm，驗證系統是否具備對不同上下文的適應能力。
