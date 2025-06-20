# Medical_ChatBot
AI-powered multilingual voice assistant using LLaMA2, speech recognition, and TTS

# 🗣️ Multilingual AI Voice Assistant Using LLaMA2

An advanced AI-powered multilingual voice assistant system that integrates real-time speech recognition, language translation, fine-tuned transformer-based response generation, and natural speech synthesis — all within a modular and optimized architecture. This project demonstrates the convergence of state-of-the-art NLP, speech processing, and multilingual AI to deliver an interactive, scalable, and intelligent voice assistant solution.

---

## 🔎 Overview

This assistant enables users to engage in voice-based conversations across multiple languages. By leveraging a fine-tuned **LLaMA2 model**, the assistant understands context-rich prompts and generates accurate responses. It supports voice or text-based input, multilingual translation via the **Google Translate API**, and real-time audio output using **Google Text-to-Speech (gTTS)**.

The system is fine-tuned using parameter-efficient techniques (PEFT with LoRA) on a domain-specific dataset (medical knowledge) and deployed with **4-bit quantization** to ensure optimized memory consumption without sacrificing performance.

---

## 🧠 Core Features

- 🎙️ **Voice and Manual Input**: Accepts both microphone input and manually typed prompts.
- 🌐 **Multilingual Understanding**: Supports multiple input/output languages (e.g., English, Hindi, Telugu, Tamil, Bengali).
- 🤖 **Domain-Aware AI Responses**: Fine-tuned LLaMA2 model generates high-quality, context-aware answers.
- 🔁 **Real-Time Translation**: Uses Google Translate API for translating prompts and responses.
- 🗣️ **Text-to-Speech Output**: Delivers AI responses in natural speech using gTTS.
- ⚙️ **Model Efficiency**: Loads LLaMA2 in 4-bit mode using `BitsAndBytes` for optimized GPU/RAM usage.
- 🧪 **Medical Use Case Focus**: Trained on a curated dataset of medical terminology to simulate real-world domain-specific conversations.
- 📊 **Custom Fine-Tuning Pipeline**: Built with Hugging Face `transformers`, `trl`, and `peft` for modular fine-tuning.

---

## 🛠️ Tech Stack

| Component              | Technology / Library                                              |
|------------------------|-------------------------------------------------------------------|
| Model Architecture     | [LLaMA2](https://huggingface.co/models) (fine-tuned via PEFT)     |
| Fine-Tuning Framework  | `transformers`, `trl`, `peft`, `datasets`                        |
| Quantization           | `BitsAndBytes` (4-bit nf4 with float16 compute)                  |
| Speech Recognition     | `SpeechRecognition`, `PyDub`, `FFmpeg`                           |
| Translation API        | `googletrans==4.0.0-rc1`                                          |
| Text-to-Speech (TTS)   | `gTTS` (Google Text-to-Speech)                                    |
| Deployment             | Google Colab / Jupyter Notebook                                   |
| Inference Pipeline     | Hugging Face `pipeline("text-generation")`                       |

---

## 📚 Model Information

- **Base Model**: [`abneraigc/llama2finetune_demo`](https://huggingface.co/abneraigc/llama2finetune_demo)
- **Dataset**: `aboonaji/wiki_medical_terms_llam2_format`
- **Fine-Tuning**: LoRA config with `r=64`, `alpha=16`, `dropout=0.1`, trained via `SFTTrainer`
- **Quantization**: 4-bit NF4 via `BitsAndBytesConfig`
- **Tokenizer**: AutoTokenizer with EOS padding enabled

---

## 🚀 Setup Instructions

### ✅ Prerequisites

- Python 3.10+
- Google Colab or local Jupyter environment
- `ffmpeg` (required for audio conversion)
- GPU with 12GB+ VRAM (for efficient model inference)

### 🔧 Installation

```bash
git clone https://github.com/yourusername/multilingual-llama-voice-assistant.git
cd multilingual-llama-voice-assistant
pip install -r requirements.txt

