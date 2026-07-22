---
license: other
language:
- ja
library_name: fireredtts2
pipeline_tag: text-to-speech
tags:
- text-to-speech
- dialogue
- japanese
- fireredtts2
- voice-cloning
base_model: FireRedTeam/FireRedTTS2
---

# Zoom1 Dialogue TTS model

> **PRIVATE / 取扱注意**: 本モデルは内部コーパスLLM-jp Zoom1でfine-tuneした派生物です。
> 学習データの公開同意・ライセンスは未整理のため、再配布不可・プロジェクト内研究用途限定です。
> 公開化は同意とライセンスの整理後に判断してください。

日本語の2話者対話を合成する
[FireRedTTS-2](https://huggingface.co/FireRedTeam/FireRedTTS2)を、LLM-jp Zoom1
（日本語2話者対話、約1,069時間）でposttrain fine-tuneしたモデルです。被せ相槌の扱いが
異なる`drop`と`keep`の2 variantを収録しています。

推論コードと利用手順は
[llm-jp/zoom1-dialogue-tts](https://github.com/llm-jp/zoom1-dialogue-tts)を参照してください。

## Variant

| Variant | 学習データ整形 | 最終step | 用途 |
|---|---|---:|---|
| `drop/` | 被せ相槌を除去した逐次データ | 3000（4 epoch） | 既定。長文・難所で比較的安定 |
| `keep/` | 相槌を独立ターンとして保持 | 4500（4 epoch） | 代替 |

### keep / drop評価

同一台本・同一声プールによる500ペアのCER評価です。

| Variant | 平均CER | 中央値 | P90 |
|---|---:|---:|---:|
| `drop` | 0.0942 | 0.0696 | 0.177 |
| `keep` | 0.1013 | 0.0676 | 0.223 |

統計的有意差は確認されていません。`drop`は情報密度の高い長文でworst-caseの裾が短いため、
推論ツールの既定値にしています。

## Files

```text
config_llm.json
config_codec.json
drop/llm_posttrain.pt
keep/llm_posttrain.pt
```

codecとQwen tokenizerは公式baseと同一なので収録していません。推論ツールが必要な資産だけを
`FireRedTeam/FireRedTTS2`から自動取得し、選択したvariantと結合します。

## Usage

CUDA GPUを搭載したLinux環境を想定しています。モデルリポジトリがprivateの間は、アクセス権の
あるHugging Faceアカウントでログインしてください。

```bash
git clone https://github.com/llm-jp/zoom1-dialogue-tts.git
cd zoom1-dialogue-tts
uv sync
uv run hf auth login

# 基本推論: Zoom1統計による間・重なり
uv run zoom1-dialogue-tts examples/dialogue.txt -o out/dialogue.wav
```

### 推奨: VAPによる文脈依存のターン交替

MaAI/VAPの`p_now`・`p_future`から通常ターンのSHIFT/HOLDを予測し、内容と韻律に応じて
間・食い込みを変えられます。FireRedTTS-2によるターン生成、VAP解析、FTO適用は1コマンドで
実行され、FireRedTTS-2の生成は1回だけです。

MaAIは依存関係が異なるため、初回だけ別環境を作成します。オフラインWAV解析ではマイク用の
PyAudioは不要です。

```bash
uv venv --python 3.12 .venv-vap
uv pip install --python .venv-vap/bin/python --no-deps maai
uv pip install --python .venv-vap/bin/python \
  torch torchaudio numpy soundfile librosa einops rich matplotlib scipy \
  transformers==5.5.3 huggingface-hub pygame

uv run zoom1-dialogue-tts examples/dialogue.txt -o out/vap_dialogue.wav \
  --turn-timing vap-auto \
  --vap-python .venv-vap/bin/python
```

出力は24 kHzステレオWAV（左=S1、右=S2）です。最終manifestに各境界のSHIFT/HOLD、要求FTO、
安全制約適用後のFTOを記録します。VAP解析用WAV、確率trace、境界JSONも同じ出力ディレクトリへ
保存されます。

聞き手の短い相槌には別のVAP backchannel予測JSONを指定できます。詳細なJSON形式、参照声の
指定、統計配置との比較手順は
[GitHub README](https://github.com/llm-jp/zoom1-dialogue-tts#readme)を参照してください。

## Training

- Base: FireRedTTS-2（Qwen 1.5B backbone + Qwen 200M decoder、12.5 Hz codec）
- Data: LLM-jp Zoom1をFireRedTTS-2形式へ変換
- Fine-tuning: posttrain、各4 epoch

## License and data governance

- Architecture and base implementation: FireRedTTS-2（Apache-2.0）
- Text tokenizer: Qwen2.5-1.5B（Apache-2.0）
- Fine-tuning data: LLM-jp Zoom1（内部データ）
- Model weights: `other`、再配布不可、プロジェクト内研究用途限定

このモデルをpublic化する前に、Zoom1参加者の同意範囲と派生モデル公開条件を確認してください。
