# Zoom1 Dialogue TTS

[Demo & audio samples](https://llm-jp.github.io/zoom1-dialogue-tts/) | [Hugging Face model](https://huggingface.co/llm-jp/zoom1-dialogue-tts)

FireRedTTS-2をLLM-jp Zoom1の日本語対話でfine-tuneし、2話者の自然なステレオ対話音声を
生成する推論ツールです。左チャンネルがS1、右チャンネルがS2です。話者交代時の間・発話の
重なりと、聞き手の相槌をZoom1で観測した統計に基づいて付与します。

- Model: [llm-jp/zoom1-dialogue-tts](https://huggingface.co/llm-jp/zoom1-dialogue-tts)
- Base implementation: [FireRedTeam/FireRedTTS2](https://github.com/FireRedTeam/FireRedTTS2)
- Output: 24 kHz, 2-channel WAV (left=S1, right=S2)

> [!IMPORTANT]
> 現在、モデルリポジトリはprivateです。初回実行前にアクセス権のあるHugging Face
> アカウントで `hf auth login` を実行してください。公開時にはこの注意を更新します。

アクセス権のある共同研究者向けに、Zoom1収録話者の許諾範囲内で実音声をvoice-clone
プロンプトとして用いた比較サンプルを`private_samples/`へ収録しています。このサンプルは
公開GitHub Pagesには含まれません。利用条件は同ディレクトリのREADMEを確認してください。

## セットアップ

CUDA GPUを搭載したLinux環境を想定しています。`uv`を使うとPythonとCUDA版PyTorchを
含む環境を再現できます。

```bash
git clone https://github.com/llm-jp/zoom1-dialogue-tts.git
cd zoom1-dialogue-tts
uv sync --extra test
uv run hf auth login  # モデルがprivateの間だけ
```

初回推論時に、Zoom1 fine-tuneと公式FireRedTTS-2のcodec/tokenizerをHugging Faceから
自動取得します。`drop`と`keep`のうち、指定した一方のチェックポイントだけを取得します。
モデルと公式baseは検証済みrevisionに固定され、不要なbase LLM重みは取得しません。

## クイックスタート

入力は1行1ターンです。

```text
[S1]お疲れさまです。この前の連休、温泉に行ってきたんですよ。
[S2]いいですね。どのあたりに行ったんですか。
```

```bash
uv run zoom1-dialogue-tts examples/dialogue.txt -o out/dialogue.wav
```

生成物:

- `out/dialogue.wav`: Zoom1風の間・重なり・相槌を含むステレオ音声
- `out/dialogue.manifest.json`: 各ターンの話者、本文、開始時刻、長さ、相槌時刻
- `out/dialogue_turns/`: 後処理前のターン別音声

## 声を指定する

収録許諾のある参照音声と、その正確な書き起こしを話者ごとに指定できます。3〜10秒程度の
単一話者・無雑音音声を推奨します。

```bash
uv run zoom1-dialogue-tts examples/dialogue.txt -o out/dialogue.wav \
  --prompt-s1 voice_s1.wav "参照音声の書き起こし" \
  --prompt-s2 voice_s2.wav "参照音声の書き起こし"
```

参照音声を省略した場合は、モデルが生成する話者音色を使います。相槌は各話者の最初の生成
ターンを声の参照として再合成するため、聞き手と同じ音色になります。

## 相槌タイミング

既定の`stat`モードは、Zoom1の相槌頻度（約3.1回/分）に基づいて長い発話へ相槌を配置します。

```bash
# 既定
uv run zoom1-dialogue-tts examples/dialogue.txt -o out/stat.wav \
  --backchannels stat --bc-per-minute 3.1

# 相槌なし
uv run zoom1-dialogue-tts examples/dialogue.txt -o out/no_bc.wav --backchannels none
```

VAPで検出した時刻を使う場合は、`[{"time": 2.2, "listener_channel": 1, "score": 0.65}]`
形式のJSONを渡します。`time`は後処理前の音声上の秒、チャンネルは0=S1、1=S2です。

```bash
uv run zoom1-dialogue-tts examples/dialogue.txt -o out/vap.wav \
  --backchannels vap --vap-json vap_points.json
```

VAP検出器はMaAIとPyAudioの環境制約があるため、本パッケージの標準依存には含めていません。
まず統計配置で利用でき、VAP環境がある場合だけ予測JSONを差し替えられる構成です。

## モデルvariant

```bash
# 推奨: 被せ相槌を除いたZoom1でfine-tune。発話が比較的明瞭
uv run zoom1-dialogue-tts examples/dialogue.txt --variant drop -o out/drop.wav

# 学習時に短い相槌ターンを保持
uv run zoom1-dialogue-tts examples/dialogue.txt --variant keep -o out/keep.wav
```

## 入力形式

`.txt`、`.json`、`.jsonl`に対応します。話者はS1とS2のみです。

```json
{
  "turns": [
    {"speaker": "S1", "text": "こんにちは。"},
    {"speaker": "S2", "text": "こんにちは。"}
  ]
}
```

## 仕組み

1. FireRedTTS-2 Zoom1 fine-tuneで各ターンを文脈付き生成
2. ターンをS1=左、S2=右へ分離
3. Zoom1由来のFTO分布から、話者交代ごとの間または重なりをサンプリング
4. 統計配置またはVAP指定時刻に、聞き手と同じ声の相槌を挿入
5. ステレオWAVと再現用manifestを保存

FireRedTTS-2の学習コード、デモ、Docker資産は同梱しません。公式リポジトリの固定commitから
推論に必要な`fireredtts2`パッケージ（約144 KB）だけを収録し、NOTICEに由来と変更点を
記載しています。それ以外は推論ラッパーとZoom1風タイミング処理です。

## 制限事項

- FireRedTTS-2は逐次的にターンを生成します。重なりと相槌は生成後の時間編集であり、
  full-duplexモデルのように両話者を同時生成しているわけではありません。
- 「うん」「はい」以外のおうむ返しや意味のある割り込み内容は、入力台本に含める必要があります。
- VAPは相槌時刻にのみ利用します。通常ターンの食い込みはZoom1統計から決まります。
- 参照声には、本人の同意と利用許諾がある音声だけを使用してください。
- 約3分を超える長い台本ではFireRedTTS-2のコンテキスト上限に達する可能性があります。

## 開発

```bash
uv run pytest
```

テストはモデルをダウンロードせず、台本解析、モデル構成、タイミング処理を検証します。

## ライセンス

コードはApache License 2.0です。FireRedTTS-2もApache License 2.0で公開されています。
モデルおよび参照音声については、それぞれのHugging Face model cardと音源の利用条件を確認してください。

## 謝辞

- FireRedTeam, FireRedTTS-2
- LLM-jp Zoom1 dataset and Dialogue Working Group
- MaAI/VAP backchannel model (optional timing detector)
