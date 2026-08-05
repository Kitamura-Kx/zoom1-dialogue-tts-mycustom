# VAPターン境界と台本内相槌の固定FTO

この文書は、通常ターンをMaAI VAPで配置し、台本に最初から含まれる短い相槌ターンだけを
固定FTOで軽く重ねる運用の正本です。ここでいう「相槌」は、新しい発話を後処理で追加する
機能ではありません。

## 採用する構成

```bash
uv run zoom1-dialogue-tts input.txt -o out/dialogue.wav \
  --turn-timing vap-auto \
  --vap-python .venv-vap/bin/python \
  --backchannel-turn-overlap-ms 200 \
  --backchannels none
```

- 通常ターン: MaAI VAPのSHIFT/HOLDから間または重なりを決める
- 台本内の短い相槌ターン: SHIFT/HOLDを使わず、要求FTOを`-200 ms`に固定する
- 新規の相槌音声: 追加しない（`--backchannels none`）
- 相槌配置後のVAP再解析: 行わない

`--backchannel-turn-overlap-ms`は正のミリ秒値で指定します。内部では負のFTOへ変換されるため、
`200`を指定すると要求FTOは`-200 ms`になります。このオプションを省略した場合、相槌を含む
すべてのターン境界に従来のVAP SHIFT/HOLD判定を使います。

## 用語の区別

本ツールには、名前が似た2種類の相槌処理があります。

| 種類 | 入力台本に発話があるか | 配置方法 |
|---|---|---|
| 台本内相槌ターン | ある（例: `[S2]うん`） | `--backchannel-turn-overlap-ms`で境界FTOを固定できる |
| 後処理による追加相槌 | ない | `--backchannels stat`または`vap`で新しい相槌クリップを追加する |

今回採用する構成は前者だけを使用します。後者を確実に無効化するため、コマンドには
`--backchannels none`を明示してください。`--backchannels`の既定値は`stat`なので、省略すると
統計的な追加相槌が有効になります。

## 処理順序

1. FireRedTTS-2が台本を上から順に、各ターンをモノラル音声として1回生成する。
2. ターン音声を無音・重なりなしで連結し、24 kHzステレオの`*.vap_input.wav`を作る。
3. MaAI用に16 kHzへ変換し、10 Hz（100 msごと）で`p_now`と`p_future`を得る。
4. 通常ターン境界はVAPのSHIFT/HOLDルールでFTOを予測する。
5. 相槌ターン境界はVAP予測を上書きし、固定の負FTOを要求する。
6. 安全制約を適用して同じターン波形を再配置し、最終ステレオWAVを作る。

固定FTOを適用した最終音声をMaAIへ再入力する反復処理は行いません。後続の通常境界も、手順2の
重なりなし音声から得た同じVAP traceに基づきます。この非反復方式を本プロジェクトの現在の
運用として採用します。

## 通常ターンのVAP判定

各チャンネルのフロアスコアは次の式です。

```text
floor_score = 0.65 * p_now + 0.35 * p_future
```

各境界では直前ターン末尾の800 ms前から100 ms前までを調べます。その窓内で次話者と前話者の
スコア差が最大になるフレームを求め、差が`0.15`以上ならSHIFT、未満ならHOLDです。

- SHIFT: 次話者が優位になった時刻から負のFTOを作る（VAP側の最大要求重なり500 ms）
- HOLD: 前話者の保持度から80〜1200 msの正のFTOを作る
- VAP予測がない境界: Zoom1統計へフォールバックする

## 相槌ターンの認識条件

次の両方を満たすターンを相槌として認識します。

1. 句読点と前後空白を除いた本文が次のいずれかに完全一致する。

   ```text
   うん / はい / ええ / ああ / へえ / うんうん / そうですね / なるほど
   ```

2. FireRedTTS-2が生成したターン音声の長さが1.2秒以下である。

部分一致は使いません。例えば「はい。きっと大丈夫です」は相槌ターンになりません。一方、
「そうですね。」は句点を除去すると一致するため、1.2秒以下なら相槌として扱われます。独立した
返答として扱いたい語がある場合は、`SHORT_BACKCHANNELS`の語彙を見直す必要があります。

## 固定FTOと安全制約

`--backchannel-turn-overlap-ms 200`の場合、相槌ターンの要求値は常に次です。

```text
requested_offset_ms = -200.0
```

実適用値は通常の安全制約で制限されます。

```text
最大重なり = min(800 ms, 前後の短い方のターン長 * 0.5)
offset_ms = max(requested_offset_ms, -最大重なり)
```

例として、320 msの「うん」に対する最大重なりは160 msなので、要求`-200 ms`に対する実適用は
`-160 ms`です。400 ms以上の相槌は、直前ターンも十分長ければ`-200 ms`がそのまま適用されます。

## Manifestと診断ファイル

相槌特別扱いの境界は最終manifestで次のように記録されます。

```json
{
  "turn_index": 18,
  "requested_offset_ms": -200.0,
  "offset_ms": -160.0,
  "event": "backchannel",
  "source": "backchannel-fixed",
  "score": 1.0
}
```

`score: 1.0`は固定ルールを示す値であり、MaAIの確率ではありません。通常境界では`event`が
`shift`または`hold`になります。

`vap-auto`は出力WAVと同じディレクトリへ次も保存します。

- `*.vap_input.wav`: VAP調整前の重なりなし音声
- `*.vap_input.manifest.json`: VAP解析用ターン位置
- `*.vap_trace.json`: 100 msごとの`p_now`/`p_future`
- `*.vap_turns.json`: VAPまたは固定ルールが要求した境界FTO
- `*.manifest.json`: 安全制約適用後の最終FTOと成果物への参照

## 参照音声付き・複数ファイルの例

```bash
HF_HOME=/path/to/huggingface \
zoom1-dialogue-tts \
  --batch inputs/dialogue1.txt out/dialogue1.wav \
  --batch inputs/dialogue2.txt out/dialogue2.wav \
  --prompt-s1 references/system.wav "参照音声の正確な書き起こし" \
  --turn-timing vap-auto \
  --vap-python .venv-vap/bin/python \
  --vap-device cpu \
  --backchannel-turn-overlap-ms 200 \
  --backchannels none \
  --seed 1
```

`--batch`ではFireRedTTS-2を1回だけロードし、複数ファイルを順番に処理します。MaAI解析は各出力の
`vap_input.wav`に対して個別に実行されます。
