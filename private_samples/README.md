# Private research sample

`vap_boundary_12_backchannels.wav`は、VAPで検出した12箇所へ聞き手反応を配置した
24 kHz・16 bit・ステレオ対話音声です。左がS1、右がS2です。

## 利用条件

- Zoom1収録話者の実音声をvoice-cloneプロンプトとして使用しています。
- 元のZoom1音声そのものやプロンプト音声は、このリポジトリには収録していません。
- LLM-jp内の許諾された研究・評価用途に限定し、公開配信、再配布、第三者提供には使用しないでください。
- このディレクトリは非公開GitHubリポジトリ専用です。GitHub Pagesの配信元である`docs/`へ移動しないでください。
- リポジトリを公開へ変更する前に、このディレクトリを履歴を含めて分離してください。

## 生成条件

- Model: FireRedTTS-2 Zoom1 fine-tune `drop`
- Voice prompts: Zoom1 dialogue `diag0013`のS1/S2実音声
- Script: `dialogue.txt`
- Backchannel timing: `vap_points.json`
- Boundary protection: turn start 0.6秒、turn end 1.6秒を候補から除外
- Backchannels: 12箇所、すべてS2
- SHA-256: `1aa7d692720eb1ba9f1b4ed6bdb59d3e8830137b261a1ccf6ba34acd5cb0f644`

発話本体と相槌はモデルによる合成音声です。自然さの差には、実音声プロンプトによる音色・
韻律条件と、VAPタイミングおよび境界保護の両方が影響しています。
