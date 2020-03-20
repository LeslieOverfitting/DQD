# Duplicate Question Detection
参考论文以及其它 Github 仓库代码实现以下模型：
- ESIM ([Enhanced LSTM for Natural Language Inference](https://arxiv.org/pdf/1609.06038.pdf))
- InferSent([Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364))
- BiMPM([Bilateral Multi-Perspective Matching for Natural Language Sentences](https://arxiv.org/pdf/1702.03814.pdf))
- SSE([Shortcut-stacked sentence encoders for multidomain inference_EMNLP 2017](https://arxiv.org/abs/1708.02312))

## Requirements
- python 3.7.4
- 1.4.0

## Performance
| Model     | ACC           |
| --------- | ------------- |
| ESIM      | 84.77%        |
| InferSent | 86.73%        |
| SSE       | 86.76%        |
| BiMPM     | 70%(training) |
