## Spectogram

Song analysis based on the Mel Spectograms (https://en.wikipedia.org/wiki/Mel_scale)

![Dixon](https://github.com/petrjanda/deepsong/raw/master/docs/dixon.png "Dixon")
![Move D](https://github.com/petrjanda/deepsong/raw/master/docs/moved.png "Move D")

Its able to overfit, so its able to learn!

```
epoch: 55, loss: 0.08677666140037
ConfusionMatrix:
[[      82       0       0       0       0       0       0       0]   100.000%  [class: 1]
 [       0      80       0       0       0       0       0       0]   100.000%  [class: 2]
 [       0       0      80       0       0       0       0       0]   100.000%  [class: 3]
 [       0       0       0      80       0       0       0       0]   100.000%  [class: 4]
 [       0       0       0       0      90       0       0       0]   100.000%  [class: 5]
 [       0       0       0       0       0     100       0       0]   100.000%  [class: 6]
 [       0       0       0       0       0       0     100       0]   100.000%  [class: 7]
 [       0       0       0       0       0       0       0     100]]  100.000%  [class: 8]
 + average row correct: 100%
 + average rowUcol correct (VOC measure): 100%
 + global correct: 100%
```
