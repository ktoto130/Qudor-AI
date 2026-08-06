# Команды запуска

В Colab сначала выберите T4 GPU. Затем загрузите папку `Qudor` в корень `MyDrive`.

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Qudor
!python -m pip install -q -e .
!python scripts/benchmark.py
!python -m quoridor_ai.train --config configs/colab_t4_fast.json --output /content/drive/MyDrive/Qudor_runs/v2
```

После успешного fast-теста:

```python
%cd /content/drive/MyDrive/Qudor
!python -m quoridor_ai.train --config configs/colab_t4_balanced.json --output /content/drive/MyDrive/Qudor_runs/v2
```

Статистика:

```python
import pandas as pd
p='/content/drive/MyDrive/Qudor_runs/v2/metrics.csv'
df=pd.read_csv(p); display(df.tail(20)); df.plot(x='iteration',y=['total_loss','games_per_sec','positions_per_sec'],subplots=True,figsize=(12,10),grid=True)
```
