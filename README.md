# Qudor AI v2

Ускоренная среда обучения Quoridor 1v1 (9×9, 10 стен): bitboard-движок, одновременная генерация десятков партий, batched GPU inference, AMP, residual policy/value network, resumable checkpoints и MCTS-модуль.

## Google Colab
Загрузите папку `Qudor` в корень Google Drive и откройте `notebooks/quoridor_fast_training.ipynb`. Включите T4 GPU и запускайте ячейки сверху вниз.

- `colab_t4_fast.json` — 5 итераций для проверки.
- `colab_t4_balanced.json` — длительное обучение.
- Результаты: `MyDrive/Qudor_runs/v2/`.
- Повторный запуск продолжает `latest.pt`.

Старые checkpoints находятся в `legacy/` и сохраняются для teacher/arena. Ночная точка старта из `quoridor_ai_runs-20260806T024625Z-1-001.zip` импортирована как основной teacher: `seed_11` iteration 50. Новая сеть имеет другие входные плоскости и residual-блоки, поэтому прямое продолжение старого optimizer-state невозможно; это намеренная смена архитектуры. Старый seed_11 iteration 10 сохранён в `legacy/archive_before_night/`.
