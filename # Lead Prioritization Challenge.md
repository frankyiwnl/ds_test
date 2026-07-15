# Lead Prioritization Challenge

Решение задачи ранжирования обращений по вероятности успешного целевого действия в течение 5 дней после назначения.

Основная модель — `CatBoostClassifier`. В решении используются готовые табличные признаки, исторические события из `events.csv`, временная валидация и подбор гиперпараметров через Optuna.

## Постановка задачи

Для каждого обращения из `test.csv` требуется предсказать `score` — вероятность успешного целевого действия.

Чем выше `score`, тем перспективнее обращение и тем выше оно должно находиться в ранжировании.

Целевая переменная:

- `target = 1` — успешное целевое действие произошло в течение 5 дней после назначения;
- `target = 0` — успешного действия не произошло.

Формат итогового файла:

```text
lead_id,score
```

## Метрика

Основная метрика — **Daily Average Precision**.

Average Precision считается отдельно для каждого дня назначения обращения, после чего значения усредняются по дням:

```text
Daily AP = mean(AP для каждого assignment_date)
```

В локальной проверке используется именно эта схема, а не один общий Average Precision по всей валидационной выборке.

## Структура проекта

```text
project/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── events.csv
├── corrected_lead_solution.ipynb
├── corrected_lead_solution.py
├── README.md
└── submissions/
```

Файлы submission могут сохраняться в корне проекта или в отдельной папке.

## Используемые библиотеки

```bash
pip install pandas numpy scikit-learn catboost optuna
```

Рекомендуемая версия Python: 3.10 или новее.

## Общая схема решения

### 1. Загрузка и подготовка данных

Временные поля приводятся к типу `datetime`.

Обучающая выборка сортируется по времени назначения:

```python
train["assignment_ts"] = pd.to_datetime(train["assignment_ts"])
train["assignment_date"] = pd.to_datetime(train["assignment_date"])
train = train.sort_values("assignment_ts").reset_index(drop=True)
```

Идентификаторы `lead_id`, `user_id`, даты и target не используются как обычные признаки модели.

### 2. Временная валидация

Случайное разбиение не используется, поскольку оно может давать слишком оптимистичную оценку на временных данных.

Вместо этого применяются rolling time folds:

```text
Fold 1: train 2026-04-07..2026-04-14, valid 2026-04-15..2026-04-16
Fold 2: train 2026-04-07..2026-04-16, valid 2026-04-17..2026-04-18
Fold 3: train 2026-04-07..2026-04-18, valid 2026-04-19..2026-04-20
Fold 4: train 2026-04-07..2026-04-20, valid 2026-04-21..2026-04-22
```

Каждый validation-период находится строго позже соответствующей обучающей части.

### 3. Baseline-модель

Первая модель обучается только на готовых табличных признаках из `train.csv`.

Категориальные признаки передаются CatBoost напрямую:

```python
cat_features = [
    "lead_source",
    "call_center",
    "region",
    "car_segment",
    "lead_channel",
    "user_tenure_bucket",
    "price_bucket",
    "assignment_hour",
    "assignment_weekday",
    "is_weekend",
]
```

CatBoost выбран по следующим причинам:

- хорошо работает с табличными данными;
- поддерживает категориальные признаки;
- устойчив к пропускам;
- не требует обязательного one-hot encoding;
- подходит для нелинейных зависимостей и взаимодействий признаков.

### 4. Признаки из `events.csv`

События используются только в том случае, если они произошли до момента назначения обращения:

```python
events_valid = events_merged[
    events_merged["event_ts"] < events_merged["assignment_ts"]
].copy()
```

Это предотвращает утечку информации из будущего.

Для каждого `lead_id` рассчитываются:

- общее количество событий;
- количество уникальных типов событий;
- средняя логарифмическая цена объектов;
- время с момента последнего события;
- количество событий каждого типа;
- количество событий за последние 24 часа;
- признак наличия исторических событий.

Примеры признаков:

```text
total_events
unique_event_types
avg_item_price_log
hours_since_last_event
events_last_24h
count_event_view
count_event_search
count_event_chat
has_events
```

Точный набор колонок зависит от типов событий в исходном файле.

### 5. Trend-признаки

Дополнительно строятся признаки изменения активности пользователя.

Примеры:

- отношение или логарифмическая разница активности за 1 день и 7 дней;
- изменение количества добавлений в избранное;
- разница между ценой текущего объекта и исторической средней ценой;
- количество просмотров на день жизни аккаунта.

Для защиты от деления на ноль и экстремальных выбросов используются устойчивые преобразования:

```python
trend = np.log1p(recent) - np.log1p(long_window / days)
trend = trend.clip(-5, 5)
```

Для лидов без событий создаётся отдельный бинарный признак `has_events`. Отсутствующая история не интерпретируется как реальное нулевое значение цены.

### 6. Подбор гиперпараметров

Для подбора параметров применяется Optuna.

Оптимизируемая функция — средний Daily AP по rolling time folds.

Подбираются:

```text
learning_rate
depth
l2_leaf_reg
random_strength
bagging_temperature
```

Для воспроизводимости фиксируются seed CatBoost и Optuna.

Пример создания study:

```python
sampler = optuna.samplers.TPESampler(seed=42)

study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
)
```

Число деревьев итоговой модели определяется по результатам early stopping на временных фолдах, а не задаётся произвольным большим значением.

## Полученный локальный результат

Лучший результат rolling time cross-validation:

```text
Rolling CV Daily AP: 0.632624
```

Результаты по временным фолдам:

```text
Fold 1: 0.581630
Fold 2: 0.640966
Fold 3: 0.674653
Fold 4: 0.633246
```

Лучшие найденные параметры:

```text
learning_rate:       0.049248519849344134
depth:               5
l2_leaf_reg:         3.3000339465110766
random_strength:     1.3822747149419152
bagging_temperature: 1.700560535237379
iterations:          841
```

Это локальная оценка. Она не гарантирует такое же значение на скрытой тестовой выборке из-за возможного временного сдвига распределения данных.

## Создаваемые submission-файлы

Ноутбук создаёт три основных файла:

```text
catboost_baseline_submission.csv
catboost_events_submission.csv
catboost_optuna_trends_submission.csv
```

Назначение файлов:

- `catboost_baseline_submission.csv` — CatBoost только на готовых табличных признаках;
- `catboost_events_submission.csv` — модель с признаками из `events.csv`;
- `catboost_optuna_trends_submission.csv` — итоговая модель с events-признаками, trend-признаками и параметрами Optuna.

Основной файл для отправки:

```text
catboost_optuna_trends_submission.csv
```

## Дополнительный rank blend

Поскольку метрика зависит от порядка объектов, можно дополнительно проверить ранговый ансамбль итоговой модели и baseline.

```python
import pandas as pd

optuna_sub = pd.read_csv("catboost_optuna_trends_submission.csv")
baseline_sub = pd.read_csv("catboost_baseline_submission.csv")

assert optuna_sub["lead_id"].equals(baseline_sub["lead_id"])

optuna_rank = optuna_sub["score"].rank(method="average", pct=True)
baseline_rank = baseline_sub["score"].rank(method="average", pct=True)

blend_submission = pd.DataFrame({
    "lead_id": optuna_sub["lead_id"],
    "score": 0.8 * optuna_rank + 0.2 * baseline_rank,
})

blend_submission.to_csv(
    "catboost_rank_blend_80_20.csv",
    index=False,
)
```

Этот вариант является дополнительным экспериментом. Его качество необходимо проверять отдельно на платформе или на локальной временной валидации.

## Проверки итогового файла

Перед сохранением submission выполняются проверки:

```python
assert list(submission.columns) == ["lead_id", "score"]
assert len(submission) == len(test)
assert submission["lead_id"].tolist() == test["lead_id"].tolist()
assert submission["lead_id"].is_unique
assert np.isfinite(submission["score"]).all()
assert submission["score"].between(0, 1).all()
```

Это гарантирует:

- правильное количество строк;
- правильные названия колонок;
- сохранение исходного порядка `test.csv`;
- отсутствие дубликатов `lead_id`;
- отсутствие `NaN` и бесконечных значений;
- нахождение всех score в диапазоне от 0 до 1.

## Как запустить решение

### Вариант 1. Jupyter Notebook

Открыть:

```text
corrected_lead_solution.ipynb
```

Запустить все ячейки строго сверху вниз.

### Вариант 2. Python-скрипт

```bash
python corrected_lead_solution.py
```

Запуск необходимо выполнять из корня проекта, в котором находится папка `data`.

## Быстрый тестовый запуск Optuna

Для проверки работоспособности кода без долгого ожидания:

```python
N_TRIALS = 3
N_FOLDS = 2
```

Для полного запуска:

```python
N_TRIALS = 20
N_FOLDS = 4
```

На CPU полный подбор может занимать от нескольких десятков минут до нескольких часов в зависимости от процессора, настроек CatBoost и эффективности early stopping.

## Воспроизводимость

Для воспроизводимости используются:

```python
RANDOM_SEED = 42
```

Seed фиксируется в:

- NumPy;
- CatBoost;
- Optuna `TPESampler`.

Также отключается запись служебных файлов CatBoost:

```python
allow_writing_files=False
```

## Ограничения решения

- Подбор параметров проводится на ограниченном числе trials.
- Скрытая тестовая выборка может отличаться от последних обучающих дат.
- Event-признаки агрегируются на уровне `lead_id`; дополнительные пользовательские временные агрегации могут улучшить результат.
- Rank blend не считается гарантированно лучшим без отдельной валидации.
- Значение локального CV не является результатом на закрытом leaderboard.


## Итог

Решение соблюдает временной порядок данных, исключает события после назначения обращения, использует официальную логику Daily Average Precision и формирует воспроизводимый submission в требуемом формате.