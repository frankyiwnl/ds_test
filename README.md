# Lead Prioritization Challenge

Решение задачи ранжирования обращений по вероятности успешного целевого действия в течение 5 дней после назначения.

Основная модель — `CatBoostClassifier`. В решении используются готовые табличные признаки, исторические события из `events.csv`, расширенные временные признаки, trend-признаки и funnel-признаки.

Финальный результат сохраняется в файл:

```text
submission_enhanced.csv
```

## Постановка задачи

Для каждого обращения из `test.csv` требуется предсказать `score`, отражающий перспективность обращения.

Чем выше `score`, тем выше обращение должно находиться в ранжировании.

Целевая переменная:

- `target = 1` — успешное целевое действие произошло в течение 5 дней после назначения;
- `target = 0` — успешного действия не произошло.

Формат итогового файла:

```text
lead_id,score
```

## Метрика

Основная метрика — **Daily Average Precision**.

Average Precision рассчитывается отдельно для каждого дня назначения обращения, после чего полученные значения усредняются:

```text
Daily AP = mean(AP для каждого assignment_date)
```

Таким образом, важен порядок объектов внутри каждого отдельного дня, а не только глобальный порядок по всей тестовой выборке.

В коде присутствует функция:

```python
daily_average_precision(...)
```

Она:

1. группирует строки по `assignment_date`;
2. рассчитывает Average Precision внутри каждого дня;
3. возвращает среднее значение по всем дням.

## Структура проекта

```text
project/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── events.csv
├── corrected_lead_solution.ipynb
├── README.md
├── requirements.txt
└── submission_enhanced.csv
```

Файл `submission_enhanced.csv` создаётся после выполнения финальной ячейки ноутбука.

Файл `sample_submission.csv` для запуска решения не требуется.

## Установка зависимостей

Рекомендуемая версия Python — 3.10 или новее.

Установка зависимостей:

```bash
pip install numpy pandas scikit-learn catboost jupyter
```

Либо через `requirements.txt`:

```bash
pip install -r requirements.txt
```

Содержимое `requirements.txt`:

```text
numpy
pandas
scikit-learn
catboost
jupyter
```

## Общая схема решения

### 1. Загрузка данных

Из папки `data` загружаются:

```text
train.csv
test.csv
events.csv
```

Временные поля преобразуются в `datetime`:

```python
frame["assignment_ts"] = pd.to_datetime(
    frame["assignment_ts"],
    errors="raise",
)

frame["assignment_date"] = pd.to_datetime(
    frame["assignment_date"],
    errors="raise",
).dt.normalize()
```

Обучающая выборка сортируется по времени назначения:

```python
train_df = train_df.sort_values(
    ["assignment_ts", "lead_id"]
).reset_index(drop=True)
```

Исходный порядок строк в `test.csv` сохраняется без изменений.

Дополнительно проверяется:

- уникальность `lead_id` в train;
- уникальность `lead_id` в test;
- отсутствие пересечений `lead_id` между train и test.

## Исключаемые признаки

Следующие колонки не передаются модели как обычные признаки:

```python
DROP_COLS = [
    "lead_id",
    "user_id",
    "assignment_ts",
    "assignment_date",
    "target",
]
```

Причины:

- `lead_id` является идентификатором обращения;
- `user_id` не используется как переносимый пользовательский признак;
- `assignment_ts` и `assignment_date` используются для временной логики;
- `target` является целевой переменной.

При этом отдельные производные временные признаки, например час и день недели, используются моделью.

## Категориальные признаки

Базовые категориальные признаки:

```python
CAT_FEATURES = [
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

После обработки событий к ним добавляются:

```python
last_event_type
last_ctx_seq
```

Категориальные признаки передаются в CatBoost напрямую, без ручного one-hot encoding.

Пропущенные категориальные значения заменяются на специальные категории:

```text
__MISSING__
__NO_EVENT__
```

## Baseline-модель

В ноутбуке присутствует дополнительная baseline-модель, использующая только исходные табличные признаки.

Baseline нужен для:

- проверки корректности загрузки данных;
- получения базового качества;
- сравнения с моделью, использующей `events.csv`.

Baseline обучается с временным holdout: последние даты используются как validation, более ранние даты — как train.

Основной итоговый submission создаёт не baseline, а финальная enhanced-модель.

Baseline-ячейку можно не запускать, если требуется только итоговый файл.

## Признаки из `events.csv`

Исторические события объединяются с обращениями по `lead_id`.

Главное условие защиты от временной утечки:

```python
events_valid = events_valid.loc[
    events_valid["event_ts"].notna()
    & (events_valid["event_ts"] < events_valid["assignment_ts"])
].copy()
```

Используются только события, произошедшие строго раньше момента назначения обращения:

```text
event_ts < assignment_ts
```

События после назначения и события в тот же момент времени исключаются.

## Типы событий

В решении отдельно обрабатываются следующие типы событий:

```python
event_types = [
    "item_view",
    "search",
    "favorite",
    "chat_open",
    "call_click",
]
```

Для них рассчитываются общие количества, количества по временным окнам, recency и экспоненциально затухающая активность.

## Общие event-признаки

Для каждого `lead_id` рассчитываются:

```text
event_total_count
event_type_nunique
event_day_nunique
event_price_log_mean
event_price_log_std
event_price_log_min
event_price_log_max
event_src_slot_mean
event_src_slot_std
event_src_slot_nunique
event_ctx_seq_nunique
event_hours_since_last
event_hours_since_first
```

Эти признаки описывают:

- общий объём истории;
- разнообразие действий;
- длительность истории;
- статистики просмотренных цен;
- разнообразие источников и контекста;
- давность первого и последнего события.

## Признаки последнего события

Для самого последнего события до назначения сохраняются:

```text
last_event_type
last_ctx_seq
last_event_src_slot
last_event_price_log
last_ctx_seq_code
```

Также создаются бинарные признаки:

```text
last_event_is_item_view
last_event_is_search
last_event_is_favorite
last_event_is_chat_open
last_event_is_call_click
```

Это позволяет модели различать ситуации, когда последнее действие пользователя было обычным просмотром, поиском, добавлением в избранное, открытием чата или нажатием на звонок.

## Временные окна событий

Количество событий рассчитывается для нескольких временных окон перед назначением:

```text
6 часов
24 часа
72 часа
7 дней
```

Окна задаются следующим образом:

```python
windows = [
    (6, "6h"),
    (24, "24h"),
    (72, "72h"),
    (168, "7d"),
]
```

Для каждого окна создаётся:

- общее количество событий;
- количество `item_view`;
- количество `search`;
- количество `favorite`;
- количество `chat_open`;
- количество `call_click`.

Примеры:

```text
event_count_6h
event_count_24h
event_count_72h
event_count_7d

event_count_item_view_24h
event_count_search_24h
event_count_favorite_72h
event_count_chat_open_7d
event_count_call_click_6h
```

Также рассчитываются количества событий каждого типа за всю доступную историю:

```text
event_count_item_view_all
event_count_search_all
event_count_favorite_all
event_count_chat_open_all
event_count_call_click_all
```

## Recency-признаки

Для каждого типа события рассчитывается время с последнего действия:

```text
event_hours_since_last_item_view
event_hours_since_last_search
event_hours_since_last_favorite
event_hours_since_last_chat_open
event_hours_since_last_call_click
```

Отдельно создаются:

```text
event_hours_since_last_missing
event_log_hours_since_last
event_history_span_hours
```

`event_log_hours_since_last` использует логарифмическое преобразование, чтобы уменьшить влияние очень больших значений.

## Экспоненциально затухающая активность

Недавние события могут быть важнее старых, поэтому дополнительно используется exponential decay.

Для каждого события рассчитывается вес:

```python
weight = np.exp(
    -hours_before_assign / tau_hours
)
```

Используются два масштаба времени:

```text
tau = 24 часа
tau = 168 часов
```

Примеры признаков:

```text
event_decay_item_view_24h
event_decay_search_24h
event_decay_favorite_24h

event_decay_item_view_168h
event_decay_chat_open_168h
event_decay_call_click_168h
```

Чем ближе событие к моменту назначения, тем больший вклад оно вносит в значение признака.

## Признак наличия событий

Для обращений без исторических событий создаётся бинарный признак:

```text
has_events
```

Он равен:

```text
1 — у обращения есть события до назначения;
0 — исторических событий нет.
```

Отсутствие истории обрабатывается отдельно и не интерпретируется как обычное нулевое значение цены или recency.

## Advanced-признаки

После объединения исходных данных и событий создаются дополнительные признаки.

### Циклические временные признаки

Час назначения и день недели преобразуются через синус и косинус:

```python
assignment_hour_sin
assignment_hour_cos
assignment_weekday_sin
assignment_weekday_cos
```

Это позволяет модели учитывать циклическую природу времени.

Например, 23:00 и 00:00 оказываются близкими значениями, несмотря на разницу числовых кодов.

## Trend-признаки

Trend-признаки сравнивают среднюю интенсивность активности за короткое и длинное временное окно.

Используется преобразование:

```python
short_rate = short_value / short_days
long_rate = long_value / long_days

trend = np.log(
    (short_rate + 0.25)
    / (long_rate + 0.25)
).clip(-5, 5)
```

Сравниваются окна:

```text
1 день против 7 дней
3 дня против 14 дней
7 дней против 30 дней
14 дней против 90 дней
```

Trend-признаки строятся для следующих групп активности:

```text
item_views
item_favorites
detail_expands
photo_swipes
seller_page_views
search_views
query_refinements
similar_item_clicks
saved_search_matches
user_contacts
chat_opens
call_clicks
leadgen_prev_assigned
leadgen_prev_answered
leadgen_prev_positive
active_days_auto
```

Примеры:

```text
trend_item_views_1d_vs_7d
trend_item_favorites_3d_vs_14d
trend_seller_page_views_7d_vs_30d
trend_user_contacts_14d_vs_90d
```

Положительное значение означает, что недавняя активность выше долгосрочного среднего.

Отрицательное значение означает снижение активности.

## Funnel-признаки

Дополнительно создаются сглаженные отношения между последовательными действиями пользователя.

Используется преобразование:

```python
rate = np.log(
    (numerator + 0.5)
    / (denominator + 1.0)
).clip(-7, 3)
```

Признаки рассчитываются для окон:

```text
7 дней
30 дней
```

Используются следующие отношения:

```text
item_favorites / item_views
detail_expands / item_views
photo_swipes / item_views
seller_page_views / item_views
query_refinements / search_views
similar_item_clicks / search_views
saved_search_matches / search_views
user_contacts / item_views
chat_opens / user_contacts
call_clicks / user_contacts
leadgen_prev_answered / leadgen_prev_assigned
leadgen_prev_positive / leadgen_prev_assigned
leadgen_prev_positive / leadgen_prev_answered
```

Примеры:

```text
rate_item_favorites_per_item_views_7d
rate_user_contacts_per_item_views_30d
rate_chat_opens_per_user_contacts_7d
rate_leadgen_prev_positive_per_leadgen_prev_assigned_30d
```

Сглаживание защищает от деления на ноль и слишком больших значений на малом количестве наблюдений.

## Ценовые признаки

Цена текущего объекта сравнивается с историей пользователя.

Создаются:

```text
price_diff_event_mean
price_diff_event_min
price_diff_event_max
price_diff_last_event
price_history_missing
```

Примеры логики:

```python
price_diff_event_mean = (
    current_item_price_log
    - historical_event_price_log_mean
)
```

Это позволяет определить, отличается ли текущий объект по цене от объектов, с которыми пользователь взаимодействовал ранее.

## Доли типов событий

Для каждого типа события рассчитывается его доля в общей активности:

```text
event_share_item_view_all
event_share_search_all
event_share_favorite_all
event_share_chat_open_all
event_share_call_click_all
```

Также доли рассчитываются внутри временных окон:

```text
24 часа
72 часа
7 дней
```

Примеры:

```text
event_share_item_view_24h
event_share_favorite_72h
event_share_chat_open_7d
event_share_call_click_24h
```

Это помогает различать пользователей с одинаковым общим количеством событий, но разной структурой поведения.

## Дополнительные интенсивности

Создаются дополнительные признаки:

```text
event_density_per_day
log_views_per_day_alive
log_contacts_per_active_day
feature_missing_count
```

Они описывают:

- плотность событий относительно длины истории;
- количество просмотров относительно возраста пользователя;
- количество контактов относительно активных дней;
- количество пропущенных исходных признаков.

## Финальная модель

Финальная модель обучается на всех доступных обучающих данных.

Используемые параметры:

```python
final_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.04,
    depth=6,
    l2_leaf_reg=8.0,
    random_strength=0.8,
    bagging_temperature=0.7,
    loss_function="Logloss",
    random_seed=42,
    task_type="CPU",
    thread_count=4,
    border_count=32,
    rsm=0.8,
    allow_writing_files=False,
    verbose=100,
)
```

Параметры были зафиксированы после локальных экспериментов с временным разбиением.

Финальная версия не запускает новый подбор гиперпараметров при каждом выполнении ноутбука. Это сокращает время работы и делает результат более воспроизводимым.

## Почему используется CatBoost

CatBoost выбран по следующим причинам:

- хорошо работает с табличными данными;
- поддерживает категориальные признаки напрямую;
- устойчив к пропускам;
- моделирует нелинейные зависимости;
- автоматически учитывает взаимодействия признаков;
- не требует обязательного one-hot encoding;
- подходит для смешанных числовых и категориальных данных.

## Ранжирование внутри дня

После получения вероятностей CatBoost прогнозы переводятся в percentile rank отдельно внутри каждого `assignment_date`:

```python
day_rank_scores = (
    rank_frame.groupby("assignment_date")["raw_score"]
    .rank(method="average", pct=True)
    .to_numpy()
)
```

Это соответствует структуре метрики, которая рассчитывается отдельно для каждого дня.

Итоговые значения `score` находятся в диапазоне от 0 до 1.

При этом порядок объектов внутри каждого дня сохраняет порядок, заданный вероятностями модели.

## Локальный результат

Во время отдельной временной проверки enhanced-подход показал:

```text
Daily AP: примерно 0.6857
```

Для сравнения, предыдущий вариант на сопоставимом последнем временном периоде показывал около:

```text
Daily AP: 0.6332
```

Полученное улучшение связано прежде всего с:

- расширенными event-признаками;
- раздельными временными окнами;
- recency по типам событий;
- признаками последнего события;
- exponential decay;
- trend-признаками;
- funnel-признаками;
- ценовыми сравнениями.

Локальное значение не гарантирует аналогичный результат на скрытой тестовой выборке из-за возможного временного сдвига распределения данных.

## Создаваемые submission-файлы

Основной файл:

```text
submission_enhanced.csv
```

При запуске optional baseline-ячейки дополнительно может быть создан:

```text
catboost_baseline_submission.csv
```

Для отправки на платформу используется:

```text
submission_enhanced.csv
```

## Проверки итогового файла

Перед сохранением выполняются проверки:

```python
assert list(submission.columns) == ["lead_id", "score"]
assert len(submission) == len(test_frame)
assert submission["lead_id"].is_unique
assert submission["lead_id"].tolist() == test_frame["lead_id"].tolist()
assert np.isfinite(submission["score"]).all()
assert submission["score"].between(0.0, 1.0).all()
```

Эти проверки гарантируют:

- правильные названия колонок;
- правильное количество строк;
- отсутствие дубликатов `lead_id`;
- сохранение исходного порядка `test.csv`;
- отсутствие `NaN`;
- отсутствие бесконечных значений;
- нахождение всех `score` в диапазоне от 0 до 1.

## Как запустить решение

### 1. Подготовить данные

Поместить файлы:

```text
train.csv
test.csv
events.csv
```

в папку:

```text
data/
```

### 2. Открыть ноутбук

```bash
jupyter notebook corrected_lead_solution.ipynb
```

### 3. Перезапустить ядро

Перед запуском рекомендуется выполнить:

```text
Kernel → Restart Kernel
```

### 4. Запустить необходимые ячейки

Если считать только кодовые ячейки:

```text
1 → 3 → 5
```

Если считать все ячейки, включая Markdown:

```text
2 → 6 → 10
```

Назначение этих ячеек:

```text
1. Загрузка train/test и определение общих функций.
2. Построение расширенных признаков из events.csv.
3. Создание advanced-признаков, обучение модели и сохранение submission.
```

Baseline-ячейку запускать необязательно.

Также можно выполнить все ячейки сверху вниз. В таком случае дополнительно обучится baseline-модель.

## Результат выполнения

После завершения финальной ячейки появится сообщение:

```text
Готов файл submission_enhanced.csv
```

Итоговый файл будет сохранён в корне проекта:

```text
project/
└── submission_enhanced.csv
```

## Воспроизводимость

Для воспроизводимости используется:

```python
RANDOM_SEED = 42
```

Seed фиксируется в CatBoost:

```python
random_seed=RANDOM_SEED
```

Также используются фиксированные:

- параметры модели;
- порядок обработки строк;
- список признаков;
- правила заполнения пропусков;
- правила построения event-признаков;
- percentile ranking внутри дня.

Запись служебной директории CatBoost отключена:

```python
allow_writing_files=False
```

## Защита от утечек

Решение не использует:

- события после момента назначения;
- события в момент назначения;
- `target` как входной признак;
- `lead_id` как входной признак;
- `user_id` как входной признак;
- данные скрытой тестовой разметки;
- внешние API;
- ручную разметку test;
- восстановление target по идентификаторам.

Ключевое условие:

```python
event_ts < assignment_ts
```

## Ограничения решения

- Финальные параметры модели зафиксированы и не подбираются заново при каждом запуске.
- Локальная временная проверка может отличаться от скрытого leaderboard.
- Распределение данных в test может отличаться от последних дат train.
- Модель использует агрегированные события, а не полную последовательную модель поведения.
- Percentile rank сохраняет порядок, но не является откалиброванной вероятностью.
- Дополнительный ансамбль моделей в финальной версии не используется.

## Итог

Решение:

- соблюдает временной порядок данных;
- исключает события после назначения;
- использует официальную логику Daily Average Precision;
- строит расширенные признаки из `events.csv`;
- учитывает recency и структуру событий;
- использует trend- и funnel-признаки;
- ранжирует прогнозы отдельно внутри каждого дня;
- сохраняет результат в требуемом формате;
- формирует воспроизводимый файл `submission_enhanced.csv`.

Основной файл решения:

```text
corrected_lead_solution.ipynb
```

Основной файл для отправки:

```text
submission_enhanced.csv
```
