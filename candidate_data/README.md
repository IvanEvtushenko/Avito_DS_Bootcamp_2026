# Данные для кандидатов

Файлы:

- `articles.f`: корпус статей справки с колонками `article_id`, `title`, `body`.
- `calibration.f`: размеченные примеры с колонками `query_id`, `query_text`,
  `ground_truth`.
- `test.f`: запросы без разметки с колонками `query_id`, `query_text`.

Кандидат должен отправить `answer.csv` с колонками `query_id` и `answer`.
