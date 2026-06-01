# Sales Forecast App

Веб-приложение на Flask для обучения и сравнения моделей прогнозирования продаж.

## Возможности

- регистрация, вход, роли `admin` и `analyst`;
- обучение моделей Linear Regression, KNN и MLP на CSV-датасете;
- сравнение моделей по R², MAE и RMSE;
- графики качества обучения и сравнения моделей без внешних CDN;
- предпросмотр CSV: первые строки, типы, диапазоны и предупреждения по качеству;
- история обучений моделей с параметрами, метриками, cross-validation и путями к артефактам;
- экспорт истории обучений и отчётов модели в CSV/PDF;
- ручной прогноз по семи признакам;
- сохранение истории прогнозов в SQLite;
- экспорт истории в CSV для администратора;
- тесты для БД, авторизации, доступов, экспорта и валидации данных.

## Ожидаемый CSV

Файл должен содержать столбцы:

```text
Sale, InStrSpending, Discount, TVSpending, StockRate, Price, Radio, OnlineAdsSpending
```

Дополнительные столбцы игнорируются. Значения должны быть числовыми.

## Запуск

Текущее окружение `venv` в проекте содержит установленные зависимости, но его `python.exe` указывает на отсутствующий WindowsApps Python. Практичный вариант для локального запуска — пересоздать окружение:

```powershell
Remove-Item -Recurse -Force .\venv
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe run.py
```

Переменные окружения:

```powershell
$env:SECRET_KEY="replace-with-real-secret"
$env:DEFAULT_ADMIN_EMAIL="admin@example.com"
$env:DEFAULT_ADMIN_PASSWORD="change-me"
```

По умолчанию для пустой базы создаётся администратор `admin@example.com / admin123`. Для учебного режима это удобно, для демонстрации проекта пароль нужно переопределить.

## Тесты

Тесты написаны на `unittest`:

```powershell
python -m unittest discover -s tests
```

Если используется текущий сломанный `venv`, тесты можно запускать системным Python/Codex Python: они подключают чистые Flask-зависимости из `venv\Lib\site-packages` через `tests/bootstrap.py`.

## Структура

- `app/routes.py` — пользовательские сценарии прогнозирования, обучения, сравнения и истории;
- `app/auth.py` — аутентификация, профиль, администрирование пользователей;
- `app/database/db.py` — SQLite-схема и операции;
- `app/ml` — подготовка данных, обучение, загрузка моделей и прогноз;
- `app/charts.py` — подготовка данных для встроенных SVG-графиков;
- `app/templates` — Jinja-шаблоны;
- `app/static/css/style.css` — единая дизайн-система интерфейса;
- `tests` — автоматические тесты.
