PRESET_DATASETS = {
    "market_data": {
        "title": "market_data.csv",
        "filename": "market_data.csv",
    }
}

FEATURE_COLUMNS = [
    "InStrSpending",
    "Discount",
    "TVSpending",
    "StockRate",
    "Price",
    "Radio",
    "OnlineAdsSpending",
]

FEATURE_LABELS = {
    "InStrSpending": "Расходы в магазине",
    "Discount": "Скидка",
    "TVSpending": "Расходы на ТВ-рекламу",
    "StockRate": "Доля товара в наличии",
    "Price": "Цена",
    "Radio": "Расходы на радиорекламу",
    "OnlineAdsSpending": "Расходы на онлайн-рекламу",
}

FEATURE_HELP = {
    "InStrSpending": "Расходы на продвижение внутри магазина за прогнозируемый период. Значение не должно быть отрицательным.",
    "Discount": "Доля скидки в диапазоне 0..1. Например, 0.15 означает скидку 15%.",
    "TVSpending": "Расходы на ТВ-рекламу за тот же период, что и Sale в обучающем датасете.",
    "StockRate": "Доля доступного товара в диапазоне 0..1. Значение 1 означает полную доступность.",
    "Price": "Цена единицы товара. Отрицательная цена недопустима.",
    "Radio": "Расходы на радиорекламу за прогнозируемый период.",
    "OnlineAdsSpending": "Расходы на онлайн-рекламу за прогнозируемый период.",
}

FEATURE_UNITS = {
    "InStrSpending": "расходы",
    "Discount": "0..1",
    "TVSpending": "расходы",
    "StockRate": "0..1",
    "Price": "цена",
    "Radio": "расходы",
    "OnlineAdsSpending": "расходы",
}

FEATURE_INPUT_RULES = {
    "InStrSpending": {"min": "0", "step": "any", "placeholder": "например 1200"},
    "Discount": {"min": "0", "max": "1", "step": "0.01", "placeholder": "например 0.15"},
    "TVSpending": {"min": "0", "step": "any", "placeholder": "например 5000"},
    "StockRate": {"min": "0", "max": "1", "step": "0.01", "placeholder": "например 0.9"},
    "Price": {"min": "0", "step": "any", "placeholder": "например 99.9"},
    "Radio": {"min": "0", "step": "any", "placeholder": "например 800"},
    "OnlineAdsSpending": {"min": "0", "step": "any", "placeholder": "например 2500"},
}

FEATURE_GROUPS = [
    {
        "title": "Цена и доступность",
        "description": "Факторы, напрямую влияющие на покупку.",
        "features": ["Price", "Discount", "StockRate"],
    },
    {
        "title": "Маркетинговые расходы",
        "description": "Каналы продвижения за прогнозируемый период.",
        "features": ["InStrSpending", "TVSpending", "Radio", "OnlineAdsSpending"],
    },
]

MODEL_OPTIONS = {
    "linear": "Линейная регрессия",
    "knn": "K ближайших соседей",
    "mlp": "MLP",
}

ALLOWED_EXTENSIONS = {"csv"}
