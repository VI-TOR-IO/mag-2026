from flask import Flask, render_template
from config import Config
from app.database.db import init_db


def create_app(test_config: dict | None = None):
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)

    init_db(
        db_path=app.config["DATABASE_PATH"],
        create_default_admin=app.config["DEFAULT_ADMIN_ENABLED"],
        default_admin_username=app.config["DEFAULT_ADMIN_USERNAME"],
        default_admin_email=app.config["DEFAULT_ADMIN_EMAIL"],
        default_admin_password=app.config["DEFAULT_ADMIN_PASSWORD"],
    )

    from app.routes import main_bp
    from app.auth import auth_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)

    register_error_handlers(app)

    return app


def register_error_handlers(app):
    @app.errorhandler(404)
    def not_found(error):
        return render_template(
            "error.html",
            title="Страница не найдена",
            message="Проверьте адрес или вернитесь на главную страницу.",
        ), 404

    @app.errorhandler(413)
    def file_too_large(error):
        return render_template(
            "error.html",
            title="Файл слишком большой",
            message="Загрузите CSV меньшего размера или измените MAX_CONTENT_LENGTH в конфигурации.",
        ), 413
