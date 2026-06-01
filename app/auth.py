from flask import Blueprint, render_template, request, redirect, url_for, session, flash, current_app

from app.database.db import (
    create_user,
    get_user_by_email,
    get_user_by_username,
    verify_user_password,
    get_all_users,
    update_user_role,
    delete_user_by_id,
    get_user_by_id,
    get_admin_dashboard_stats,
    update_user_profile,
    update_user_password,
    VALID_ROLES,
)
from app.decorators import login_required, role_required

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        if not email or not password:
            return render_template("login.html", error="Заполните все поля.")

        user = get_user_by_email(current_app.config["DATABASE_PATH"], email)

        if not user or not verify_user_password(user, password):
            return render_template("login.html", error="Неверный email или пароль.")

        session["user_id"] = user["id"]
        session["username"] = user["username"]
        session["user_role"] = user["role"]

        flash("Вы успешно вошли в систему.", "success")
        return redirect(url_for("main.index"))

    return render_template("login.html")


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not username or not email or not password or not confirm_password:
            return render_template("register.html", error="Заполните все поля.")

        if len(username) < 3:
            return render_template("register.html", error="Логин должен содержать минимум 3 символа.")

        if "@" not in email:
            return render_template("register.html", error="Введите корректный email.")

        if len(password) < 6:
            return render_template("register.html", error="Пароль должен содержать минимум 6 символов.")

        if password != confirm_password:
            return render_template("register.html", error="Пароли не совпадают.")

        if get_user_by_username(current_app.config["DATABASE_PATH"], username):
            return render_template("register.html", error="Пользователь с таким логином уже существует.")

        if get_user_by_email(current_app.config["DATABASE_PATH"], email):
            return render_template("register.html", error="Пользователь с таким email уже существует.")

        create_user(
            db_path=current_app.config["DATABASE_PATH"],
            username=username,
            email=email,
            password=password,
            role="analyst"
        )

        flash("Регистрация выполнена успешно. Теперь войдите в систему.", "success")
        return redirect(url_for("auth.login"))

    return render_template("register.html")


@auth_bp.route("/logout")
@login_required
def logout():
    session.clear()
    flash("Вы вышли из системы.", "info")
    return redirect(url_for("main.index"))


@auth_bp.route("/users", methods=["GET", "POST"])
@login_required
@role_required("admin")
def users():
    db_path = current_app.config["DATABASE_PATH"]

    if request.method == "POST":
        action = request.form.get("action")
        user_id_raw = request.form.get("user_id", "").strip()

        if not user_id_raw.isdigit():
            flash("Некорректный идентификатор пользователя.", "danger")
            return redirect(url_for("auth.users"))

        user_id = int(user_id_raw)
        target_user = get_user_by_id(db_path, user_id)

        if not target_user:
            flash("Пользователь не найден.", "danger")
            return redirect(url_for("auth.users"))

        current_user_id = session.get("user_id")

        if action == "change_role":
            new_role = request.form.get("new_role", "").strip()

            if new_role not in VALID_ROLES:
                flash("Некорректная роль.", "danger")
                return redirect(url_for("auth.users"))

            if user_id == current_user_id and new_role != "admin":
                flash("Нельзя изменить собственную роль администратора.", "danger")
                return redirect(url_for("auth.users"))

            update_user_role(db_path, user_id, new_role)
            flash("Роль пользователя успешно обновлена.", "success")
            return redirect(url_for("auth.users"))

        elif action == "delete_user":
            if user_id == current_user_id:
                flash("Нельзя удалить собственную учетную запись.", "danger")
                return redirect(url_for("auth.users"))

            delete_user_by_id(db_path, user_id)
            flash("Пользователь успешно удалён.", "success")
            return redirect(url_for("auth.users"))

        else:
            flash("Неизвестное действие.", "danger")
            return redirect(url_for("auth.users"))

    rows = get_all_users(db_path)
    user_stats = {
        "total": len(rows),
        "admins": sum(1 for row in rows if row["role"] == "admin"),
        "analysts": sum(1 for row in rows if row["role"] == "analyst"),
    }
    return render_template("users.html", rows=rows, user_stats=user_stats)


@auth_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    db_path = current_app.config["DATABASE_PATH"]
    user_id = session.get("user_id")
    user = get_user_by_id(db_path, user_id)

    if not user:
        session.clear()
        flash("Пользователь не найден. Войдите снова.", "danger")
        return redirect(url_for("auth.login"))

    if request.method == "POST":
        action = request.form.get("action")

        if action == "update_profile":
            username = request.form.get("username", "").strip()
            email = request.form.get("email", "").strip().lower()

            if not username or not email:
                return render_template("profile.html", error="Заполните логин и email.", user=user)

            if len(username) < 3:
                return render_template("profile.html", error="Логин должен содержать минимум 3 символа.", user=user)

            if "@" not in email:
                return render_template("profile.html", error="Введите корректный email.", user=user)

            existing_username = get_user_by_username(db_path, username)
            if existing_username and existing_username["id"] != user_id:
                return render_template("profile.html", error="Этот логин уже занят.", user=user)

            existing_email = get_user_by_email(db_path, email)
            if existing_email and existing_email["id"] != user_id:
                return render_template("profile.html", error="Этот email уже используется.", user=user)

            update_user_profile(db_path, user_id, username, email)

            session["username"] = username
            flash("Профиль успешно обновлён.", "success")
            return redirect(url_for("auth.profile"))

        elif action == "update_password":
            current_password = request.form.get("current_password", "").strip()
            new_password = request.form.get("new_password", "").strip()
            confirm_password = request.form.get("confirm_password", "").strip()

            if not current_password or not new_password or not confirm_password:
                return render_template("profile.html", error="Заполните все поля для смены пароля.", user=user)

            if len(new_password) < 6:
                return render_template("profile.html", error="Новый пароль должен содержать минимум 6 символов.", user=user)

            if not verify_user_password(user, current_password):
                return render_template("profile.html", error="Текущий пароль указан неверно.", user=user)

            if new_password != confirm_password:
                return render_template("profile.html", error="Новые пароли не совпадают.", user=user)

            update_user_password(db_path, user_id, new_password)
            flash("Пароль успешно изменён.", "success")
            return redirect(url_for("auth.profile"))

    refreshed_user = get_user_by_id(db_path, user_id)
    return render_template("profile.html", user=refreshed_user)


@auth_bp.route("/admin")
@login_required
@role_required("admin")
def admin_dashboard():
    stats = get_admin_dashboard_stats(current_app.config["DATABASE_PATH"])
    return render_template("admin.html", stats=stats)
