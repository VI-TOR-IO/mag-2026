from functools import wraps
from flask import session, redirect, url_for, flash


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            flash("Для доступа к разделу необходимо войти в систему.", "warning")
            return redirect(url_for("auth.login"))
        return view_func(*args, **kwargs)
    return wrapped_view


def role_required(*allowed_roles):
    def decorator(view_func):
        @wraps(view_func)
        def wrapped_view(*args, **kwargs):
            if "user_id" not in session:
                flash("Для доступа к разделу необходимо войти в систему.", "warning")
                return redirect(url_for("auth.login"))

            user_role = session.get("user_role")
            if user_role not in allowed_roles:
                flash("У вас недостаточно прав для доступа к этой странице.", "danger")
                return redirect(url_for("main.index"))

            return view_func(*args, **kwargs)
        return wrapped_view
    return decorator