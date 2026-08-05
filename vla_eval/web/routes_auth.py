from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from vla_eval.models import User
from vla_eval.security import (
    authenticate_user,
    get_current_user,
    new_csrf_token,
    require_csrf,
    require_html_user,
)

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
_INVALID_CREDENTIALS = "用户名或密码无效"


def _login_response(request: Request, *, error: str | None = None, status_code: int = 200):
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"csrf_token": request.session["csrf_token"], "error": error},
        status_code=status_code,
    )


@router.get("/login", name="login")
def login_page(request: Request):
    if get_current_user(request) is not None:
        return RedirectResponse("/datasets", status_code=303)
    request.session.clear()
    request.session["csrf_token"] = new_csrf_token()
    return _login_response(request)


@router.post("/login")
async def login(request: Request):
    form = await request.form()
    require_csrf(request, form.getlist("csrf_token"))
    usernames = form.getlist("username")
    passwords = form.getlist("password")
    user = None
    if (
        len(usernames) == 1
        and len(passwords) == 1
        and isinstance(usernames[0], str)
        and isinstance(passwords[0], str)
        and usernames[0]
        and passwords[0]
    ):
        user = await run_in_threadpool(
            authenticate_user,
            request.app.state.engine,
            usernames[0],
            passwords[0],
        )
    if user is None:
        request.session.clear()
        request.session["csrf_token"] = new_csrf_token()
        return _login_response(request, error=_INVALID_CREDENTIALS, status_code=401)

    request.session.clear()
    request.session.update(user_id=user.id, csrf_token=new_csrf_token())
    return RedirectResponse("/datasets", status_code=303)


@router.post("/logout")
async def logout(request: Request):
    form = await request.form()
    require_csrf(request, form.getlist("csrf_token"))
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@router.get("/datasets", name="datasets")
def datasets_placeholder(
    request: Request,
    current_user: Annotated[User, Depends(require_html_user)],
):
    return templates.TemplateResponse(
        request=request,
        name="base.html",
        context={"current_user": current_user, "csrf_token": request.session["csrf_token"]},
    )
