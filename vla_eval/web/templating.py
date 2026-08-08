from pathlib import Path

from fastapi.templating import Jinja2Templates

from vla_eval.time_utils import format_beijing_time

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
templates.env.filters["beijing_time"] = format_beijing_time
