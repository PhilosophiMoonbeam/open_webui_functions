"""
title: Gemini Enterprise
id: gemini_enterprise_toggle
description: Switches to Gemini Enterprise API
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 1.0.0
"""

from pydantic import BaseModel


class Filter:
    class Valves(BaseModel):
        pass

    def __init__(self) -> None:
        self.valves = self.Valves()
        # Makes the filter toggleable in the front-end.
        self.toggle = True
        # Gemini Enterprise toggle icon.
        self.icon = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSI+PHRpdGxlPkdlbWluaSBFbnRlcnByaXNlPC90aXRsZT48cGF0aCBkPSJNNCAyMFY4bDgtNCA4IDR2MTJNOCAyMHYtOGg4djhNMiAyMGgyMCIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+PC9zdmc+"
