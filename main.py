"""
DrowsyGuard AI - application entry point.

Kept as a thin wrapper so the project has a single obvious start file
(``python main.py``). The dashboard can still be started directly.
"""
from dashboard import Dashboard


def main():
    Dashboard().mainloop()


if __name__ == "__main__":
    main()
