from __future__ import annotations

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from paint_review.ui import PaintReviewWindow


def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("Layered Paint Review")
    app.setOrganizationName("PaintReview")
    app.setStyle("Fusion")
    window = PaintReviewWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

