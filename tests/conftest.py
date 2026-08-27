"""Shared fixtures for the viewer's test suite.

`qapp` is session-scoped: PyQt5 only allows one QApplication per process,
and creating vispy SceneCanvas/QWidget instances (MeshEditorPanel,
GaussianViewer, MainWindow) requires one to already exist -- exactly the
same requirement the app itself has (see src/viewer/app.py's `main()`).
"""

import sys

import pytest
from PyQt5 import QtWidgets
from vispy import app as vispy_app


@pytest.fixture(scope="session")
def qapp():
    vispy_app.use_app("pyqt5")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    yield app
