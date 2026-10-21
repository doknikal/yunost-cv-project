"""Installer/launcher regressions without downloading anything or touching .venv."""
import hashlib
import socket

import pytest

import run_app
import setup_app


def test_foreign_environment_is_not_replaced(tmp_path, monkeypatch):
    environment = tmp_path / '.venv-app'
    environment.mkdir()
    sentinel = environment / 'my-data'
    sentinel.write_text('preserve')
    monkeypatch.setattr(setup_app, 'HERE', tmp_path)
    monkeypatch.setattr(setup_app, 'check_tesseract', lambda *a: 'fixture')
    monkeypatch.setattr(setup_app, 'ensure_weights', lambda: tmp_path)
    with pytest.raises(RuntimeError, match='уже существует'):
        setup_app.main()
    assert sentinel.read_text() == 'preserve'
    assert not (environment / '.yunost-app').exists()


def test_wrong_model_is_rejected_without_modifying_it(tmp_path):
    model = tmp_path / 'rus.traineddata'
    model.write_bytes(b'untrusted model')
    before = model.read_bytes()
    with pytest.raises(RuntimeError, match='контрольная сумма'):
        setup_app.verify_weights(model)
    assert model.read_bytes() == before


def test_known_model_checksum_is_accepted(tmp_path, monkeypatch):
    model = tmp_path / 'rus.traineddata'
    model.write_bytes(b'fixture')
    monkeypatch.setattr(setup_app, 'WEIGHTS_SHA256', hashlib.sha256(b'fixture').hexdigest())
    setup_app.verify_weights(model)


def test_missing_tesseract_has_actionable_error(monkeypatch):
    monkeypatch.setattr(setup_app.shutil, 'which', lambda _: None)
    with pytest.raises(RuntimeError, match='brew install tesseract'):
        setup_app.check_tesseract()


def test_free_port_is_accepted():
    run_app.check_port(0)


def test_listening_port_is_not_mistaken_for_stopped_server():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        with pytest.raises(RuntimeError, match='занят'):
            run_app.check_port(listener.getsockname()[1])


def test_port_probe_uses_same_reuse_semantics_as_uvicorn(monkeypatch):
    events = []
    class Socket:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def setsockopt(self, *args):
            events.append(args)
        def bind(self, address):
            assert events == [(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)]
            assert address == ('127.0.0.1', 8000)
    monkeypatch.setattr(run_app.socket, 'socket', Socket)
    run_app.check_port(8000)
