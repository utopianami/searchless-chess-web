"""Fast API contract checks; real GPU/browser checks live in smoke_test.py."""
import chess
import pytest
from fastapi.testclient import TestClient

import server

client = TestClient(server.app)


def test_model_catalog():
    data = client.get('/api/models').json()
    assert data['default_model'] == '270M'
    assert len(data['models']) == 5


@pytest.mark.parametrize('endpoint', ['/api/analyse', '/api/move'])
@pytest.mark.parametrize('payload', [
    {'fen': 'not a FEN'},
    {'fen': '8/8/8/8/8/8/8/8 w - - 0 1'},
    {'fen': chess.STARTING_FEN, 'model': '../../other'},
    {'fen': 'a' * 201},
])
def test_rejects_invalid_requests(endpoint, payload):
    assert client.post(endpoint, json=payload).status_code == 422


def test_model_selection_is_per_request(monkeypatch):
    calls = []
    def analyse(model, board):
        calls.append((model, board.fen()))
        return {'model': model, 'game_over': False,
                'moves': [{'uci': 'e2e4', 'from': 'e2', 'to': 'e4'}]}
    monkeypatch.setattr(server.runtime, 'analyse', analyse)
    for model in ['9M', '270M', '9M_state_value']:
        result = client.post('/api/move', json={'fen': chess.STARTING_FEN, 'model': model})
        assert result.json()['model'] == model
    assert [c[0] for c in calls] == ['9M', '270M', '9M_state_value']


def test_missing_model_returns_recoverable_error(monkeypatch):
    def missing(*args):
        raise FileNotFoundError('missing checkpoint')
    monkeypatch.setattr(server.runtime, 'analyse', missing)
    assert client.post('/api/analyse', json={'fen': chess.STARTING_FEN}).status_code == 503


@pytest.mark.parametrize('model', list(server.MODEL_SPECS))
def test_terminal_position_without_loading_model(model):
    board = chess.Board()
    for move in ['f2f3', 'e7e5', 'g2g4', 'd8h4']:
        board.push_uci(move)
    result = client.post('/api/analyse', json={'fen': board.fen(), 'model': model}).json()
    assert result['game_over'] is True
    assert result['white_win_prob'] == 0
    assert result['moves'] == []
