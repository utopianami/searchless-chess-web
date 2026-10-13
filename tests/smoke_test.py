"""Live end-to-end verification: python tests/smoke_test.py [base URL]."""
import json
from pathlib import Path
import sys
import time

import chess
import httpx
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8888'
OUT = Path(__file__).resolve().parents[1] / 'test-results'
OUT.mkdir(exist_ok=True)
HEADERS = {'ngrok-skip-browser-warning': 'true'}


def api_checks():
    timings = {}
    with httpx.Client(base_url=BASE, timeout=120, headers=HEADERS) as client:
        catalog = client.get('/api/models').json()
        assert len(catalog['models']) == 5
        for model in catalog['models']:
            assert model['available']
            board = chess.Board()
            samples = []
            for _ in range(4):
                start = time.monotonic()
                response = client.post('/api/analyse', json={'fen': board.fen(), 'model': model['id']})
                response.raise_for_status()
                data = response.json()
                samples.append(round(time.monotonic() - start, 3))
                assert data['model'] == model['id']
                assert {m['uci'] for m in data['moves']} == {m.uci() for m in board.legal_moves}
                assert all(0 <= m['prob'] <= 1 for m in data['moves'])
                if model['policy'] == 'behavioral_cloning':
                    assert data['white_win_prob'] is None
                    assert abs(sum(m['prob'] for m in data['moves']) - 1) < 1e-5
                else:
                    assert 0 <= data['white_win_prob'] <= 1
                move = client.post('/api/move', json={'fen': board.fen(), 'model': model['id']}).json()
                assert move['uci'] == data['moves'][0]['uci']
                board.push_uci(move['uci'])
            timings[model['id']] = samples
            print('GPU/API', model['id'], samples, flush=True)
    return timings


def browser_checks():
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(viewport={'width': 1200, 'height': 1000}, extra_http_headers=HEADERS)
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(BASE)
        page.set_default_timeout(120000)
        page.wait_for_function('modelCatalog.length === 5 && !analysisPending && allMoveProbs.length === 20')
        assert page.locator('#model-select').input_value() == '270M'
        for model in ['9M', '136M', '270M', '9M_state_value', '9M_behavioral_cloning']:
            page.select_option('#model-select', model)
            page.wait_for_function('!analysisPending && allMoveProbs.length === 20')
            assert not page.locator('#btn-retry').is_visible()
            assert page.locator('#eval-bar').is_visible() == (model != '9M_behavioral_cloning')
        assert '수 선택 확률' in page.locator('#cand-header').inner_text()
        page.reload()
        page.wait_for_function('!analysisPending && allMoveProbs.length === 20')
        assert page.locator('#model-select').input_value() == '9M_behavioral_cloning'
        page.select_option('#model-select', '270M')
        page.wait_for_function('!analysisPending && allMoveProbs.length === 20')
        # A real canvas move, followed by the model's reply.
        box = page.locator('#board-canvas').bounding_box()
        for file, rank in [(4, 1), (4, 3)]:
            page.mouse.click(box['x']+(file+0.5)*box['width']/8, box['y']+(7-rank+0.5)*box['height']/8)
        page.wait_for_function('historyIdx === 2 && !isThinking && !analysisPending')
        assert chess.Board(page.evaluate('currentFEN()')).is_valid()
        page.click('#btn-first')
        page.wait_for_function('historyIdx === 0 && !analysisPending')
        page.click('[data-mode="review"]')
        page.wait_for_function('!analysisPending')
        page.select_option('#model-select', '9M')
        page.select_option('#model-select', '270M')
        page.wait_for_function('selectedModel === "270M" && !analysisPending')
        # Deliberately deliver an obsolete model response after the latest one.
        page.evaluate('''() => {
          window.savedFetch = window.fetch;
          window.oldResponseDelivered = false;
          window.fetch = (url, options) => {
            if (url === '/api/analyse' && JSON.parse(options.body).model === '9M') {
              return new Promise(resolve => setTimeout(() => {
                window.oldResponseDelivered = true;
                resolve(new Response(JSON.stringify({model: '9M', moves: [], game_over: false,
                  white_win_prob: 0.123, score_type: 'win_probability'})));
              }, 1200));
            }
            return window.savedFetch(url, options);
          };
        }''')
        page.select_option('#model-select', '9M')
        page.select_option('#model-select', '270M')
        page.wait_for_function('window.oldResponseDelivered && !analysisPending')
        assert page.evaluate('whiteWinProb !== 0.123 && allMoveProbs.length === 20')
        page.evaluate('window.fetch = window.savedFetch')
        # Compare browser FEN updates with python-chess for rights/counters.
        for fen, move in [
            (chess.STARTING_FEN, 'g1f3'),
            ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', 'a1a8'),
            ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', 'e1g1'),
            ('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1', 'e5d6'),
            ('4k3/P7/8/8/8/8/8/4K3 w - - 0 1', 'a7a8q'),
        ]:
            expected = chess.Board(fen)
            expected.push_uci(move)
            actual = page.evaluate('([fen, move]) => applyMoveToFEN(fen, move)', [fen, move])
            assert actual == expected.fen(en_passant='fen'), (actual, expected.fen())
        page.wait_for_function('!document.getElementById("gpu-status").textContent.includes("준비 중")')
        page.screenshot(path=str(OUT / 'desktop.png'), full_page=True)
        page.set_viewport_size({'width': 390, 'height': 844})
        page.screenshot(path=str(OUT / 'mobile.png'), full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        # Explicit failure and recovery, not just a happy-path screenshot.
        page.route('**/api/analyse', lambda route: route.fulfill(status=503, json={'detail': 'test failure'}))
        page.select_option('#model-select', '9M')
        page.wait_for_selector('#btn-retry:visible')
        page.unroute('**/api/analyse')
        page.click('#btn-retry')
        page.wait_for_function('!analysisPending && allMoveProbs.length === 20 && !apiError')
        page.wait_for_selector('#btn-unload:enabled')
        page.click('#btn-unload')
        page.wait_for_function('document.getElementById("gpu-status").textContent.includes("GPU 해제됨")')
        assert not page.request.get(BASE + '/api/gpu').json()['loaded']
        page.select_option('#model-select', '270M')
        page.wait_for_function('!analysisPending && allMoveProbs.length === 20 && !apiError')
        assert page.request.get(BASE + '/api/gpu').json()['loaded']
        assert not errors, errors
        browser.close()
        print('Browser: five models, persistence, game, stale responses, special moves, navigation, mobile, error recovery, GPU release/reload passed', flush=True)


if __name__ == '__main__':
    print(json.dumps(api_checks(), indent=2))
    browser_checks()
