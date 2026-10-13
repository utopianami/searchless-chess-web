"""Click every control and measure layout throughout real network transitions."""
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = os.getenv('CHESS_TEST_URL', 'https://deep-chess.ngrok.pro')
OUT = Path(__file__).resolve().parents[1] / 'test-results'
SAMPLE = '''() => {
  window.layoutSamples = [];
  window.recordLayout = true;
  function sample() {
    const result = {height: document.documentElement.scrollHeight, scroll: scrollY};
    for (const id of ['board-wrap', 'board-canvas', 'top-bar', 'mode-bar', 'model-bar', 'model-select', 'model-status', 'gpu-bar', 'cand-panel']) {
      const r = document.getElementById(id).getBoundingClientRect();
      result[id] = [r.x, r.y + scrollY, r.width, r.height];
    }
    window.layoutSamples.push(result);
    if (window.recordLayout) requestAnimationFrame(sample);
  }
  sample();
}'''


def settled(page):
    page.wait_for_function('!analysisPending && !isThinking')
    page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')


def measure(page, label, action):
    page.evaluate(SAMPLE)
    action()
    settled(page)
    samples = page.evaluate('() => {recordLayout = false; return layoutSamples;}')
    shifts = {}
    for key in samples[0]:
        values = [row[key] if isinstance(row[key], list) else [row[key]] for row in samples]
        delta = max(max(axis) - min(axis) for axis in zip(*values))
        if delta > 0.6:
            shifts[key] = round(delta, 2)
    print(label, 'shifts:', shifts, flush=True)
    assert not shifts, f'{label}: layout moved {shifts}'


def click(page, selector):
    locator = page.locator(selector)
    locator.scroll_into_view_if_needed()
    settled(page)
    measure(page, selector, locator.click)


def move(page, uci):
    page.locator('#board-canvas').scroll_into_view_if_needed()
    settled(page)
    def play():
        box = page.locator('#board-canvas').bounding_box()
        for square in [uci[:2], uci[2:4]]:
            file, rank = ord(square[0])-97, int(square[1])-1
            if page.evaluate('boardFlipped'):
                file, rank = 7-file, 7-rank
            page.mouse.click(box['x']+(file+.5)*box['width']/8,
                             box['y']+(7-rank+.5)*box['height']/8)
    previous = page.evaluate('historyIdx')
    measure(page, f'canvas {uci}', play)
    assert page.evaluate('historyIdx') > previous


def run():
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for width, height in [(1200, 900), (500, 800), (390, 844), (320, 650)]:
            page = browser.new_page(viewport={'width': width, 'height': height},
                                    extra_http_headers={'ngrok-skip-browser-warning': 'true'})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.set_default_timeout(120000)
            page.goto(BASE)
            page.wait_for_function('modelCatalog.length === 5 && !analysisPending && allMoveProbs.length === 20')
            page.wait_for_function('Object.values(pieceImages).length === 12 && Object.values(pieceImages).every(img => img.complete && img.naturalWidth > 0)')
            # Keep genuine API results; expose loading frames even on a fast cache hit.
            page.evaluate('''() => {
              const realFetch = window.fetch;
              window.fetch = async (...args) => {
                const response = await realFetch(...args);
                if (args[0] === '/api/analyse') await new Promise(resolve => setTimeout(resolve, 200));
                return response;
              };
            }''')
            print('VIEWPORT', width, height, flush=True)
            for selector in ['[data-mode="review"]', '[data-mode="ai"]', '[data-color="b"]', '[data-color="w"]']:
                click(page, selector)
            move(page, 'e2e4')
            for selector in ['#btn-first', '#btn-next', '#btn-last', '#btn-prev', '#btn-last']:
                click(page, selector)
            click(page, '[data-mode="review"]')
            move(page, 'e2e4')
            move(page, 'd7d5')
            move(page, 'e4d5')
            for selector in ['#btn-first', '#btn-next', '#btn-last', '#btn-prev', '#btn-last']:
                click(page, selector)
            for idx in range(page.locator('.cand-row').count()):
                click(page, f'.cand-row:nth-child({idx+1})')
            if width == 390:
                for model in ['9M', '136M', '270M', '9M_state_value', '9M_behavioral_cloning', '270M']:
                    page.locator('#model-select').scroll_into_view_if_needed()
                    measure(page, 'model ' + model, lambda m=model: page.select_option('#model-select', m))
                    assert page.evaluate('!apiError && allMoveProbs.length > 0')
                page.wait_for_selector('#btn-unload:enabled')
                click(page, '#btn-unload')
                page.wait_for_selector('#btn-retry:visible')
                click(page, '#btn-retry')
                assert page.evaluate('!apiError && allMoveProbs.length > 0')
                page.route('**/api/analyse', lambda route: route.fulfill(status=503, json={
                    'detail': 'GPU 메모리가 부족합니다. 더 작은 모델을 선택하거나 잠시 후 다시 시도해 주세요.'}))
                click(page, '[data-mode="ai"]')
                page.wait_for_selector('#btn-retry:visible')
                page.unroute('**/api/analyse')
                click(page, '#btn-retry')
                assert page.evaluate('!apiError && allMoveProbs.length > 0')
            assert not errors, errors
            page.evaluate('window.scrollTo(0, 0)')
            page.screenshot(path=str(OUT / f'layout-fixed-{width}.png'), full_page=True)
            page.close()
        browser.close()
    print('PASS: every button, all models, loading/error states, captures and viewport layouts stayed fixed.', flush=True)


if __name__ == '__main__':
    run()
