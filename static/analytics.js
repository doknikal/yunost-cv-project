'use strict';
const $ = id => document.getElementById(id);
const kinds = ['histogram', 'trend', 'scatter'];
const charts = {}, requests = {}, labels = {};
const sourceLabels = { scan: 'Сканы', photo: 'Фотографии' };
const number = value => Number(value).toLocaleString('ru-RU', { maximumFractionDigits: 6 });

async function jsonRequest(url, controller) {
  const timer = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(url, { signal: controller.signal });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Ошибка сервера (${response.status})`);
    return data;
  } finally { clearTimeout(timer); }
}

function clearChart(kind, state, message) {
  if (charts[kind]) { Plotly.purge($(kind + '-chart')); delete charts[kind]; }
  $(kind + '-panel').dataset.state = state;
  $(kind + '-panel').setAttribute('aria-busy', String(state === 'loading'));
  $(kind + '-status').textContent = message;
  $(kind + '-table').replaceChildren();
}

function table(kind, headers, rows) {
  const target = $(kind + '-table');
  const caption = document.createElement('caption');
  caption.textContent = 'Числовые данные текущего графика (SQLite)';
  const head = document.createElement('thead'), header = document.createElement('tr');
  for (const text of headers) {
    const th = document.createElement('th'); th.scope = 'col'; th.textContent = text; header.append(th);
  }
  head.append(header);
  const body = document.createElement('tbody');
  for (const values of rows) {
    const row = document.createElement('tr');
    for (const value of values) { const td = document.createElement('td'); td.textContent = value; row.append(td); }
    body.append(row);
  }
  target.replaceChildren(caption, head, body);
}

// Фигуру (данные и оформление) строит сервер на plotly (yunost/charts.py); здесь её только рисуем.
function draw(kind, figure) {
  const ink = getComputedStyle(document.documentElement).getPropertyValue('--ink').trim();
  const layout = { ...figure.layout, autosize: true, font: { ...(figure.layout.font || {}), color: ink } };
  Plotly.react($(kind + '-chart'), figure.data, layout,
    { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['select2d', 'lasso2d'] });
  charts[kind] = true;
}

function render(kind, data) {
  const state = data.used_pages ? 'ready' : 'empty';
  const count = `Выбрано страниц: ${data.selected_pages}; на графике: ${data.used_pages}; пропуски: ${data.missing_pages}.`;
  clearChart(kind, state, count + (data.used_pages ? '' : ' Нет данных для выбранных параметров.'));
  if (!data.used_pages) return;
  draw(kind, data.figure);
  if (kind === 'histogram') {
    table(kind, ['Левая граница', 'Правая граница', 'Страниц'], data.bars.map(b => [number(b.left), number(b.right), b.count]));
  } else if (kind === 'trend') {
    table(kind, ['Источник', 'Год', data.aggregation === 'mean' ? 'Среднее' : 'Медиана', 'Страниц', 'Номеров'],
      data.series.flatMap(s => s.points.map(p => [sourceLabels[s.source], p.year, number(p.value), p.pages, p.issues])));
  } else {
    table(kind, ['Страница', 'Год', 'Источник', labels[data.x_metric], labels[data.y_metric]],
      data.points.map(p => [p.file, p.year, sourceLabels[p.source], number(p.x), number(p.y)]));
  }
}

function selection() {
  const min = Number($('year-min').value), max = Number($('year-max').value);
  if (!$('year-min').value || !$('year-max').value || !Number.isInteger(min) || !Number.isInteger(max) || min < 1800 || max > 2200 || min > max) {
    throw new Error('Проверьте диапазон лет: от 1800 до 2200, начальный год не больше конечного.');
  }
  return new URLSearchParams({ year_min: min, year_max: max, source: $('source').value });
}

function cancel(kind) {
  if (requests[kind]) requests[kind].abort();
  delete requests[kind];
}

async function refresh(kind) {
  let params;
  try {
    params = selection();
    $('filters-error').hidden = true;
  } catch (error) {
    $('filters-error').hidden = false; $('filters-error').textContent = error.message;
    for (const key of kinds) { cancel(key); clearChart(key, 'error', 'Некорректный диапазон лет.'); }
    return;
  }
  cancel(kind);
  const controller = new AbortController(); requests[kind] = controller;
  clearChart(kind, 'loading', 'Читаем выборку из SQLite…');
  try {
    if (kind === 'histogram') {
      const bins = Number($('bins').value);
      if (!Number.isInteger(bins) || bins < 2 || bins > 60) throw new Error('Число интервалов должно быть целым от 2 до 60.');
      params.set('metric', $('histogram-metric').value); params.set('bins', bins);
    } else if (kind === 'trend') {
      params.set('metric', $('trend-metric').value); params.set('aggregation', $('aggregation').value);
    } else {
      params.set('x_metric', $('x-metric').value); params.set('y_metric', $('y-metric').value);
    }
    const data = await jsonRequest('/api/analytics/' + kind + '?' + params, controller);
    if (requests[kind] !== controller) return;
    if (!Number.isInteger(data.used_pages) || !Array.isArray(data.bars || data.series || data.points) || !data.figure) throw new Error('Некорректный ответ сервера');
    render(kind, data);
  } catch (error) {
    if (requests[kind] !== controller) return;
    clearChart(kind, 'error', error.name === 'AbortError' ? 'Нет ответа сервера. Повторите обновление.' : error.message);
  } finally {
    if (requests[kind] === controller) delete requests[kind];
  }
}

function refreshAll() { return Promise.all(kinds.map(refresh)); }

async function initialize() {
  try {
    if (typeof Plotly === 'undefined') throw new Error('Не загружен локальный plotly.js. Проверьте, что установлен пакет plotly (pip install -r requirements-app.txt).');
    const data = await jsonRequest('/api/analytics/options', new AbortController());
    if (!Array.isArray(data.metrics) || !Array.isArray(data.years) || !data.years.length) throw new Error('В базе нет данных корпуса');
    for (const metric of data.metrics) labels[metric.key] = metric.label;
    const defaults = { 'histogram-metric': 'raggedness', 'trend-metric': 'raggedness', 'x-metric': 'n_columns', 'y-metric': 'raggedness' };
    for (const select of document.querySelectorAll('[data-metric]')) {
      for (const metric of data.metrics) { const option = document.createElement('option'); option.value = metric.key; option.textContent = metric.label; select.append(option); }
      select.value = defaults[select.id];
    }
    $('year-min').value = Math.min(...data.years); $('year-max').value = Math.max(...data.years);
    $('corpus-summary').textContent = `SQLite · ${data.pages} страниц · ${data.issues} номеров · ${$('year-min').value}–${$('year-max').value}`;
    document.querySelectorAll('[data-controls]').forEach(fieldset => { fieldset.disabled = false; });
    $('filters-form').addEventListener('submit', event => { event.preventDefault(); refreshAll(); });
    $('filters').addEventListener('change', refreshAll);
    for (const kind of kinds) $(kind + '-controls').addEventListener('change', () => refresh(kind));
    await refreshAll();
  } catch (error) {
    $('corpus-summary').textContent = error.message; $('corpus-summary').className = 'error';
  }
}
initialize();
