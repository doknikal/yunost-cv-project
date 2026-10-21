'use strict';
const $ = id => document.getElementById(id);
const kinds = ['histogram', 'trend', 'scatter'];
const charts = {}, requests = {}, labels = {};
const sourceLabels = { scan: 'Сканы', photo: 'Фотографии' };
const colors = { scan: '#d33852', photo: '#6586ad' };
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
  if (charts[kind]) { charts[kind].destroy(); delete charts[kind]; }
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

function chartOptions(xTitle, yTitle, linearX = false) {
  const ink = getComputedStyle(document.documentElement).getPropertyValue('--ink').trim();
  return {
    responsive: true, maintainAspectRatio: false, animation: false, color: ink,
    plugins: { legend: { labels: { color: ink } } },
    scales: {
      x: { ...(linearX ? { type: 'linear' } : {}), title: { display: true, text: xTitle, color: ink }, ticks: { color: ink } },
      y: { title: { display: true, text: yTitle, color: ink }, ticks: { color: ink } }
    }
  };
}

function render(kind, data) {
  const state = data.used_pages ? 'ready' : 'empty';
  const count = `Выбрано страниц: ${data.selected_pages}; на графике: ${data.used_pages}; пропуски: ${data.missing_pages}.`;
  clearChart(kind, state, count + (data.used_pages ? '' : ' Нет данных для выбранных параметров.'));
  if (!data.used_pages) return;
  if (kind === 'histogram') {
    const options = chartOptions(labels[data.metric], 'Число страниц');
    options.plugins.legend.display = false;
    options.scales.y.beginAtZero = true;
    options.scales.y.ticks.precision = 0;
    options.plugins.tooltip = { callbacks: { title: items => {
      const bar = data.bars[items[0].dataIndex]; return `${number(bar.left)} — ${number(bar.right)}`;
    } } };
    charts[kind] = new Chart($(kind + '-chart'), {
      type: 'bar', data: { labels: data.bars.map(b => `${Number(b.left.toPrecision(3))}–${Number(b.right.toPrecision(3))}`),
        datasets: [{ label: 'Страниц', data: data.bars.map(b => b.count), backgroundColor: '#b5142ab8' }] }, options
    });
    table(kind, ['Левая граница', 'Правая граница', 'Страниц'], data.bars.map(b => [number(b.left), number(b.right), b.count]));
  } else if (kind === 'trend') {
    const options = chartOptions('Год', labels[data.metric], true);
    options.scales.x.ticks.precision = 0;
    options.scales.x.ticks.callback = value => String(value);
    options.plugins.tooltip = { callbacks: { label: item => `${item.dataset.label}: ${number(item.raw.y)}; страниц: ${item.raw.pages}; номеров: ${item.raw.issues}` } };
    charts[kind] = new Chart($(kind + '-chart'), {
      type: 'line', data: { datasets: data.series.map(s => ({ label: sourceLabels[s.source],
        data: s.points.map(p => ({ x: p.year, y: p.value, pages: p.pages, issues: p.issues })),
        borderColor: colors[s.source], backgroundColor: colors[s.source], pointRadius: 5, spanGaps: false })) }, options
    });
    table(kind, ['Источник', 'Год', data.aggregation === 'mean' ? 'Среднее' : 'Медиана', 'Страниц', 'Номеров'],
      data.series.flatMap(s => s.points.map(p => [sourceLabels[s.source], p.year, number(p.value), p.pages, p.issues])));
  } else {
    const options = chartOptions(labels[data.x_metric], labels[data.y_metric], true);
    options.plugins.tooltip = { callbacks: { title: items => `${items[0].raw.file} (${items[0].raw.year})`,
      label: item => `${labels[data.x_metric]}: ${number(item.raw.x)}; ${labels[data.y_metric]}: ${number(item.raw.y)}` } };
    charts[kind] = new Chart($(kind + '-chart'), {
      type: 'scatter', data: { datasets: ['scan', 'photo'].filter(s => data.points.some(p => p.source === s)).map(source => ({
        label: sourceLabels[source], data: data.points.filter(p => p.source === source), backgroundColor: colors[source], pointRadius: 4
      })) }, options
    });
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
    if (!Number.isInteger(data.used_pages) || !Array.isArray(data.bars || data.series || data.points)) throw new Error('Некорректный ответ сервера');
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
    if (typeof Chart === 'undefined') throw new Error('Не загружен локальный Chart.js. Проверьте файл static/vendor/chart.umd.js.');
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
