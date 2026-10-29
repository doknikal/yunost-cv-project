const $ = id => document.getElementById(id);
const SOURCES = { scan: 'Сканы', photo: 'Фотографии' };
const COLORS = { scan: '#d33852', photo: '#6586ad' };
const num = x => Number(x).toLocaleString('ru-RU', { maximumFractionDigits: 4 });
const labels = {}, charts = {}, requests = {};

async function getJSON(url) {
  const response = await fetch(url);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Ошибка сервера ' + response.status);
  return data;
}

function fillTable(id, header, rows) {
  const row = (cells, tag) => '<tr>' + cells.map(c => `<${tag}>${c}</${tag}>`).join('') + '</tr>';
  $(id).innerHTML = row(header, 'th') + rows.map(r => row(r, 'td')).join('');
}

function axes(x, y, years = false) {
  return { responsive: true, maintainAspectRatio: false, animation: false,
           scales: { x: { type: 'linear', title: { display: true, text: x },
                          ticks: years ? { stepSize: 1, callback: String } : {} },
                     y: { title: { display: true, text: y } } } };
}

// Для каждого графика: какие параметры отправить, как нарисовать и что показать в таблице.
const CHARTS = {
  histogram: {
    params: () => ({ metric: $('histogram-metric').value, bins: $('bins').value }),
    draw: d => ({ type: 'bar', options: { ...axes(labels[d.metric], 'Страниц'), plugins: { legend: { display: false } } },
                  data: { datasets: [{ data: d.bars.map(b => ({ x: (b.left + b.right) / 2, y: b.count })),
                                       backgroundColor: '#b5142a', barPercentage: 1, categoryPercentage: 1 }] } }),
    table: d => [['От', 'До', 'Страниц'], d.bars.map(b => [num(b.left), num(b.right), b.count])],
  },
  trend: {
    params: () => ({ metric: $('trend-metric').value, aggregation: $('aggregation').value }),
    draw: d => ({ type: 'line', options: axes('Год', labels[d.metric], true),
                  data: { datasets: d.series.map(s => ({ label: SOURCES[s.source], borderColor: COLORS[s.source],
                                                         backgroundColor: COLORS[s.source],
                                                         data: s.points.map(p => ({ x: p.year, y: p.value })) })) } }),
    table: d => [['Источник', 'Год', d.aggregation === 'mean' ? 'Среднее' : 'Медиана', 'Страниц'],
                 d.series.flatMap(s => s.points.map(p => [SOURCES[s.source], p.year, num(p.value), p.pages]))],
  },
  scatter: {
    params: () => ({ x_metric: $('x-metric').value, y_metric: $('y-metric').value }),
    draw: d => ({ type: 'scatter', options: axes(labels[d.x_metric], labels[d.y_metric]),
                  data: { datasets: ['scan', 'photo'].map(s => ({ label: SOURCES[s], backgroundColor: COLORS[s],
                                                                  data: d.points.filter(p => p.source === s) })) } }),
    table: d => [['Страница', 'Год', 'Источник', 'X', 'Y'],
                 d.points.map(p => [p.file, p.year, SOURCES[p.source], num(p.x), num(p.y)])],
  },
};

async function refresh(kind) {
  const id = (requests[kind] || 0) + 1;
  requests[kind] = id;
  const params = new URLSearchParams({ year_min: $('year_min').value, year_max: $('year_max').value,
                                       source: $('source').value, ...CHARTS[kind].params() });
  try {
    const data = await getJSON(`/api/analytics/${kind}?${params}`);
    if (requests[kind] !== id) return;     // пришёл ответ на устаревший запрос
    charts[kind]?.destroy();
    $(kind + '-status').className = '';
    $(kind + '-status').textContent = `Страниц в выборке: ${data.selected_pages}, на графике: ${data.used_pages}` +
      (data.used_pages ? '' : ' — нет данных');
    charts[kind] = new Chart($(kind + '-chart'), CHARTS[kind].draw(data));
    fillTable(kind + '-table', ...CHARTS[kind].table(data));
  } catch (error) {
    if (requests[kind] !== id) return;
    charts[kind]?.destroy();
    $(kind + '-status').className = 'error';
    $(kind + '-status').textContent = error.message;
    $(kind + '-table').innerHTML = '';
  }
}

async function init() {
  const options = await getJSON('/api/analytics/options');
  const defaults = { 'histogram-metric': 'raggedness', 'trend-metric': 'raggedness', 'x-metric': 'n_columns', 'y-metric': 'raggedness' };
  for (const m of options.metrics) labels[m.key] = m.label;
  for (const select of document.querySelectorAll('.metric')) {
    select.innerHTML = options.metrics.map(m => `<option value="${m.key}">${m.label}</option>`).join('');
    select.value = defaults[select.id];
  }
  $('year_min').value = Math.min(...options.years);
  $('year_max').value = Math.max(...options.years);
  $('summary').textContent = `SQLite: ${options.pages} страниц, ${options.issues} номеров`;
  $('filters').addEventListener('change', () => Object.keys(CHARTS).forEach(refresh));
  for (const kind of Object.keys(CHARTS)) {
    $(kind + '-chart').closest('section').addEventListener('change', () => refresh(kind));
    refresh(kind);
  }
}

init().catch(error => { $('summary').textContent = error.message; $('summary').className = 'error'; });
