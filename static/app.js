const $ = id => document.getElementById(id);
const drop = $('drop'), fileInput = $('file'), statusEl = $('status');
let uploadSequence = 0;
let current = null;  // последний ответ /api/ocr — по нему строится разбор «проза / стихи»

function setStatus(msg, kind) {
  statusEl.className = 'status' + (kind ? ' ' + kind : '');
  statusEl.replaceChildren();
  if (kind === 'busy') {
    const spinner = document.createElement('span');
    spinner.className = 'spinner';
    statusEl.append(spinner);
  }
  statusEl.append(document.createTextNode(msg));
}

async function upload(file) {
  if (!file) return;
  const sequence = ++uploadSequence;
  fileInput.value = ''; // Allow selecting the same file again after an error.
  $('result').classList.remove('show');
  $('diagnostics').hidden = true;
  $('text').value = '';
  resetGenre();
  if (file.size > 25 * 1024 * 1024) {
    setStatus('Файл больше 25 МБ', 'error');
    return;
  }
  setStatus('Распознаём страницу… это может занять до 90 секунд', 'busy');
  const form = new FormData();
  form.append('file', file);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 100000);
  try {
    const resp = await fetch('/api/ocr', { method: 'POST', body: form, signal: controller.signal });
    const data = await resp.json().catch(() => ({}));
    if (sequence !== uploadSequence) return;
    if (!resp.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Ошибка сервера (' + resp.status + ')');
    if (typeof data.text !== 'string' || !Array.isArray(data.blocks)) throw new Error('Некорректный ответ сервера');
    show(data);
  } catch (err) {
    if (sequence !== uploadSequence) return;
    setStatus(err.name === 'AbortError' ? 'Сервер не ответил вовремя. Повторите загрузку.' :
      (err.message || 'Не удалось связаться с сервером'), 'error');
  } finally {
    clearTimeout(timer);
  }
}

function show(data) {
  current = data;
  $('overlay').src = data.overlay;
  $('genre').classList.toggle('show', data.blocks.length > 0);
  $('text').value = data.text;
  const note = $('note');
  note.hidden = false;
  if (data.text_blocks === 0) {
    note.textContent = 'Текст не найден. Это не гарантирует его отсутствия: попробуйте более чёткий снимок или отдельный фрагмент.';
    setStatus('Текст не найден', '');
  } else {
    note.textContent = 'Автоматическая выборка может быть неполной. Сверьте начало и конец каждой колонки с оригиналом; число найденных блоков не показывает полноту.';
    setStatus('Найдено текстовых блоков: ' + data.text_blocks, '');
  }
  $('result').classList.add('show');
  const list = $('block-list');
  list.replaceChildren();
  (data.all_blocks || data.blocks).forEach((block, i) => {
    const item = document.createElement('div');
    item.className = 'block-check';
    const title = document.createElement('strong');
    title.textContent = `Блок ${i + 1}: ${block.is_text ? 'текст' : (block.rejected || 'сомнительные знаки')}`;
    const meta = document.createElement('small');
    meta.textContent = `Строк OCR: ${block.lines}; уверенность: ${block.confidence ?? 'нет данных'} / 100; ` +
      `рамка: ${block.x}, ${block.y}, ${block.w} × ${block.h}; ` +
      (block.source === 'sparse' ? 'дополнительный поиск короткого текста' : 'разметка страницы') +
      (block.refined_words ? `; перечитано слов: ${block.refined_words}` : '');
    const text = document.createElement('pre');
    text.textContent = block.text || 'Текст не прочитан.';
    item.append(title, document.createElement('br'), meta, text);
    list.append(item);
  });
  $('diagnostics').hidden = false;
}

fileInput.addEventListener('change', () => upload(fileInput.files[0]));
['dragenter', 'dragover'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', ev => upload(ev.dataTransfer.files[0]));

$('copy').addEventListener('click', async () => {
  const btn = $('copy'), text = $('text').value;
  try {
    try { await navigator.clipboard.writeText(text); }
    catch {
      $('text').select();
      if (!document.execCommand('copy')) throw new Error('copy failed');
    }
    btn.textContent = 'Скопировано';
    setTimeout(() => btn.textContent = 'Скопировать', 1500);
  } catch {
    setStatus('Не удалось скопировать. Выделите текст и нажмите Ctrl+C (на Mac ⌘C).', 'error');
  }
});
$('download').addEventListener('click', () => {
  const blob = new Blob([$('text').value], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'raspoznannyj-tekst.txt';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

// ---------- дополнительно: разбор «проза / стихи» по рваности правого края ----------
const SVG = 'http://www.w3.org/2000/svg';

function resetGenre() {
  current = null;
  $('genre').classList.remove('show');
  $('genre-result').hidden = true;
  $('genre-status').textContent = '';
  $('genre-boxes').replaceChildren();
  $('genre-run').disabled = false;
}

function downloadText(text, name) {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function drawGenreBoxes(blocks, verdicts) {
  const svg = $('genre-boxes');
  svg.setAttribute('viewBox', `0 0 ${current.page_size.width} ${current.page_size.height}`);
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.replaceChildren();
  verdicts.forEach((v, i) => {
    const b = blocks[i];
    const rect = document.createElementNS(SVG, 'rect');
    Object.entries({ x: b.x, y: b.y, width: b.w, height: b.h, rx: 6 }).forEach(([k, val]) => rect.setAttribute(k, val));
    rect.setAttribute('class', 'kind-' + v.kind);
    const label = document.createElementNS(SVG, 'text');
    label.setAttribute('x', b.x + 8); label.setAttribute('y', b.y + 42);
    label.setAttribute('class', 'kind-' + v.kind);
    label.textContent = i + 1;
    svg.append(rect, label);
  });
}

function renderGenre(result) {
  const blocks = current.blocks, verdicts = result.blocks;
  const labels = { prose: 'проза', verse: 'стихи', short: 'короткие фрагменты', unknown: 'не определено' };
  const summary = $('genre-summary');
  summary.replaceChildren();
  for (const kind of ['prose', 'verse', 'short', 'unknown']) {
    if (!result.counts[kind]) continue;
    const item = document.createElement('span');
    item.className = 'kind-' + kind;
    item.append(document.createElement('i'), document.createTextNode(`${labels[kind]}: ${result.counts[kind]}`));
    item.style.fill = 'none';
    summary.append(item);
  }

  const downloads = $('genre-downloads');
  downloads.replaceChildren();
  for (const [kind, name, title] of [['prose', 'proza.txt', 'Скачать прозу .txt'], ['verse', 'stihi.txt', 'Скачать стихи .txt']]) {
    const texts = verdicts.filter(v => v.kind === kind).map(v => blocks[v.index].text);
    if (!texts.length) continue;
    const button = document.createElement('button');
    button.textContent = title;
    button.addEventListener('click', () => downloadText(texts.join('\n\n'), name));
    downloads.append(button);
  }

  const list = $('genre-list');
  list.replaceChildren();
  for (const v of verdicts) {
    const row = document.createElement('div');
    row.className = 'genre-row';
    const number = document.createElement('strong');
    number.textContent = v.index + 1;
    const kind = document.createElement('span');
    kind.className = 'kind-' + v.kind;
    kind.style.fill = 'none';
    kind.append(document.createElement('i'), document.createTextNode(v.label));
    const why = document.createElement('span');
    why.className = 'why';
    if (v.raggedness !== null) {
      const bar = document.createElement('span'), fill = document.createElement('b');
      bar.className = 'bar ' + 'kind-' + v.kind; bar.style.fill = 'none';
      fill.style.width = Math.round(v.raggedness * 100) + '%';
      bar.append(fill);
      why.append(document.createTextNode('рваность края ' + v.raggedness.toLocaleString('ru-RU', { maximumFractionDigits: 2 })), bar);
    }
    why.append(document.createTextNode(`строк: ${v.lines} — ${v.reason}`));
    const first = document.createElement('span');
    first.className = 'first';
    first.textContent = (blocks[v.index].text || '').trim().split('\n')[0];
    row.append(number, kind, why, first);
    list.append(row);
  }
  drawGenreBoxes(blocks, verdicts);
  $('genre-result').hidden = false;
}

$('genre-run').addEventListener('click', async () => {
  if (!current) return;
  const data = current, button = $('genre-run');
  button.disabled = true;
  $('genre-status').className = 'status';
  $('genre-status').textContent = 'Разбираем блоки…';
  try {
    const resp = await fetch('/api/genre', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ blocks: data.blocks.map(b => ({ line_boxes: b.line_boxes })) }),
    });
    const result = await resp.json().catch(() => ({}));
    if (current !== data) return;  // пока ждали ответ, загрузили другую страницу
    if (!resp.ok || !Array.isArray(result.blocks)) throw new Error(typeof result.detail === 'string' ? result.detail : 'Ошибка сервера (' + resp.status + ')');
    $('genre-status').textContent = 'Цвет рамок на странице слева — тип блока; номера совпадают со списком.';
    renderGenre(result);
  } catch (err) {
    if (current !== data) return;
    $('genre-status').className = 'status error';
    $('genre-status').textContent = err.message || 'Не удалось разобрать блоки';
  } finally {
    if (current === data) button.disabled = false;
  }
});
