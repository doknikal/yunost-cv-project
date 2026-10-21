const $ = id => document.getElementById(id);
const drop = $('drop'), fileInput = $('file'), statusEl = $('status');
let uploadSequence = 0;
let edgeView = null;

function clearEdgeOverlay() {
  edgeView = null;
  $('show-edges').checked = false;
  $('show-edges').disabled = true;
  const canvas = $('edge-canvas');
  canvas.hidden = true;
  canvas.width = canvas.height = 0;
  $('overlay').removeAttribute('src');
  $('edge-note').textContent = 'Фиолетовые точки — правые края строк, найденные по изображению.';
}

function drawEdgeOverlay() {
  const canvas = $('edge-canvas'), toggle = $('show-edges');
  canvas.hidden = true;
  const view = edgeView;
  if (!view || view.sequence !== uploadSequence || !view.ready) return;
  const image = view.image;
  const size = view.pageSize;
  const validSize = Number.isFinite(size?.width) && size.width > 0 &&
    Number.isFinite(size?.height) && size.height > 0;
  const available = validSize && view.lines.length > 0;
  toggle.disabled = !available;
  $('edge-note').textContent = available ?
    'Фиолетовые точки — правые края строк, найденные по изображению.' :
    'Данные правых краёв строк недоступны. Распознанный текст показан без изменений.';
  if (!available || !toggle.checked) return;
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext('2d');
  if (!context) return;
  // The API uses full page coordinates; the overlay PNG may be downscaled.
  const scaleX = image.naturalWidth / size.width;
  const scaleY = image.naturalHeight / size.height;
  context.fillStyle = '#9c27b0';
  context.strokeStyle = '#ffffff';
  context.lineWidth = Math.max(1, image.naturalWidth / 900);
  const radius = Math.max(3, image.naturalWidth / 220);
  for (const line of view.lines) {
    context.beginPath();
    context.arc(line.right * scaleX, line.center_y * scaleY, radius, 0, 2 * Math.PI);
    context.fill();
    context.stroke();
  }
  canvas.hidden = false;
}

function showOverlay(data) {
  const previous = $('overlay');
  // A distinct image per response keeps late load/error events from an older
  // upload from drawing old coordinates on the current page.
  const image = new Image();
  image.id = previous.id;
  image.alt = previous.alt;
  const blocks = Array.isArray(data.edge_analysis?.blocks) ? data.edge_analysis.blocks : [];
  const lines = blocks.flatMap(block => Array.isArray(block.lines) ? block.lines : [])
    .filter(line => Number.isFinite(line.right) && Number.isFinite(line.center_y));
  const view = { image, lines, pageSize: data.page_size, sequence: uploadSequence, ready: false };
  edgeView = view;
  $('show-edges').checked = false;
  $('show-edges').disabled = true;
  $('edge-canvas').hidden = true;
  image.addEventListener('load', () => {
    if (edgeView !== view || view.sequence !== uploadSequence) return;
    view.ready = image.naturalWidth > 0 && image.naturalHeight > 0;
    drawEdgeOverlay();
  });
  image.addEventListener('error', () => {
    if (edgeView !== view || view.sequence !== uploadSequence) return;
    view.ready = false;
    $('show-edges').checked = false;
    $('show-edges').disabled = true;
    $('edge-canvas').hidden = true;
    $('edge-note').textContent = 'Не удалось загрузить разметку изображения. Текст доступен справа.';
  });
  previous.replaceWith(image);
  image.src = data.overlay;
}

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
  clearEdgeOverlay();
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
  showOverlay(data);
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
  const edgeBlocks = new Map();
  if (Array.isArray(data.edge_analysis?.blocks)) {
    for (const block of data.edge_analysis.blocks) {
      if (Number.isInteger(block.block_number) && block.block_number > 0) {
        edgeBlocks.set(block.block_number, block);
      }
    }
  }
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
    const edge = edgeBlocks.get(i + 1);
    const geometry = document.createElement('small');
    const count = Number.isInteger(edge?.line_count) && edge.line_count >= 0 ? edge.line_count : 'нет данных';
    const value = Number.isFinite(edge?.value) ? edge.value.toFixed(4) : 'нет оценки';
    geometry.textContent = `Строк CV: ${count}; неровность правого края (σ/ширина блока с полями): ${value}`;
    item.append(title, document.createElement('br'), meta, document.createElement('br'), geometry, text);
    list.append(item);
  });
  $('diagnostics').hidden = false;
}

fileInput.addEventListener('change', () => upload(fileInput.files[0]));
$('show-edges').addEventListener('change', drawEdgeOverlay);
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
  a.download = 'yunost-text.txt';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});
