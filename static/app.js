const $ = id => document.getElementById(id);
const drop = $('drop'), fileInput = $('file'), statusEl = $('status');
let uploadSequence = 0;

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
  $('overlay').src = data.overlay;
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
  a.download = 'yunost-text.txt';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});
