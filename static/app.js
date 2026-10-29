const $ = id => document.getElementById(id);
let page = null;      // последний ответ /api/ocr
let uploads = 0;      // чтобы поздний ответ старой загрузки не перезаписал новую

function setStatus(text, error = false) {
  $('status').textContent = text;
  $('status').className = error ? 'error' : '';
}

async function upload(file) {
  if (!file) return;
  const id = ++uploads;
  page = null;
  $('result').hidden = $('diagnostics').hidden = true;
  $('show-edges').checked = false;
  drawEdges();
  if (file.size > 25 * 1024 * 1024) return setStatus('Файл больше 25 МБ', true);
  setStatus('Распознаём страницу… это может занять до 90 секунд');
  const form = new FormData();
  form.append('file', file);
  try {
    const response = await fetch('/api/ocr', { method: 'POST', body: form });
    const data = await response.json().catch(() => ({}));
    if (id !== uploads) return;
    if (!response.ok) throw new Error(data.detail || 'Ошибка сервера ' + response.status);
    show(data);
  } catch (error) {
    if (id === uploads) setStatus(error.message, true);
  }
}

function show(data) {
  page = data;
  $('overlay').src = data.overlay;
  $('text').value = data.text;
  setStatus(data.text_blocks ? 'Найдено текстовых блоков: ' + data.text_blocks : 'Текст не найден');
  const edges = new Map((data.edge_analysis?.blocks || []).map(b => [b.block_number, b]));
  $('show-edges').disabled = !edges.size;
  $('block-list').replaceChildren(...data.all_blocks.map((block, i) => {
    const edge = edges.get(i + 1);
    const item = document.createElement('div');
    const title = document.createElement('h4');
    const info = document.createElement('small');
    const text = document.createElement('pre');
    title.textContent = `Блок ${i + 1}: ${block.is_text ? 'текст' : block.rejected || 'сомнительные знаки'}`;
    info.textContent = `строк: ${block.lines}, уверенность OCR: ${block.confidence}` +
      (edge?.value != null ? `, неровность правого края: ${edge.value.toFixed(3)}` : '');
    text.textContent = block.text;
    item.append(title, info, text);
    return item;
  }));
  $('result').hidden = $('diagnostics').hidden = false;
}

// Фиолетовые точки на концах строк. Координаты сервера — в пикселях страницы,
// картинка с рамками может быть уменьшена, поэтому пересчитываем масштаб.
function drawEdges() {
  const canvas = $('edge-canvas'), image = $('overlay');
  canvas.hidden = !(page && $('show-edges').checked && image.naturalWidth);
  if (canvas.hidden) return;
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const sx = canvas.width / page.page_size.width, sy = canvas.height / page.page_size.height;
  const context = canvas.getContext('2d');
  context.fillStyle = '#9c27b0';
  for (const block of page.edge_analysis.blocks) {
    for (const line of block.lines) {
      context.beginPath();
      context.arc(line.right * sx, line.center_y * sy, canvas.width / 220 + 2, 0, 2 * Math.PI);
      context.fill();
    }
  }
}

$('file').addEventListener('change', () => upload($('file').files[0]));
$('drop').addEventListener('dragover', event => event.preventDefault());
$('drop').addEventListener('drop', event => { event.preventDefault(); upload(event.dataTransfer.files[0]); });
$('show-edges').addEventListener('change', drawEdges);
$('overlay').addEventListener('load', drawEdges);
$('copy').addEventListener('click', () => navigator.clipboard.writeText($('text').value)
  .then(() => setStatus('Текст скопирован'), () => setStatus('Не удалось скопировать: выделите текст и нажмите ⌘C', true)));
$('download').addEventListener('click', () => {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([$('text').value], { type: 'text/plain;charset=utf-8' }));
  link.download = 'text.txt';
  link.click();
});
