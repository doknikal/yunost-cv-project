# Visual prose/verse review

## Scope and method

`page_labels.json` contains **26 visually reviewed development pages**: 18 `hasprose`, 8 `pureverse`. It is a regression fixture, **not held-out evaluation data** or an OCR transcription benchmark.

**Final audit, 2026-10-20:** every one of the 26 listed `data/norm/<file>` paths was opened individually as a full image with `functions.read`. Earlier contact-sheet interpretations contained incorrect author/title associations and, in several cases, incorrect prose locations. Those interpretations are superseded by revision 2 of the JSON. Main-session corrections were checked against the exact source images. Temporary contact sheets/grids under `.venv/evaluation/` are not annotation evidence and are not needed to use the fixture.

`data/manifest.csv` supplies the scan/photo category. PIL verifies each PNG's dimensions. Filenames identify **normalized page indices**, not printed magazine folios. No production code or tests were edited.

Visual evidence—not recognized text or the detector's `is_prose` decisions—determines the labels. Paragraph layout, sentence continuity, poetic lineation, stanza spacing and readable content were considered together. In particular, a poetry page **with an editorial or biographical introduction is `hasprose`**. A portrait, heading, author name, dedication or illustration alone is not prose.

## Annotation contract

- `expected_min_prose_blocks`/`expected_max_prose_blocks` express page presence: 0/0 on pure verse, at least 1 on prose-positive pages. Positive maximum is `null` because valid OCR segmentation may split/merge a passage.
- `regions` are approximate manually drawn `[x0, y0, x1, y1]` rectangles in the **original normalized PNG**, not resized sheet coordinates. Allow approximately 30 pixels of boundary tolerance.
- Each prose rectangle is an independent recall target. Detecting one prose column does not establish recall of the other columns or a separate introduction. For detailed scoring, inspect coverage across each region rather than merely touching its edge.
- Verse rectangles are exclusion targets. Whitespace/headings may be enclosed; these are not exact pixel masks, and regions are not exhaustive. Do not infer “everything outside a prose rectangle is verse.” Pure-verse pages have a strict page-level zero target without pretending that portraits are verse pixels.
- Short explanatory footnotes and translation credits are not treated as independent running-prose blocks. Their presence is noted where relevant. Editorial introductions and substantive explanatory notes do count.
- Bounds were checked against full images, not inferred from accepted detector boxes. For example, prose on `1957-12_p06.png` continues to about y770.

## Full-image audit: verse-only pages

The following baseline-positive candidates are visually **verse-only**, including wide, long-line or stepped verse that can resemble prose. The label is based on the image, not on OCR text. This is a list of confirmed failure cases, not a detector performance estimate.

| Normalized page | Full-image identification |
| --- | --- |
| `1957-11_p02.png` | Михаил Вершинин, “Юность Октября”; Виктор Виноградов, “Смелость”. Long stepped verse, no prose introduction. |
| `1957-11_p05.png` | Виктор Боков, “ПОЛЯНА”, above Владимир Зуев, “Как это прекрасно”. The wide upper text is verse. |
| `1965-01_p01.png` | Фазиль Искандер portrait; “Поэты”, “Стога”, “Размышления об эссе”. Upper-right text is stanzaic verse. |
| `1965-01_p04.png` | Two ruled verse columns, including “Баллада об украденном козле”. |
| `1965-01_p05.png` | Continuation of that ballad, then “Летний лес”. |
| `1965-02_p01.png` | Борис Слуцкий portrait; “Мой комбат Назаров”, “Казахи под Можайском”, “Духовые оркестры”. |

Two further full-image negative controls are `1957-07_p02.png` (“Стихи молодых зарубежных поэтов”, Romanian and Indian poems) and `1965-02_p09.png` (stepped verse and “Баллада об автопортрете”). Neither contains an editorial introduction. All five initially suggested negative filenames are included and confirmed.

## Full-image audit: prose-positive pages

| Normalized page | Confirmed visual expectation |
| --- | --- |
| `1957-02_p04.png` | Two upper narrative columns; lower poems including “Река” and “Жизнь”, plus a portrait. |
| `1957-04_p03.png` | Сергей Есенин heading; wide bold editorial introduction; three lower verse columns headed “Речь Никанора Рассветова”. No portrait. |
| `1957-04_p05.png` | Upper poems; two lower columns headed “Примечания” are scholarly/editorial prose. |
| `1957-04_p06.png` | Two upper narrative columns; lower framed poem “Другу” beside a portrait. |
| `1957-04_p08.png` | “Из стихов молодых поэтов Таджикистана”; genuine wide editorial paragraph above the poems, approximately x215–1270, y240–560. |
| `1957-06_p05.png` | Three portraits: И. Лапиров, Ф. Балкарова, З. Тхагазитов. The prose is the **wide introduction x330–1365, y600–740**, not a lower-left biography. Poems are below. |
| `1957-07_p01.png` | Two upper prose columns through approximately y1320; Валентин Берестов's short poem “Мир” at the bottom. |
| `1957-07_p04.png` | Short editorial paragraph explaining Эммет Тилл's murder, **x145–890, y1300–1405**, above the lower poem. Added as the 26th case following the main session's discovery and a direct full-image check. |
| `1957-10_p04.png` | Article “Страничка памятной тетрадки” above; Роберт Рождественский's “Восемьдесят восемь” in three lower verse columns. |
| `1957-10_p05.png` | Three prose columns above approximately y1320; В. Константинов's autumn poem in two columns below. |
| `1957-11_p03.png` | Сергей Космынин heading; narrow italic prose introduction signed Евгений Винокуров; poems beside/below it. |
| `1957-12_p03.png` | Three upper prose columns; lower “Перед отъездом” beside Михаил Петраков's portrait. |
| `1957-12_p06.png` | Upper prose continues through short closing paragraphs to about y770; lower Лев Ошанин poem “СЫН”. |
| `1963-12_p07.png` | Photographed prose in two upper columns through approximately y900. Below: В. Берестов portrait, “К. И. Чуковскому” and “Бродячий кот”. |
| `1965-02_p08.png` | **Genuine prose** below Стефан Цанев's portrait, approximately x245–710, y625–1085. Other columns contain poems; a brief explanatory footnote is also visible. |
| `1965-02_p10.png` | “Так сказали мудрецы”: **wide top prose introduction x380–1310, y175–360**, signed Наум Гребнев. Below are two framed columns of versified aphorisms. There is no bottom editorial request. |
| `1967-09_p04.png` | Two upper prose columns. Below: Нина Королева portrait and verse, with the first poem beginning at bottom left and continuing at upper right. |
| `1967-10_p03.png` | Interview prose fills the left column and **upper right**. Right-side “Горизонт” and “Грусть” are poems. Left quoted speech is prose, not an inset verse quotation. |

These are visual retention/exclusion targets. This document does not report detector precision, recall, accuracy, or a final corpus result; use the corrected fixture for a separate reproducible evaluation.

## Caveats

- The 85 unlabeled pages are not certified negative. Corpus-wide sheets were only triage, not a reliable full-image content audit.
- None of these selected pages is claimed to be a prose-only magazine page. Positive controls deliberately contain genuine prose alongside poetry, matching the available development corpus and the relevant filtering failure mode.
- Case selection used known failures, baseline-positive candidates and main-session discoveries. This creates development bias by design.
- Initial contact-sheet interpretations were materially wrong. Revision 2 corrects the author/title evidence and prose locations, especially `1957-06_p05`, `1965-02_p10` and `1967-10_p03`; earlier summaries should not be reused.
- These labels evaluate prose/verse retention and exclusion, not OCR CER, reading order, precise paragraph counts, or production readiness. Revisit ambiguous boundaries at full resolution before tightening geometric thresholds.
