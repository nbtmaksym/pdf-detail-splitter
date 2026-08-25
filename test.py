import os
import re
import json
import shutil
import threading
import fitz
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

try:
    import pytesseract
    OCR_AVAILABLE = True

    # Skompilowany .exe uruchomiony podwojnym kliknieciem (Explorer)
    # czasem NIE widzi tych samych zmiennych PATH co terminal, z ktorego
    # program byl budowany/testowany - przez co pytesseract nie moze
    # znalezc tesseract.exe mimo ze w terminalu dziala bez problemu.
    # Zamiast polegac na PATH, szukamy go wprost w typowych lokalizacjach
    # instalacji i ustawiamy sciezke na sztywno, jesli PATH zawiedzie.
    if not shutil.which(pytesseract.pytesseract.tesseract_cmd):
        for candidate in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ):
            if os.path.isfile(candidate):
                pytesseract.pytesseract.tesseract_cmd = candidate
                break
        else:
            OCR_AVAILABLE = False
except ImportError:
    OCR_AVAILABLE = False

from concurrent.futures import ThreadPoolExecutor

MAX_DISPLAY = 1300
MIN_ZOOM = 0.25
MAX_ZOOM = 8.0
ZOOM_STEP = 1.25

PROFIL_LABEL_RE = re.compile(r'/[A-Z]\d{2,3}[A-Z0-9]*$')
PART_NUMBER_RE = re.compile(r'^\d{1,5}$')
POS_WORD_TEXT_RE = re.compile(r'^POS\.?,?$', re.IGNORECASE)
MARK_RE = re.compile(r'^[A-Za-z0-9]{1,6}[-._]\d{2,6}$|^\d{1,5}$')


def sanitize_filename(name):
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, '_')
    name = name.strip().strip('.')
    return name or "detal"


def unique_path(folder, filename):
    dest = os.path.join(folder, filename)
    if not os.path.exists(dest):
        return dest
    base, ext = os.path.splitext(filename)
    i = 1
    while True:
        new = os.path.join(folder, f"{base}_{i}{ext}")
        if not os.path.exists(new):
            return new
        i += 1


def has_text_layer(page, min_words=15):
    words = page.get_text("words")
    return len(words) >= min_words, words


def detect_anchors(words):
    """Znajduje pary (etykieta czesci, pozycja) dwiema strategiami na raz,
    bo rozne pliki uzywaja roznych ukladow:

      A) etykieta profilu konczaca sie gatunkiem stali (np. '.../S355JR'),
         a numer czesci (czysta liczba) stoi na tej samej linii dalej w
         prawo - dziala dla wiekszosci plikow Tekla/Kräss.

      B) doslowne slowo 'Pos.' na linii, a zaraz po nim znak pozycji ktory
         NIE musi byc czysta liczba (np. '3R-4001', litery+myslnik+cyfry) -
         spotykane w niektorych plikach (np. eksport z innego CAD-u), gdzie
         strategia A zawodzi bo numer czesci nie jest samymi cyframi.

    Wyniki obu strategii sa laczone, z odrzuceniem duplikatow (ta sama
    fizyczna pozycja znaleziona przez obie na raz)."""
    anchors = []
    used_positions = []

    # --- strategia A: etykieta profilu + czysta liczba na tej samej linii ---
    labels = [w for w in words if PROFIL_LABEL_RE.search(w[4])]
    numbers = [w for w in words if PART_NUMBER_RE.match(w[4])]
    used_numbers = set()
    for x0, y0, x1, y1, text, *_ in labels:
        cy = (y0 + y1) / 2
        candidates = [
            n for idx, n in enumerate(numbers)
            if idx not in used_numbers
            and abs(((n[1] + n[3]) / 2) - cy) < 8
            and n[0] > x1 - 5
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda n: n[0])
        best = candidates[0]
        idx = numbers.index(best)
        used_numbers.add(idx)
        nx0, ny0, nx1, ny1, ntext, *_ = best
        ax, ay = (nx0 + nx1) / 2, (ny0 + ny1) / 2
        anchors.append((ntext, ax, ay))
        used_positions.append((ax, ay))

    # --- strategia B: slowo 'Pos.' + znak pozycji (dowolny format) obok ---
    # Etykieta 'Pos.' czesto siedzi NIZEJ niz rysunek/schemat danego
    # elementu (ktory jest NAD nia), a opis profilu jest PONIZEJ - wiec
    # kotwica postawiona doslownie na 'Pos.' jest przesunieta w dol
    # wzgledem prawdziwego srodka elementu, przez co segmentacja
    # (watershed) czesto "spływa" w dol i zabiera rysunek NASTEPNEGO
    # elementu zamiast wlasnego. Naprawiamy to centrujac kazda kotwice
    # w polowie odleglosci do poprzedniej kotwicy W TEJ SAMEJ KOLUMNIE
    # (ta sama okolica X) - to symetrycznie obejmuje zarowno wlasny
    # rysunek nad etykieta, jak i wlasny opis pod nia.
    raw_b = []
    pos_words = [w for w in words if POS_WORD_TEXT_RE.match(w[4])]
    for x0, y0, x1, y1, text, *_ in pos_words:
        cy = (y0 + y1) / 2
        candidates = [
            w for w in words
            if MARK_RE.match(w[4]) and w[0] > x1 - 5 and abs(((w[1] + w[3]) / 2) - cy) < 8
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda w: w[0])
        mx0, my0, mx1, my1, mtext, *_ = candidates[0]
        ax, ay = (mx0 + mx1) / 2, (my0 + my1) / 2
        if any(abs(ax - ux) < 60 and abs(ay - uy) < 20 for ux, uy in used_positions):
            continue
        raw_b.append((mtext, ax, ay))

    for mtext, ax, ay in raw_b:
        anchors.append((mtext, ax, ay))
        used_positions.append((ax, ay))

    return anchors


def point_to_rect_dist(px, py, x0, y0, x1, y1):
    dx = max(x0 - px, 0, px - x1)
    dy = max(y0 - py, 0, py - y1)
    return (dx * dx + dy * dy) ** 0.5


def _rasterize_and_clean(page, analysis_zoom):
    """Renderuje strone i usuwa ramke arkusza oraz dlugie/cienkie linie
    odniesienia (osie, linie referencyjne), ktore nie naleza do zadnego
    pojedynczego elementu i psulyby dalsza analize."""
    mat = fitz.Matrix(analysis_zoom, analysis_zoom)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img[:, :, 0]
    binary = (gray < 250).astype(np.uint8)

    h, w = binary.shape
    border = max(2, int(min(h, w) * 0.012))
    binary[:border, :] = 0
    binary[-border:, :] = 0
    binary[:, :border] = 0
    binary[:, -border:] = 0

    # lekka dylatacja - laczy przerwy antyaliasingu w liniach/tekscie,
    # dzieki czemu elementy jednego detalu tworza spojna siatke "atramentu"
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.dilate(binary, close_kernel)

    # znajdz i usun dlugie, cienkie linie odniesienia/osiowe (niska
    # gestosc atramentu w duzym, wydluzonym bounding-boxie) - liczone
    # wektorowo (ndimage) zamiast petli po calym obrazie na kazdy
    # komponent, bo to bylo glowne waskie gardlo wydajnosci
    analysis_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    dilated_for_filter = cv2.dilate(closed, analysis_kernel)
    num, labeled = cv2.connectedComponents(dilated_for_filter, connectivity=8)
    long_dim_limit = 0.25 * min(w, h)

    if num > 1:
        objs = ndi.find_objects(labeled)
        ink_sums = ndi.sum(closed, labeled, index=np.arange(1, num))
        to_clear = []
        for i, sl in enumerate(objs):
            if sl is None:
                continue
            y_sl, x_sl = sl
            bw = x_sl.stop - x_sl.start
            bh = y_sl.stop - y_sl.start
            ink = ink_sums[i]
            density = ink / max(1, bw * bh)
            if max(bw, bh) > long_dim_limit and density < 0.05:
                to_clear.append(i + 1)
        if to_clear:
            clear_mask = np.isin(labeled, to_clear)
            closed[clear_mask] = 0

    return closed, pix.width, pix.height


def segment_from_anchors(page, anchors, analysis_zoom=1.5, margin_pt=4, seed_radius_px=4):
    """Rdzen segmentacji watershed - niezalezny od zrodla kotwic (moga
    pochodzic z prawdziwego tekstu PDF albo z OCR). Zwraca liste
    (x0,y0,x1,y1,nazwa) w punktach PDF.

    Dla kazdego piksela atramentu liczona jest "odleglosc geodezyjna"
    preferujaca podroz wzdluz polaczonych linii rysunku, a nie w linii
    prostej. Dzieki temu dlugi, pojedynczy element trzyma sie swojej
    etykiety nawet jesli jego drugi koniec lezy fizycznie blizej
    etykiety sasiada - a segmentacja z definicji nie pozwala dwom
    regionom sie nakladac."""
    if not anchors:
        return []

    ink_mask, img_w, img_h = _rasterize_and_clean(page, analysis_zoom)

    # elewacja: 0 na atramencie, rosnie z odlegloscia od najblizszego
    # atramentu w pustej przestrzeni -> "tanio" plynac wzdluz linii,
    # "drogo" skakac przez pustke do sasiedniego elementu
    elevation = ndi.distance_transform_edt(ink_mask == 0)

    markers = np.zeros((img_h, img_w), dtype=np.int32)
    id_to_name = {}
    name_to_id = {}
    next_id = 1
    for name, ax, ay in anchors:
        if name not in name_to_id:
            name_to_id[name] = next_id
            id_to_name[next_id] = name
            next_id += 1
        idx = name_to_id[name]
        px = int(round(ax * analysis_zoom))
        py = int(round(ay * analysis_zoom))
        px = min(max(px, 0), img_w - 1)
        py = min(max(py, 0), img_h - 1)
        y0, y1 = max(0, py - seed_radius_px), min(img_h, py + seed_radius_px + 1)
        x0, x1 = max(0, px - seed_radius_px), min(img_w, px + seed_radius_px + 1)
        markers[y0:y1, x0:x1] = idx

    labels = watershed(elevation, markers=markers, connectivity=2)

    # tylko piksele atramentu licza sie do ramki (puste tlo przypisane
    # danej etykiecie przez watershed nas nie interesuje)
    labels_ink = np.where(ink_mask == 1, labels, 0)
    objs = ndi.find_objects(labels_ink, max_label=len(id_to_name))

    page_w_pt = page.rect.width
    page_h_pt = page.rect.height

    boxes = []
    for idx, name in id_to_name.items():
        if idx - 1 >= len(objs):
            continue
        sl = objs[idx - 1]
        if sl is None:
            continue
        y_sl, x_sl = sl
        x0 = x_sl.start / analysis_zoom - margin_pt
        y0 = y_sl.start / analysis_zoom - margin_pt
        x1 = x_sl.stop / analysis_zoom + margin_pt
        y1 = y_sl.stop / analysis_zoom + margin_pt
        x0 = max(0, min(x0, page_w_pt))
        y0 = max(0, min(y0, page_h_pt))
        x1 = max(0, min(x1, page_w_pt))
        y1 = max(0, min(y1, page_h_pt))
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        boxes.append((x0, y0, x1, y1, name))

    return trim_overlaps(boxes)


SHORT_NUM_RE = re.compile(r'^\d{1,4}$')
GRADE_WORD_RE = re.compile(r'^S\d{3}[A-Z0-9]{0,3}$')


def _refine_box_from_text(box, words, anchor_x, anchor_y):
    """Dopracowuje pionowe granice ramki na podstawie DOKLADNYCH pozycji
    prawdziwego tekstu, zamiast liczyc wylacznie na segmentacje watershed
    (ktora w gestych, wielokolumnowych ukladach potrafi 'przeskoczyc'
    granice miedzy sasiednimi elementami w sposob trudny do
    przewidzenia). Szuka:
      - najnizszego punktu wlasnego opisu profilu (linia z samodzielnym
        slowem gatunku stali, np. 'S235JRH') PONIZEJ kotwicy 'Pos.',
      - najwyzszego punktu etykiety wymiaru (krotka liczba, np. dlugosc
        lub srednica rurki) POWYZEJ kotwicy - to czesc rysunku/schematu
        ktory nalezy do TEGO SAMEGO elementu.
    Jesli oba znajdzie, ustawia je jako twarde granice ramki - to
    deterministyczne i precyzyjne, w przeciwienstwie do zgadywania
    pozycji kotwicy dla segmentacji."""
    x0, y0, x1, y1, name = box

    desc_bottom = None
    for w in words:
        wx0, wy0, wx1, wy1, wtext = w[:5]
        if (PROFIL_LABEL_RE.search(wtext) or GRADE_WORD_RE.match(wtext)) \
                and anchor_y < wy0 < anchor_y + 60 and x0 - 40 < wx0 < x1 + 40:
            if desc_bottom is None or wy1 > desc_bottom:
                desc_bottom = wy1

    icon_top = None
    for w in words:
        wx0, wy0, wx1, wy1, wtext = w[:5]
        if SHORT_NUM_RE.match(wtext) and anchor_y - 100 < wy0 < anchor_y - 15 \
                and x0 - 60 < wx0 < x1 + 60:
            if icon_top is None or wy0 < icon_top:
                icon_top = wy0

    if desc_bottom is not None and icon_top is not None:
        return (x0, max(0, icon_top - 6), x1, desc_bottom + 6, name)
    return box


def auto_detect_boxes(page, words, analysis_zoom=1.5, margin_pt=4, seed_radius_px=4):
    """Wersja dla plikow z warstwa tekstowa - kotwice z prawdziwego
    tekstu PDF (patrz segment_from_anchors dla opisu samej segmentacji)."""
    anchors = detect_anchors(words)
    boxes = segment_from_anchors(page, anchors, analysis_zoom, margin_pt, seed_radius_px)

    anchor_pos = {}
    for name, ax, ay in anchors:
        anchor_pos[name] = (ax, ay)

    refined = []
    for box in boxes:
        name = box[4]
        if name in anchor_pos:
            ax, ay = anchor_pos[name]
            box = _refine_box_from_text(box, words, ax, ay)
        refined.append(box)
    return trim_overlaps(refined)


# --- Wersja dla plikow BEZ warstwy tekstowej: kotwice znajdowane przez
# OCR na calym arkuszu (tylko lokalizacja - dokladny odczyt numeru
# nastepuje pozniej, osobno, na juz wycietym i czystym fragmencie,
# gdzie OCR dziala dużo lepiej niz na calej, gestej stronie) ---

POS_WORD_RE = re.compile(r"^[PF][o0O][sz]\.?,?$", re.IGNORECASE)
NUM_LABEL_RE = re.compile(r"^\d{2,4}_\d{1,4}[.,)]*$")
PROFILE_WORD_RE = re.compile(r"^[A-Za-z]{1,5}\d")


def find_pos_anchors_via_ocr(page, analysis_zoom=1.5):
    """Renderuje cala strone i szuka kotwic - pozycji etykiet
    poszczegolnych elementow - dwiema rownoleglymi strategiami, bo
    rozne pliki/projekty uzywaja roznej konwencji:

      A) slowo 'Pos.'/'Poz.' (PL/ENG) tuz przed numerem
      B) sam numer w formacie 'NNN_N' polozony blisko oznaczenia
         profilu (np. '100_3 ... BL20*140') - bez zadnego
         poprzedzajacego slowa - spotykane w niektorych plikach

    W obu przypadkach chodzi TYLKO o znalezienie przyblizonej pozycji
    kazdej etykiety, nie o odczytanie numeru - dokladny odczyt
    nastepuje pozniej, osobno, na juz wycietym i czystym fragmencie.
    Zwraca liste (tymczasowa_nazwa, x_pt, y_pt)."""
    if not OCR_AVAILABLE:
        return []

    mat = fitz.Matrix(analysis_zoom, analysis_zoom)
    pix = page.get_pixmap(matrix=mat)
    mode = "RGB" if pix.n < 4 else "RGBA"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    if mode == "RGBA":
        img = img.convert("RGB")

    gray = np.array(img.convert("L"))
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    proc = Image.fromarray(binarized)

    def _run_psm(psm):
        try:
            return pytesseract.image_to_data(proc, config=f"--psm {psm}",
                                              output_type=pytesseract.Output.DICT)
        except Exception:
            return None

    all_words = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for data in executor.map(_run_psm, (11, 6)):
            if data is None:
                continue
            for i, word in enumerate(data['text']):
                w = word.strip()
                if w:
                    all_words.append((w, data['left'][i], data['top'][i],
                                       data['width'][i], data['height'][i]))

    # --- strategia A: slowo Pos./Poz. ---
    strategy_a = {}
    for w, x, y, ww, hh in all_words:
        if POS_WORD_RE.match(w):
            grid = (round(x / 40), round(y / 40))
            strategy_a.setdefault(grid, (x, y + hh // 2))

    # --- strategia B: numer_numer blisko oznaczenia profilu ---
    strategy_b = {}
    for w, x, y, ww, hh in all_words:
        if not NUM_LABEL_RE.match(w):
            continue
        cy = y + hh / 2
        near_profile = any(
            PROFILE_WORD_RE.match(w2) and abs(x2 - (x + ww)) < 250 and abs((y2 + hh2 / 2) - cy) < 30
            for w2, x2, y2, ww2, hh2 in all_words
        )
        if near_profile:
            key = w.strip(".,()")
            strategy_b.setdefault(key, []).append((x, cy))

    anchors = []
    used_positions = []
    idx = 1
    for key, positions in strategy_b.items():
        ax = sum(p[0] for p in positions) / len(positions)
        ay = sum(p[1] for p in positions) / len(positions)
        anchors.append((f"det_{idx}", ax / analysis_zoom, ay / analysis_zoom))
        used_positions.append((ax, ay))
        idx += 1

    for (x, y) in strategy_a.values():
        # pomin jesli blisko juz istniejacej kotwicy ze strategii B -
        # to prawdopodobnie ten sam element, nie trzeba dublowac
        if any(abs(x - ux) < 150 and abs(y - uy) < 150 for ux, uy in used_positions):
            continue
        anchors.append((f"det_{idx}", x / analysis_zoom, y / analysis_zoom))
        used_positions.append((x, y))
        idx += 1

    return anchors


LABEL_CORE_RE = re.compile(r"POS[.,]?\s*([A-Za-z0-9]{2,4})[\s_]+([A-Za-z0-9]{1,5})", re.IGNORECASE)
_DIGIT_CONFUSABLES = {"O": "0", "I": "1", "L": "1", "S": "5", "B": "8"}


def _normalize_digit_group(g):
    return "".join(_DIGIT_CONFUSABLES.get(ch.upper(), ch) for ch in g)


def _extract_label_core(text):
    """Wyciaga sam rdzen numeru pozycji ('NNN_N') wprost jako dwie grupy
    cyfr wystepujace po 'Pos.', ignorujac WSZYSTKO co jest po nich
    (ilosc sztuk, myslniki, nawiasy) niezaleznie jak bardzo jest
    znieksztalcone przez OCR - zamiast probowac wyciac 'ogon' po
    dopasowaniu, co zawodzilo przy mocno poszarpanych koncowkach (np.
    '-_J5__x'). Zwraca None jesli ktoras z grup nie da sie sprowadzic
    do samych cyfr (lepiej przyznac ze sie nie udalo, niz pokazac
    niepelny/mylacy numer)."""
    m = LABEL_CORE_RE.search(text)
    if not m:
        return None
    g1 = _normalize_digit_group(m.group(1))
    g2 = _normalize_digit_group(m.group(2))
    if not (g1.isdigit() and g2.isdigit()):
        return None
    return f"{g1}_{g2}"


def refine_label_via_ocr(page, box, target_max_px=2200):
    """Wycina dany fragment strony (juz wyznaczony przez segmentacje) i
    robi na nim celowany, dokladny odczyt OCR - na malym, czystym
    fragmencie (bez reszty arkusza w tle) czyta sie DUZO lepiej niz na
    calej stronie. Rozdzielczosc jest ograniczana do target_max_px na
    dluzszym boku fragmentu - bez tego fizycznie duze fragmenty (np.
    dlugie profile) generowalyby ogromne obrazy i dominowaly caly czas
    przetwarzania mimo rownoleglosci. Zwraca sugerowana nazwe pliku
    (numer pozycji jesli sie uda go odczytac, inaczej None)."""
    if not OCR_AVAILABLE:
        return None
    try:
        x0, y0, x1, y1 = box[:4]
        w, h = max(x1 - x0, 1), max(y1 - y0, 1)
        zoom = min(8.0, max(2.0, target_max_px / max(w, h)))

        clip = fitz.Rect(x0, y0, x1, y1)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        mode = "RGB" if pix.n < 4 else "RGBA"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        if mode == "RGBA":
            img = img.convert("RGB")

        gray = np.array(img.convert("L"))
        _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bordered = cv2.copyMakeBorder(binarized, 20, 20, 20, 20,
                                       cv2.BORDER_CONSTANT, value=255)
        proc = Image.fromarray(bordered)

        text = pytesseract.image_to_string(proc, config="--psm 6", timeout=15)
        return _extract_label_core(text)
    except Exception:
        return None


def auto_detect_boxes_no_text(page, analysis_zoom=1.5, refine_target_px=2200,
                               min_size_pt=80, refine_workers=4, progress_cb=None):
    """Pelny pipeline dla plikow BEZ warstwy tekstowej:
      1. znajdz przyblizone pozycje etykiet 'Pos.' przez OCR calej strony
      2. uzyj ich jako kotwic w tej samej segmentacji watershed co dla
         plikow z tekstem
      3. dla kazdego wyznaczonego fragmentu zrob DRUGI, celowany odczyt
         OCR (rownolegle - to jedyny wolny krok) - juz na czystym,
         wycietym obrazku - zeby otrzymac prawdziwy numer pozycji do
         nazwy pliku
    Fragmenty mniejsze niz min_size_pt (w obu wymiarach) sa odrzucane
    jako szum (np. przypadkowe znaleziska OCR bez realnej geometrii
    w poblizu). progress_cb (opcjonalny) jest wywolywany z (i, n) w
    trakcie kroku 3, do pokazania postepu w GUI."""
    anchors = find_pos_anchors_via_ocr(page, analysis_zoom)
    if not anchors:
        return []

    boxes = segment_from_anchors(page, anchors, analysis_zoom)
    boxes = [b for b in boxes if (b[2] - b[0]) >= min_size_pt and (b[3] - b[1]) >= min_size_pt]
    if not boxes:
        return []

    labels = [None] * len(boxes)
    done_count = [0]

    def _refine(i):
        label = refine_label_via_ocr(page, boxes[i], refine_target_px)
        done_count[0] += 1
        if progress_cb:
            progress_cb(done_count[0], len(boxes))
        return i, label

    with ThreadPoolExecutor(max_workers=refine_workers) as executor:
        for i, label in executor.map(_refine, range(len(boxes))):
            labels[i] = label

    refined = []
    for box, label in zip(boxes, labels):
        name = sanitize_filename(label.replace(" ", "_")) if label else box[4]
        refined.append((box[0], box[1], box[2], box[3], name))
    return refined


def trim_overlaps(boxes):
    """Ogranicza zachodzenie na siebie sasiadujacych wykrytych ramek -
    przy nakladaniu przycina obie w polowie odleglosci miedzy ich
    srodkami, wzdluz osi z wiekszym nakladaniem."""
    boxes = [list(b) for b in boxes]
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            ax0, ay0, ax1, ay1, _ = boxes[i]
            bx0, by0, bx1, by1, _ = boxes[j]
            ox0, oy0 = max(ax0, bx0), max(ay0, by0)
            ox1, oy1 = min(ax1, bx1), min(ay1, by1)
            if ox1 <= ox0 or oy1 <= oy0:
                continue
            overlap_w, overlap_h = ox1 - ox0, oy1 - oy0
            acx, acy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
            bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
            if overlap_w < overlap_h:
                mid = (acx + bcx) / 2
                if acx < bcx:
                    boxes[i][2] = min(boxes[i][2], mid)
                    boxes[j][0] = max(boxes[j][0], mid)
                else:
                    boxes[i][0] = max(boxes[i][0], mid)
                    boxes[j][2] = min(boxes[j][2], mid)
            else:
                mid = (acy + bcy) / 2
                if acy < bcy:
                    boxes[i][3] = min(boxes[i][3], mid)
                    boxes[j][1] = max(boxes[j][1], mid)
                else:
                    boxes[i][1] = max(boxes[i][1], mid)
                    boxes[j][3] = min(boxes[j][3], mid)
    return [tuple(b) for b in boxes if (b[2] - b[0]) > 5 and (b[3] - b[1]) > 5]


class DetailSplitterApp:
    HANDLE_SIZE = 7

    def __init__(self, root):
        self.root = root
        self.root.title("PDF - rozbijanie na pojedyncze detale")
        self.root.geometry("1500x900")

        self.doc = None
        self.page = None
        self.page_w = 0
        self.page_h = 0
        self.zoom = 1.0
        self.tk_img = None
        self.pil_img = None
        self.rects = []  # list of dict: x0,y0,x1,y1 (pdf pt), name
        self.has_text = False
        self.words_cache = []

        self.start_x = None
        self.start_y = None
        self.temp_rect_id = None

        self.selected_index = None
        self.drag_mode = None       # None | 'new' | 'move' | 'resize'
        self.drag_corner = None
        self.drag_orig_box = None   # canvas coords at drag start
        self.drag_start_xy = None

        self._build_ui()

    def _build_ui(self):
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=6)

        tk.Button(top, text="Otworz PDF...", command=self.open_pdf).pack(side="left", padx=(0, 6))
        self.btn_auto = tk.Button(top, text="Auto-wykryj", command=self.auto_detect, state="disabled")
        self.btn_auto.pack(side="left", padx=6)
        tk.Button(top, text="Wyczysc wszystko", command=self.clear_all).pack(side="left", padx=6)
        tk.Button(top, text="Eksportuj wszystkie...", command=self.export_all,
                  bg="#0078D4", fg="white", font=("Segoe UI", 10, "bold")).pack(side="left", padx=6)

        tk.Frame(top, width=2, bg="#ccc").pack(side="left", fill="y", padx=10, pady=2)
        tk.Button(top, text="-", width=3, command=lambda: self.zoom_at(1 / ZOOM_STEP)).pack(side="left")
        tk.Button(top, text="+", width=3, command=lambda: self.zoom_at(ZOOM_STEP)).pack(side="left", padx=(2, 6))
        tk.Button(top, text="Dopasuj", command=self.fit_to_window).pack(side="left")

        self.status_var = tk.StringVar(value="Otworz plik PDF zeby zaczac.")
        tk.Label(top, textvariable=self.status_var, fg="#555").pack(side="left", padx=16)

        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        canvas_frame = tk.Frame(main, bd=1, relief="sunken")
        canvas_frame.pack(side="left", fill="both", expand=True)
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="#e8e8e8", cursor="cross")
        h_scroll = tk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        v_scroll = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Delete>", self.on_delete_key)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self.on_shift_wheel)
        self.canvas.bind("<Control-MouseWheel>", self.on_ctrl_wheel)

        side = tk.Frame(main, width=320)
        side.pack(side="left", fill="y", padx=(8, 0))
        side.pack_propagate(False)

        tk.Label(side, text="Wykryte / narysowane elementy:", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", pady=(0, 6))

        list_container = tk.Frame(side)
        list_container.pack(fill="both", expand=True)

        canvas_scroll = tk.Canvas(list_container, highlightthickness=0)
        scrollbar = tk.Scrollbar(list_container, orient="vertical", command=canvas_scroll.yview)
        self.list_frame = tk.Frame(canvas_scroll)

        self.list_frame.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        )
        canvas_scroll.create_window((0, 0), window=self.list_frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        canvas_scroll.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas_scroll.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_wheel(_e):
            canvas_scroll.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_wheel(_e):
            canvas_scroll.unbind_all("<MouseWheel>")

        canvas_scroll.bind("<Enter>", _bind_wheel)
        canvas_scroll.bind("<Leave>", _unbind_wheel)

        help_text = (
            "Jak uzywac:\n"
            "1. Otworz plik PDF.\n"
            "2. Jesli plik ma tekst, kliknij Auto-wykryj\n"
            "   - sprawdz i popraw wykryte ramki.\n"
            "3. Przyblizanie: Ctrl + kolko myszy, albo\n"
            "   przyciski +/- . Zwykle kolko przewija\n"
            "   w pionie, Shift+kolko w poziomie.\n"
            "4. Rysuj wlasne ramki lewym przyciskiem\n"
            "   myszy, przeciagajac po pustym miejscu.\n"
            "5. Kliknij ramke i przeciagnij SRODEK\n"
            "   zeby ja przesunac, albo ROG zeby\n"
            "   zmienic rozmiar.\n"
            "6. Zaznaczona ramka: klawisz Delete\n"
            "   tez ja usuwa.\n"
            "7. Nazwij kazdy element w polu obok.\n"
            "8. Kliknij Eksportuj wszystkie."
        )
        tk.Label(side, text=help_text, justify="left", fg="#555",
                 wraplength=300).pack(anchor="w", pady=(10, 0), side="bottom")

    # --- plik ---

    def open_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("Pliki PDF", "*.pdf")])
        if not path:
            return
        try:
            self.doc = fitz.open(path)
            self.page = self.doc[0]
        except Exception as e:
            messagebox.showerror("Blad", f"Nie mozna otworzyc pliku:\n{e}")
            return

        self.pdf_path = path
        self.page_w = self.page.rect.width
        self.page_h = self.page.rect.height
        self.has_text, self.words_cache = has_text_layer(self.page)

        self.zoom = min(MAX_DISPLAY / self.page_w, MAX_DISPLAY / self.page_h, 3.0)
        self._render_base_image()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

        self.rects = []
        self.selected_index = None
        self.refresh_list()

        info = "plik ma warstwe tekstowa - dostepne Auto-wykryj" if self.has_text \
            else "plik BEZ warstwy tekstowej - Auto-wykryj dostepne (przez OCR), ale wolniejsze"
        self.status_var.set(f"{os.path.basename(path)}  |  {info}")
        self.btn_auto.config(state="normal" if OCR_AVAILABLE or self.has_text else "disabled")

    def _render_base_image(self):
        """Renderuje strone PDF przy aktualnym self.zoom i odswieza tlo
        canvasu. Uzywane zarowno przy otwieraniu pliku, jak i przy kazdej
        zmianie przyblizenia."""
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = self.page.get_pixmap(matrix=mat)
        mode = "RGB" if pix.n < 4 else "RGBA"
        self.pil_img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        if mode == "RGBA":
            self.pil_img = self.pil_img.convert("RGB")

        self.tk_img = ImageTk.PhotoImage(self.pil_img)
        self.canvas.config(scrollregion=(0, 0, pix.width, pix.height))
        self.canvas.delete("bg")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img, tags="bg")
        self.canvas.tag_lower("bg")

    # --- przyblizanie / przewijanie ---

    def zoom_at(self, factor, widget_x=None, widget_y=None):
        if self.doc is None:
            return
        new_zoom = min(MAX_ZOOM, max(MIN_ZOOM, self.zoom * factor))
        if abs(new_zoom - self.zoom) < 1e-6:
            return

        if widget_x is None:
            widget_x = self.canvas.winfo_width() / 2
        if widget_y is None:
            widget_y = self.canvas.winfo_height() / 2

        cx = self.canvas.canvasx(widget_x)
        cy = self.canvas.canvasy(widget_y)
        pdf_x = cx / self.zoom
        pdf_y = cy / self.zoom

        self.zoom = new_zoom
        self._render_base_image()
        self.redraw_rects()

        target_cx = pdf_x * self.zoom
        target_cy = pdf_y * self.zoom
        total_w = max(1, self.pil_img.width)
        total_h = max(1, self.pil_img.height)
        frac_x = max(0, min(1, (target_cx - widget_x) / total_w))
        frac_y = max(0, min(1, (target_cy - widget_y) / total_h))
        self.canvas.xview_moveto(frac_x)
        self.canvas.yview_moveto(frac_y)

    def fit_to_window(self):
        if self.doc is None:
            return
        cw = self.canvas.winfo_width() or MAX_DISPLAY
        ch = self.canvas.winfo_height() or MAX_DISPLAY
        self.zoom = min(cw / self.page_w, ch / self.page_h, MAX_ZOOM)
        self._render_base_image()
        self.redraw_rects()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def on_wheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def on_shift_wheel(self, event):
        self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

    def on_ctrl_wheel(self, event):
        factor = ZOOM_STEP if event.delta > 0 else 1 / ZOOM_STEP
        self.zoom_at(factor, event.x, event.y)

    # --- rysowanie / przesuwanie / zmiana rozmiaru ---

    def _corner_hit(self, cx, cy, x0, y0, x1, y1):
        tol = self.HANDLE_SIZE
        corners = {'tl': (x0, y0), 'tr': (x1, y0), 'bl': (x0, y1), 'br': (x1, y1)}
        for name, (hx, hy) in corners.items():
            if abs(cx - hx) <= tol and abs(cy - hy) <= tol:
                return name
        return None

    def on_mouse_down(self, event):
        if self.doc is None:
            return
        self.canvas.focus_set()
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)

        # 1) czy klikamy w rog istniejacej ramki (zmiana rozmiaru)?
        for i in reversed(range(len(self.rects))):
            x0, y0, x1, y1 = [v * self.zoom for v in self.rects[i]["box"]]
            corner = self._corner_hit(cx, cy, x0, y0, x1, y1)
            if corner:
                self.selected_index = i
                self.drag_mode = 'resize'
                self.drag_corner = corner
                self.drag_orig_box = (x0, y0, x1, y1)
                self.drag_start_xy = (cx, cy)
                self.redraw_rects()
                return

        # 2) czy klikamy wewnatrz istniejacej ramki (przesuwanie)?
        for i in reversed(range(len(self.rects))):
            x0, y0, x1, y1 = [v * self.zoom for v in self.rects[i]["box"]]
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                self.selected_index = i
                self.drag_mode = 'move'
                self.drag_orig_box = (x0, y0, x1, y1)
                self.drag_start_xy = (cx, cy)
                self.redraw_rects()
                return

        # 3) w przeciwnym razie zaczynamy rysowac nowa ramke
        self.selected_index = None
        self.drag_mode = 'new'
        self.start_x, self.start_y = cx, cy
        self.temp_rect_id = self.canvas.create_rectangle(
            cx, cy, cx, cy, outline="#00AA00", width=2, tags="temp"
        )
        self.redraw_rects()

    def on_mouse_drag(self, event):
        if self.drag_mode is None:
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)

        if self.drag_mode == 'new':
            if self.temp_rect_id is not None:
                self.canvas.coords(self.temp_rect_id, self.start_x, self.start_y, cx, cy)
            return

        if self.selected_index is None:
            return
        x0, y0, x1, y1 = self.drag_orig_box

        if self.drag_mode == 'move':
            dx = cx - self.drag_start_xy[0]
            dy = cy - self.drag_start_xy[1]
            nx0, ny0, nx1, ny1 = x0 + dx, y0 + dy, x1 + dx, y1 + dy
        elif self.drag_mode == 'resize':
            nx0, ny0, nx1, ny1 = x0, y0, x1, y1
            if self.drag_corner == 'tl':
                nx0, ny0 = cx, cy
            elif self.drag_corner == 'tr':
                nx1, ny0 = cx, cy
            elif self.drag_corner == 'bl':
                nx0, ny1 = cx, cy
            elif self.drag_corner == 'br':
                nx1, ny1 = cx, cy
            nx0, nx1 = sorted([nx0, nx1])
            ny0, ny1 = sorted([ny0, ny1])
        else:
            return

        self.rects[self.selected_index]["box"] = (
            nx0 / self.zoom, ny0 / self.zoom, nx1 / self.zoom, ny1 / self.zoom
        )
        self.redraw_rects()

    def on_mouse_up(self, event):
        if self.drag_mode == 'new' and self.temp_rect_id is not None:
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            self.canvas.delete("temp")
            self.temp_rect_id = None

            x0, x1 = sorted([self.start_x, cx])
            y0, y1 = sorted([self.start_y, cy])
            if (x1 - x0) >= 5 and (y1 - y0) >= 5:
                pdf_box = (x0 / self.zoom, y0 / self.zoom, x1 / self.zoom, y1 / self.zoom)
                name = f"detal_{len(self.rects) + 1}"
                self.rects.append({"box": pdf_box, "name": name})
                self.selected_index = len(self.rects) - 1
                self.refresh_list()

        self.drag_mode = None
        self.drag_corner = None
        self.redraw_rects()

    def on_delete_key(self, event):
        if self.selected_index is not None:
            self.delete_rect(self.selected_index)

    # --- auto-wykrywanie ---

    def auto_detect(self):
        if self.doc is None:
            return
        if not self.has_text and not OCR_AVAILABLE:
            messagebox.showinfo("Niedostepne",
                                 "Ten plik nie ma warstwy tekstowej, a OCR nie jest "
                                 "zainstalowany (pytesseract). Narysuj elementy recznie.")
            return

        self.btn_auto.configure(state="disabled")
        if self.has_text:
            self.status_var.set("Wykrywanie elementow, prosze czekac...")
        else:
            self.status_var.set(
                "Wykrywanie elementow przez OCR - to moze potrwac az do minuty..."
            )

        def progress(i, n):
            self.root.after(0, lambda: self.status_var.set(
                f"Auto-wykrywanie (OCR): doprecyzowanie nazw {i}/{n}..."))

        def task():
            try:
                if self.has_text:
                    boxes = auto_detect_boxes(self.page, self.words_cache)
                else:
                    boxes = auto_detect_boxes_no_text(self.page, progress_cb=progress)
            except Exception as e:
                self.root.after(0, lambda: self._auto_detect_error(e))
                return
            self.root.after(0, lambda: self._auto_detect_done(boxes))

        threading.Thread(target=task, daemon=True).start()

    def _auto_detect_error(self, e):
        self.btn_auto.configure(state="normal")
        messagebox.showerror("Blad", f"Auto-wykrywanie nie powiodlo sie:\n{e}")
        self.status_var.set("Auto-wykrywanie: blad.")

    def _auto_detect_done(self, boxes):
        self.btn_auto.configure(state="normal")
        if not boxes:
            messagebox.showinfo("Brak wynikow",
                                 "Nie udalo sie automatycznie wykryc elementow.\n"
                                 "Narysuj je recznie.")
            self.status_var.set("Auto-wykrywanie: brak wynikow.")
            return

        for x0, y0, x1, y1, name in boxes:
            self.rects.append({"box": (x0, y0, x1, y1), "name": f"pos_{name}"})

        self.redraw_rects()
        self.refresh_list()
        extra = "" if self.has_text else " (nazwy z OCR bywaja niedokladne - SPRAWDZ je uwaznie)"
        self.status_var.set(
            f"Auto-wykrywanie: znaleziono {len(boxes)} elementow.{extra} "
            f"SPRAWDZ i popraw recznie brakujace/bledne."
        )

    # --- rysowanie / lista ---

    def redraw_rects(self):
        self.canvas.delete("rectbox")
        for i, r in enumerate(self.rects):
            x0, y0, x1, y1 = [v * self.zoom for v in r["box"]]
            selected = (i == self.selected_index)
            color = "#0078D4" if selected else "#E00000"
            width = 3 if selected else 2
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=width,
                                          tags="rectbox")
            self.canvas.create_text(x0 + 4, y0 + 4, anchor="nw", fill=color,
                                     text=r["name"], font=("Segoe UI", 10, "bold"),
                                     tags="rectbox")
            if selected:
                hs = self.HANDLE_SIZE
                for hx, hy in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]:
                    self.canvas.create_rectangle(hx - hs, hy - hs, hx + hs, hy + hs,
                                                  fill=color, outline="white", tags="rectbox")

    def refresh_list(self):
        for w in self.list_frame.winfo_children():
            w.destroy()

        for i, r in enumerate(self.rects):
            row = tk.Frame(self.list_frame)
            row.pack(fill="x", pady=2)

            var = tk.StringVar(value=r["name"])

            def on_change(*_, i=i, var=var):
                self.rects[i]["name"] = var.get()
                self.redraw_rects()

            var.trace_add("write", on_change)

            tk.Entry(row, textvariable=var, width=22).pack(side="left", padx=(0, 4))
            tk.Button(row, text="Usun", command=lambda i=i: self.delete_rect(i),
                      fg="#B00000").pack(side="left")

    def delete_rect(self, index):
        del self.rects[index]
        if self.selected_index == index:
            self.selected_index = None
        elif self.selected_index is not None and self.selected_index > index:
            self.selected_index -= 1
        self.redraw_rects()
        self.refresh_list()

    def clear_all(self):
        if not self.rects:
            return
        if messagebox.askyesno("Potwierdz", "Usunac wszystkie zaznaczone elementy?"):
            self.rects = []
            self.redraw_rects()
            self.refresh_list()

    # --- eksport ---

    def export_all(self):
        if self.doc is None:
            messagebox.showwarning("Brak pliku", "Najpierw otworz plik PDF.")
            return
        if not self.rects:
            messagebox.showwarning("Brak elementow", "Nie zaznaczono zadnych elementow do eksportu.")
            return

        out_folder = filedialog.askdirectory(title="Wybierz folder docelowy")
        if not out_folder:
            return

        errors = []
        saved = 0
        for r in self.rects:
            x0, y0, x1, y1 = r["box"]
            x0 = max(0, min(x0, self.page_w))
            y0 = max(0, min(y0, self.page_h))
            x1 = max(0, min(x1, self.page_w))
            y1 = max(0, min(y1, self.page_h))
            if x1 - x0 < 1 or y1 - y0 < 1:
                errors.append(f"{r['name']}: nieprawidlowy rozmiar, pominieto")
                continue
            try:
                new_doc = fitz.open()
                new_doc.insert_pdf(self.doc, from_page=0, to_page=0)
                new_page = new_doc[0]
                rect = fitz.Rect(x0, y0, x1, y1)
                new_page.set_cropbox(rect)

                filename = sanitize_filename(r["name"]) + ".pdf"
                out_path = unique_path(out_folder, filename)
                new_doc.save(out_path, garbage=4, deflate=True)
                new_doc.close()
                saved += 1
            except Exception as e:
                errors.append(f"{r['name']}: {e}")

        if errors:
            messagebox.showwarning(
                "Zakonczono z bledami",
                f"Zapisano: {saved}\nBledy:\n" + "\n".join(errors)
            )
        else:
            messagebox.showinfo("Gotowe", f"Zapisano {saved} plikow PDF do:\n{out_folder}")


def main():
    root = tk.Tk()
    app = DetailSplitterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()