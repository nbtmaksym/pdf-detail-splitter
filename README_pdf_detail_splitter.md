# PDF Detail Splitter

Narzędzie do rozbijania zbiorczych arkuszy rysunkowych PDF na pojedyncze pliki – jeden detal = jeden PDF. Zbudowane do pracy z dokumentacją warsztatową konstrukcji stalowych.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Problem

Biura projektowe często dostarczają jeden duży arkusz (A0/A1) z kilkudziesięcioma detalami naraz. Do produkcji potrzebny jest osobny plik na każdy element.

Program automatycznie wykrywa granice poszczególnych detali, pozwala je zweryfikować i poprawić w podglądzie, a następnie eksportuje każdy jako oddzielny, w pełni **wektorowy** PDF (bez utraty jakości – to wycinek oryginału, nie zrzut ekranu).

## Jak to działa

### Pliki z warstwą tekstową

1. Program znajduje etykiety pozycji w tekście PDF – obsługuje dwie konwencje:
   - `Pos. 733_1` (słowo `Pos.` przed numerem)
   - `100_3 (1x), BL20*140-S355` (sam numer obok oznaczenia profilu, bez słowa)
2. Każda etykieta staje się „kotwicą" dla segmentacji
3. **Watershed sterowany markerami** przypisuje geometrię do właściwych detali – liczy „odległość geodezyjną" preferującą podróż wzdłuż narysowanych linii zamiast w linii prostej. Dzięki temu długi element trzyma się swojej etykiety, nawet jeśli jego drugi koniec leży fizycznie bliżej sąsiada
4. Granice są doprecyzowywane na podstawie dokładnych współrzędnych tekstu (koniec opisu profilu poniżej etykiety, początek rysunku powyżej)

### Pliki bez warstwy tekstowej

Rysunki wyeksportowane jako czysta geometria wektorowa (np. z Tekla Structures) nie mają tekstu do odczytania. Wtedy:

1. OCR na całej stronie **lokalizuje** etykiety `Pos.` (nie musi ich dokładnie odczytać – wystarczy pozycja)
2. Te pozycje trafiają do tej samej segmentacji watershed
3. Dopiero na **wyciętym, czystym fragmencie** wykonywany jest drugi, celowany odczyt OCR, żeby uzyskać numer pozycji do nazwy pliku

Rozdzielenie „lokalizacji" od „odczytu" jest kluczowe: OCR gubi się na gęstym arkuszu, ale radzi sobie dobrze na małym, czystym wycinku.

## Obsługa w podglądzie

- **Auto-wykryj** – automatyczne wykrycie wszystkich detali
- **Ctrl + kółko myszy** – przybliżanie; zwykłe kółko przewija w pionie, Shift+kółko w poziomie
- **Przeciągnięcie środka ramki** – przesunięcie
- **Przeciągnięcie rogu** – zmiana rozmiaru
- **Delete** – usunięcie zaznaczonej ramki
- Rysowanie własnych ramek przez przeciągnięcie po pustym miejscu
- Edycja nazwy każdego elementu (staje się nazwą pliku)

## Instalacja

```bash
pip install -r requirements.txt
```

Dla plików bez warstwy tekstowej wymagany jest **Tesseract OCR**:

- Pobierz: https://github.com/UB-Mannheim/tesseract/wiki
- Zainstaluj i dodaj do `PATH` (program sprawdza też standardowe lokalizacje instalacji)

Bez Tesseracta pliki z tekstem działają w pełni, a dla pozostałych zostaje ręczne rysowanie ramek.

## Uruchomienie

```bash
python pdf_detail_splitter.py
```

## Kompilacja do .exe

```powershell
pyinstaller --onefile --windowed --name "PDF_Detail_Splitter" --clean ^
  --collect-all scipy --collect-all skimage --collect-all pytesseract ^
  --collect-all cv2 --collect-all PIL pdf_detail_splitter.py
```

## Wyniki na rzeczywistych plikach

| Typ pliku | Wykryte detale | Uwagi |
|---|---|---|
| Arkusz z warstwą tekstową (98 pozycji) | 49/49 | zero ramek nachodzących na siebie |
| Arkusz z numerami typu `3R-4001` (109 pozycji) | 109/109 | wszystkie wyeksportowane bez błędu |
| Arkusz bez tekstu (Tekla, ~10 pozycji) | 10/10 | geometria trafna, nazwy z OCR wymagają przejrzenia |

## Znane ograniczenia

- **Nazwy z OCR bywają niedokładne** przy plikach bez warstwy tekstowej – program pokazuje `det_N` zamiast zgadywać, gdy nie jest pewien odczytu. Geometria (granice wycinania) jest przy tym trafna, a nazwy poprawia się w podglądzie.
- Przy bardzo gęstych, wielokolumnowych arkuszach część ramek może wymagać ręcznej korekty krawędzi.
- Program celowo **nie wycina niczego bez Twojego zatwierdzenia** – podgląd jest obowiązkowym krokiem, bo to dane produkcyjne.

## Zależności

`PyMuPDF` (odczyt i zapis PDF), `opencv-python-headless` + `numpy` (przetwarzanie obrazu), `scipy` + `scikit-image` (segmentacja watershed), `pytesseract` (OCR), `Pillow` (podgląd).
