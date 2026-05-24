# Änderungsvorschläge — Behauptung 2 / Konzept A (gleitendes Fenster)

Bezug: `WIMA_Masterarbeit_JeremiasLeserer.docx` (aktuelle Fassung, Stand 2026-05-22).

Geprüft gegen die aktuelle docx. Die Absatznummerierung hat sich gegenüber der
früheren Fassung verschoben (jetzt 949 Absätze) — die Stellenangaben unten sind
gegen die **aktuelle** Datei verifiziert.

---

## Hintergrund

Behauptung 2 betrifft die Aussage, das Verfahren sei „nicht als rollierende
Mehrfach-Fold-Variante mit wanderndem Trainingsfenster" umgesetzt.

**Konzept A (gleitendes Fenster der Eingabesequenzen):** bestätigt. Der Code
verwendet sehr wohl ein gleitendes Lookback-Fenster (`create_sequences`,
Schrittweite 1, T − L Sequenzen), und zwar mit identischem L in Trainings-,
Validierungs- und Test-Split. Die Bedingung „L gleich in Training, Validierung
und Test" ist damit erfüllt; das Experiment ist in diesem Punkt methodisch
korrekt.

**Problem im Text:** Die Formulierung „mit wanderndem Trainingsfenster" in
Abschnitt 4.4 liest sich so, als gäbe es gar kein gleitendes Fenster. Das ist
missverständlich. Die beiden folgenden Änderungen präzisieren das, ohne den
inhaltlichen Befund zu verändern.

(Konzept B — rollierende Mehrfach-Fold-Walk-Forward-Validierung — wird hier
bewusst nicht behandelt, da für den Fokus der Arbeit nicht relevant.)

---

## B1 · Präzisierung — Abschnitt 4.4 „Validierung und Hyperparameter-Tuning"

**Stelle:** Absatz **P0294** (5 markante Worte: „einmaliger chronologischer
Split umgesetzt, also").

**Bisher:**

> „In der vorliegenden Arbeit wird das Verfahren als einmaliger chronologischer
> Split umgesetzt, also nicht als rollierende Mehrfach-Fold-Variante mit
> wanderndem Trainingsfenster. Der einmalige Split hat den Vorteil eines
> deutlich reduzierten Rechenaufwands bei einem Suchraum von mehreren Tausend
> Trainingsläufen pro Index. Die Eigenschaft der Walk-Forward-Validierung, der
> vollständige Verzicht auf Look-Ahead-Bias, bleibt erhalten."

**Problem:** „mit wanderndem Trainingsfenster" liest sich, als gäbe es gar kein
gleitendes Fenster — es gibt aber sehr wohl ein gleitendes Lookback-Fenster der
Eingabesequenzen.

**Vorschlag:**

> „In der vorliegenden Arbeit wird das Verfahren als einmaliger chronologischer
> Hold-Out-Split umgesetzt, also nicht als rollierende Mehrfach-Fold-Variante,
> bei der ein Trainings- und Testfenster über mehrere Evaluationsrunden hinweg
> verschoben wird. Davon zu unterscheiden ist das innerhalb jedes Splits
> eingesetzte gleitende Lookback-Fenster der Eingabesequenzen (siehe
> nachfolgenden Abschnitt zur Sequenzbildung); dieses kommt in allen drei
> Splits — Training, Validierung und Test — gleichermaßen zum Einsatz. Der
> einmalige Hold-Out-Split hat den Vorteil eines deutlich reduzierten
> Rechenaufwands bei einem Suchraum von mehreren Tausend Trainingsläufen pro
> Index. Die zentrale Eigenschaft der Walk-Forward-Validierung, der vollständige
> Verzicht auf Look-Ahead-Bias, bleibt dabei erhalten."

**Anmerkung:** Der Begriff „Hold-Out-Split" ist konsistent mit Absatz P0289, der
ihn bereits verwendet.

---

## B2 · Neuer Absatz — Abschnitt 4.4, Sequenzbildung

**Stelle:** einzufügen **direkt nach Absatz P0299** (5 markante Worte:
„Eingabevektor X ∈ ℝ^L mit variabler").

**Kontext:** Absatz P0299 endet mit „… und ein Eingabevektor X ∈ ℝ^L mit
variabler Sequenzlänge L." — er beschreibt die Eingabedimension, aber nicht,
*wie* die Sequenzen aus der Reihe gebildet werden. Der neue Absatz schließt
diese Lücke und erklärt das gleitende Fenster explizit.

**Vorschlag (neuer Absatz):**

> „Die Bildung der Eingabesequenzen erfolgt nach dem Prinzip des gleitenden
> Fensters (Sliding Window). Aus einer Renditereihe r₁, …, r_T werden für ein
> Lookback-Fenster der Länge L alle Paare aus Eingabe und Ziel der Form
> ([r_{i−L}, …, r_{i−1}], r_i) gebildet, wobei das Fenster mit Schrittweite eins
> über die Reihe wandert. Eine Reihe der Länge T liefert auf diese Weise T − L
> Sequenzen. Dieses Verfahren wird mit identischem L auf den Trainings-, den
> Validierungs- und den Test-Split angewandt; die Sequenzlänge ist somit
> innerhalb eines Laufs in allen drei Phasen konsistent. Dadurch ist
> sichergestellt, dass die Bedingung des Lookback-Fensters für Training,
> Validierung und Test unter exakt gleichen Voraussetzungen gilt und kein Modell
> mit einer von seiner Trainingslänge abweichenden Eingabelänge bewertet wird."

**Code-Beleg:** `data_preparation/data_preparation.py`, Funktion
`create_sequences` (Schrittweite 1, T − L Sequenzen); Anwendung mit identischem
`lookback` auf alle drei Splits in `parameter_tuning/parameter_tuning.py`.

---

## Übersicht

| #  | Stelle (5 markante Worte)                    | Abschnitt              | Typ          |
|----|----------------------------------------------|------------------------|--------------|
| B1 | „einmaliger chronologischer Split umgesetzt, also" | 4.4 Validierung    | Absatz präzisieren |
| B2 | nach „Eingabevektor X ∈ ℝ^L mit variabler"   | 4.4 Sequenzbildung     | neuer Absatz |
