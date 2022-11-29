# Shared Task #4 - German Named Entity Recognition

Es soll ein Neuronales Netz für Named Entity Recognition entwickelt werden. 
Ziel des Modells ist es, in deutschsprachigen Daten Entitäten der drei Typen Person, Organization und Place zu erkennen.
Es darf ein vortrainiertes Language Model verwendet werden.

## Daten

Es werden folgende Dateien zur Verfügung gestellt:

### Training
Es werden Daten der [GermEval2014](https://drive.google.com/drive/folders/1kC0I2UGl2ltrluI9NqDjaQJGw5iliw_J) genutzt.
Für diesen Shared Task sollen nur die Typen PER, ORG und LOC betrachtet werden. Andere Typen werden ignoriert. Der Datensatz nutzt das BIO Encoding Schema.

- train.tsv: Enthält Trainingssätze mit zwei Entitäten-Ebenen. Jeder Satz enthält eine Kommentarzeile mit dem Ursprung des Satzes. Danach folgen in nummerierten Zeilen jeweils ein Wort inklusive zweier Entitätenlabels.  
- val.tsv: Enthält Validationsätze im gleichen Format.
- sampleTest.tsv: Beispieldatei im Format der Testdaten, welche in der Evaluationsphase verarbeitet werden sollen.
- sampleSubmission.tsv: Beispieldatei im Format der erwarteten Submission. Für jedes Wort soll ein Label vorhergesagt werden. Es soll nur eine Entitätenebene vorhergesagt werden.

### Evaluation

- test.tsv: Enthält zu verarbeitende Testsätze.
- test_y.tsv: Enthält Testsätze inklusive der NER Labels je Wort (wird nach dem Ende der Evaluationsphase veröffentlicht).

## Evaluationsmetrik

Die Bewertung erfolgt durch die Metrik "F1-Score". Der F1-Score wird für jeden der drei Zieltypen separat berechnet. 
Das (ungewichtete) arithmetische Mittel dieser Scores bildet die Gesamtbewertung. 

Aktuelle Modelle erreichen F1-Scores im Bereich um 85%. Eine weitere Baseline zur Orientierung wird zeitnah nachgeliefert.

## Important Dates

| Event                   | Date       |
|-------------------------|------------|
| Task Announcement       | 24.11.2022 |
| Training Data Release   | 24.11.2022 |
| Evaluation Data Release | 07.12.2022 |
| Submission Deadline     | 08.12.2022 |

## Allgemeine Submission Guidelines

Alle Studierenden sollen jeweils einzeln eine Lösung über Moodle einreichen.
Bewertungsrelevant ist das Evaluationsergebnis
bezüglich der definierten Metrik (unter Vorbehalt). Zusätzlich soll der
Quelltext als gitlab-Projekt eingereicht werden,
damit die Erzeugung der Submission nachvollziehbar ist.  
