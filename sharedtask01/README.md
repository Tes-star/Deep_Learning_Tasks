# Shared Task #1 - Vehicle Type Classification
![car_showroom.png](car_showroom.png)

Es soll ein Neuronales Netz zur Klassifikation von Fahrzeugen in 9 Fahrzeugklassen entwickelt werden. Für diese Aufgabe
steht ein Datensatz zur Verfügung, welcher für jedes der 49502 Fahrzeuge 60 quantitative Attribute enthält.

## Daten

Es werden folgende Dateien zur Verfügung gestellt:

### Training

- train.csv: enthält Attribute und Zielattribute der Fahrzeuge, welche für das Modelltraining genutzt werden können
- sampleTest.csv: Beispieldatei im Format der Testdaten, welche in der Evaluationsphase klassifiziert werden sollen
- sampleSubmission.csv: Beispieldatei im Format der erwarteten Submission

### Evaluation

- test_x.csv: enthält Attribute der zu klassifizierenden Fahrzeuge
- test_y.csv: enthält korrekte Klassen für die zu klassifizierenden Fahrzeuge (wird nach dem Ende der Evaluationsphase
  veröffentlicht)

## Evaluationsmetrik

Die Bewertung erfolgt durch die Metrik "Accuracy".

Für dieses Dataset existiert kein sota. Eine nicht näher bezeichnete Testperson hat innerhalb weniger Minuten eine
Accuracy von 77 % erreicht. Dies kann als _grobe_ Baseline betrachtet werden.

## Important Dates

| Event                   | Date       |
|-------------------------|------------|
| Task Announcement       | 13.10.2022 |
| Training Data Release   | 13.10.2022 |
| Evaluation Data Release | 20.10.2022 |
| Submission Deadline     | 21.10.2022 |

## Allgemeine Submission Guidelines

Alle Studierenden sollen jeweils einzeln eine Lösung einreichen. Bewertungsrelevant ist das Evaluationsergebnis
bezüglich der definierten Metrik (unter Vorbehalt). Zusätzlich soll der Quelltext als gitlab-Projekt eingereicht werden,
damit die Erzeugung der Submission nachvollziehbar ist.  
