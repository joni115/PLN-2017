# PLN: Procesamiento de lenguaje natural 2017
---------------------------------------------
## Práctico 2
--------------------------------------------
#### Ejercicio 1

En este ejercicio se nos pedía calcular las estadisticas en un corpus, para ello implementamos stats.py. El corpus usado es ancora-3.0.1 que es pasado por la cátedra.

__Basic Statics__

Sents: 17378
Total words: 517194
Vocabulary words: 46501
Vocabulary taggs: 85
________________________________________________________________________________
__Frequencies Taggs__

  | Tagg    | Frequencie  |Percent  |     Words |
  |:-------:|:-----------:|:-------:|:---------:|
  | sp000   |    79884    |15.45%   |'de','en','a','del','con' |
  | nc0s000 |     63452   |  12.27% | 'presidente','equipo','partido','país','año' |
  |da0000   |    54549    | 10.55%  |'la','el','los','las','El' |
  |aq0000   |    33906    | 6.56%   |'pasado','gran','mayor','nuevo','próximo' |
  |  fc     |    30147    | 5.83%   |',' |
  |np00000  |    29111    | 5.63%   |'Gobierno','España','PP','Barcelona','Madrid' |
  |nc0p000  |    27736    | 5.36%   |'años','millones','personas','países','días' |
  |  fp     |    17512    | 3.39%   |'.'' |
  |  rg     |    15336    | 2.97%   |'más','hoy','también','ayer','ya' |
  |  cc     |    15023    | 2.90%   |'y','pero','o','Pero','e' |
________________________________________________________________________________
__Ambiguity__
| Level  |   Amount    | Percent |    Words  |
|:------:|:-----------:|:-------:|:---------:|
|    1   |     43972   | 94.56%  |',','con','por','su','El'|
|    2   |      2318   |  4.98%  |'el','en','y','"','los'|
|    3   |       180   |  0.39%  |'de','la','.','un','no'|
|    4   |        23   |  0.05%  |'que','a','dos','este','fue'|
|    5   |         5   |  0.01%  |'mismo','cinco','medio','ocho','vista'|
|    6   |         3   |  0.01%  |'una','como','uno'|
|    7   |         0   |  0.00%  |'-'|
|    8   |         0   |  0.00%  |'-'|
|    9   |         0   |  0.00%  |'-'|

A continuación daré el significado de las 10 tags más frecuentes:
1. sp000: Preposición.
2. nc0s000: Sustantivo común singular.
3. da0000: Artículo.
4. aq0000: Adjetivo descriptivo.
5. fc: Coma.
6. np00000: Sustantivo propio.
7. nc0p000: Sustantivo común plural
8. fp: Punto.
9. rg: Advervio general.
10. cc: Conjunción.

#### Ejercicio2

Se implemento la clase baseline, en el cual se elige para cada palabra su etiqueta más frecuente observada en entrenamiento. Para las palabras desconocidas, devolver la etiqueta 'nc0s000'.

#### Ejercicio3

En este ejercicio entrenamos un modelo baseline. Luego evaluamos el porcentaje de etiquetas correctas, sobre las palabras conocidas y sobre las palabras desconocidas (accuracy). Además graficamos una matriz de confusión para ver que tan bueno es el etiquetado.

Nota: la matriz de confusión se hará sobre las 10 tags más vistas para mejor comprensión.

__Accuracy__: 87.59%

__Accuracy known words__:95.27%

__Acurracy unknown words__:18.01%

**Confusion matrix**

|label/label|  sp000  | nc0s000 | da0000  | aq0000  |   fc    | nc0p000 |   rg    | np00000 |   fp    |   cc    |
|:---------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|  sp000    |14.28375 | 0.04749 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00528 | 0.00000 | 0.00000 | 0.00000 |
| nc0s000   | 0.00211 |12.22060 | 0.00000 | 0.24061 | 0.00000 | 0.00106 | 0.03271 | 0.00106 | 0.00000 | 0.00106 |
|  da0000   | 0.00000 | 0.15091 | 9.54326 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
|  aq0000   | 0.00528 | 2.05049 | 0.00000 | 4.84814 | 0.00000 | 0.12136 | 0.00317 | 0.00000 | 0.00000 | 0.00000 |
|    fc     | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 5.84964 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| nc0p000   | 0.00000 | 1.23683 | 0.00000 | 0.21001 | 0.00000 | 4.07353 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
|    rg     | 0.01794 | 0.31449 | 0.00000 | 0.03166 | 0.00000 | 0.00000 | 3.28310 | 0.00000 | 0.00000 | 0.02216 |
| np00000   | 0.00317 | 2.03888 | 0.00000 | 0.00106 | 0.00000 | 0.00317 | 0.00000 | 1.52283 | 0.00000 | 0.00106 |
|    fp     | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 3.55010 | 0.00000 |
|    cc     | 0.00106 | 0.01372 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.04854 | 0.00106 | 0.00000 | 3.34114 |

Este modelo es bastante bueno si los textos de entrenamientos son "parecidos" a los textos de predicción debido a la gran acurracy de las palabras conocidas y a la baja acurracy de las palabras desconocidas. Es el modelo más rápido de este proyecto. Con "parecidos" me refiero a que los textos a predecir no tienen muchas palabras desconocidas.

#### Ejercicio4

En este ejercicio se implemento Hidden Markov Models y Algoritmo de Viterbi. Para implementar el algoritmo de viterbi utilizamos programación dinamica por lo tanto tuvimos que hacer el algoritmo iterativo.

#### Ejercicio5

En este ejercio se implemento una clase MLHMM un Hidden Markov Model cuyos parámetros se estiman usando Maximum Likelihood sobre un corpus de oraciones etiquetado. La clase debe tiene la misma interfaz que HMM con algunas modificaciones. Además se agrego una nueva opción en el training para entrenar modelos de MLHMMM.

Los resultados son los siguientes:

| n | etiquetas correctas | etiquetas conocidas correctas | etiquetas desconocidas correctas | tiempo |
|:-:|:-------------------:|:-----------------------------:|:--------------------------------:|:------:|
| 1 |       85.84%        |           95.28%              |           0.45%                  |0m21.41s|
| 2 |       91.34%        |           97.63%              |           34.33%                 |1m7.848s|
| 3 |       91.86%        |           97.65%              |           39.49%                 |3m24.76s|
| 4 |       91.61%        |           97.31%              |           40.01%                 |25m41.1s|


Notar que estos modelos son mejores para etiquetas desconocidas que el baseline. Además es importante ver que en n igual a 3 y 4 no hay mucha diferencia. En n=1 es claro que el acurracy es demasiado bajo, por lo que conviene usar con n > 2 (e=2 si los textos que queremos predecir no tienen muchas palabras desconocidas).

#### Ejercicio6
Implementamos features. Tanto básicos como paramétricos.

Los básicos son:
1. word_lower: la palabra actual en minúsculas.
2. word_istitle: la palabra actual empieza en mayúsculas.
3. word_isupper: la palabra actual está en mayúsculas.
4. word_isdigit: la palabra actual es un número.

Por otro lado en los paramétricos tenemos:
1. NPrevTags(n): la tupla de los últimos n tags.
2. PrevWord(f): Dado un feature f, aplicarlo sobre la palabra anterior en lugar de la actual.

#### Ejercicio7

En este ejercicio implementamos la clase MEMM en el cual modela a Maximum Entropy Markov Models.

Recordemos que un memm es (por definición):

* Conjunto de palabras V.
* Conjunto finito de tags T.
* Conjunto de hisotrias H.
* Un entero d que identifica el número de __features__.
* Una función f:HxK -> R^d donde especifica los features del modelo.
* Parametro p y theta.

Luego podremos etiquetar una oración como un hmm pero con features.
A continuación algunos resultados:

__Logistic Regression__


| n | etiquetas correctas | etiquetas conocidas correctas | etiquetas desconocidas correctas | tiempo |
|:-:|:-------------------:|:-----------------------------:|:--------------------------------:|:------:|
| 1 |       91.10%        |           94.55%              |           59.84%                 | 34.99s |
| 2 |       90.70%        |           94.17%              |           59.32%                 | 34.78s |
| 3 |       90.87%        |           94.24%              |           60.42%                 | 39.79s |
| 4 |       90.87%        |           94.23%              |           60.47%                 |48.603s |


__Linear SVC__

| n | etiquetas correctas | etiquetas conocidas correctas | etiquetas desconocidas correctas | tiempo |
|:-:|:-------------------:|:-----------------------------:|:--------------------------------:|:------:|
| 1 |       93.59%        |           97.11%              |           61.74%                 |27.958s | 
| 2 |       93.55%        |           97.04%              |           61.98%                 |39.082s |
| 3 |       93.68%        |           97.10%              |           62.73%                 |39.319s |
| 4 |       93.69%        |           97.13%              |           62.54%                 |43.904s |


__Multinomial NB__

| n | etiquetas correctas | etiquetas conocidas correctas | etiquetas desconocidas correctas | tiempo |
|:-:|:-------------------:|:-----------------------------:|:--------------------------------:|:------:|
| 1 |       77.02%        |           81.47%              |           36.72%                 |69m40.2s|
| 2 |       71.25%        |           75.48%              |           33.00%                 |90m54.1s|
| 3 |       66.43%        |           70.28%              |           31.62%                 |60m23.4s|
| 4 |       63.52%        |           66.92%              |           32.77%                 |62m2.26s|




SVC tuvo mejor desempeño tanto en acurracy como en tiempo.