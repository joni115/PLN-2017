# PLN: Procesamiento de lenguaje natural 2017
---------------------------------------------
## Práctico 2
--------------------------------------------
#### Ejercicio 1

En este ejercicio se implemento eval.py. Allí teniamos que calcular las mediciones precision, recall y F1 (tanto para labeled e unlabeled). Además teníamos que limitar el número de oraciones (n) y el largo de las oraciones (m). Luego entrenamos y evaluamos los modelos “baseline” para todas las oraciones de largo menor o igual a 20.

Antes de mostrar resultados explicaremos sobre unlabeled e labeled. Además es imporante entender para que sirven las metricas precision, recall y F1.

Los parseos unlabeled son aquellos que no nos interesan los *tags*. En cambio los parseos labeled, no sólo interesa como esta parseado el árbol si no los tags de cada hoja.

*Precision* es la fracción de instancias recuperadas que son relevantes; *Recall* es la fracción de instancias relevantes que han sido recuperadas. Suponemos un programa donde identifica fotos de perros. Si tenemos 12 fotos donde 8 son de perros y el resto gatos, en el cual nuestro programa logra identificar sólo 5 fotos de perros, entonces *precision* sería 5/8 mientras que *recall* sería 5/12. *F1* combina *precision* y *recall*.



__Labeled__

|   Metric  |  Flat  | Rbranch | Lbranch |
|:---------:|:------:|:-------:|:-------:|
| Precision | 99.33% |  8.81%  |  8.81%  |
| Recall    | 14.58% | 14.58%  | 14.58%  |
|    F1     | 25.44% | 10.98%  | 10.98%  |


__Unlabeled__

|  Metric   |  Flat  | Rbranch | Lbranch |
|:---------:|:------:|:-------:|:-------:|
| Precision | 100%   |  8.88%  | 14.71%  |
| Recall    | 14.59% | 14.69%  | 24.35%  |
|    F1     | 25.46% | 11.07%  | 18.34%  |

__Tiempos__

| Metric | Tiempo(seg) |
|:------:|:-----------:|
|  Flat  |     6.85    |
|Rbranch |     7.35    |
|Lbranch |     7.59    |

Aquí podemos ver que los parseos son bastantes malos. Flat tiene un 100% de precision en unlabeled, pero gracias a la metrica F1 podemos ver que sigue siendo muy mala. Este 100% se debe a que el árbol se parsea (start_symbol, oración).

--------------------------------------------
#### Ejercicio 2

En este ejercicio se implemento CKY con backpointers (guardando el árbol).

Además se agrego a los test una gramática y una oración tal que la oración tenga más de un análisis posible (sintácticamente ambigua). Lo agregue en test_cky_parser.py

Oración: "the man saw the dog with telescope"

Es decir es ambiguo debido a que puede tener dos significados:

1. El hombre esta mirando al perro con un telescopio. Donde el árbol será el siguiente:

![Árbol coherente](graph1.png "Árbol 1")

2. El hombre esta mirando al perro, pero quien tiene el telescopio es el perro. El árbol será:

![Árbol sin coherencia](graph2.png "Árbol 2")

en test_cky_paser.py se puede ver la grámatica. Sólo basta cambiar las probabilidades de VP y da como resultante otro árbol.

--------------------------------------------
#### Ejercicio 3

Implementamos una UPCFG, es decir PCFG cuyas reglas y probabilidades se obtienen a partir de un corpus de entrenamiento. Luego entrenamos y evaluamos la UPCFG para todas las oraciones de largo menor o igual a 20.

Antes de exponer los resultados explicaremos que es una PCFG. Una PCFG es una gramatica libre de contexto (CFG) donde tiene un parametro *q* probabilistico, es decir para cada regla X -> Y perteneciente en la gramatica, *q* le asigna una probabilidad.

__Resultados__

|   Metric  |  Labaled  | Unlabeled |
|:---------:|:---------:|:---------:|
| Precision |   72.59%  |   74.76%  |
| Recall    |   72.44%  |   74.61%  |
|    F1     |   72.51%  |   74.69%  |

Claramente este parseador es mucho mejor que el del ejercicio1.

--------------------------------------------
#### Ejercicio 4

En este ejercicio se modifico la UPCFG para admitir el uso de Markovización Horizontal de orden n para un n dado (con el parámetro opcional horzMarkov). También agregamos al script de entrenamiento (train.py) una opción de línea de comandos que habilite esta funcionalidad. Por último entrenamos y evaluamos para varios valores de n (0, 1, 2 y 3).

__Labeled__

| n | Precision  | Recall |   F1   |
|:-:|:----------:|:------:|:------:|
| 0 |   69.77%   | 69.85% | 69.81% |
| 1 |   74.26%   | 74.27% | 74.27% |
| 2 |   74.60%   | 74.15% | 74.37% |
| 3 |   73.78%   | 73.15% | 73.46% |

__Unlabeled__

| n | Precision  | Recall |   F1   |
|:-:|:----------:|:------:|:------:|
| 0 |   71.67%   | 71.74% | 71.71% |
| 1 |   76.32%   | 76.33% | 76.33% |
| 2 |   76.61%   | 76.15% | 76.38% |
| 3 |   75.92%   | 75.27% | 75.59% |


__Time__

| n | Tiempo |
|:-:|:------:|
| 0 |  2m35s |
| 1 |  3m52s |
| 2 |  5m51s |
| 3 |  7m42s |
