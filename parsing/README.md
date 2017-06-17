# PLN: Procesamiento de lenguaje natural 2017
---------------------------------------------
## Práctico 2
--------------------------------------------
#### Ejercicio 1

En este ejercicio se implemento eval.py. Allí teniamos que calcular las mediciones precision, recall y F1 (tanto para labeled e unlabeled). Además teníamos que limitar el número de oraciones (n) y el largo de las oraciones (m). Luego entrenamos y evaluamos los modelos “baseline” para todas las oraciones de largo menor o igual a 20.

Antes de mostrar resultados explicaremos sobre unlabeled e labeled. Además es imporante entender para que sirven las metricas precision, recall y F1.

Los parseos unlabeled son aquellos que no nos interesan las tags por lo contrario de las labeled.

*Precision* es la fracción de instancias recuperadas que son relevantes. Por otro lado, *Recall* es la fracción de instancias relevantes que han sido recuperadas. Por ejemplo si tenemos 12 fotos donde 8 son perros y el resto gatos. Suponemos que nuestro programa identifica 5 de los 8 perros, *precision* sería 5/8 mientras que *recall* sería 5/12. Luego *F1* combina *precision* y *recall*.



__Labeled__

|     \     |  Flat  | Rbranch | Lbranch |
|:---------:|:------:|:-------:|:-------:|
| Precision | 99.33% |  8.81%  |  8.81%  |
| Recall    | 14.58% | 14.58%  | 14.58%  |
|    F1     | 25.44% | 10.98%  | 10.98%  |


__Unlabeled__

|     \     |  Flat  | Rbranch | Lbranch |
|:---------:|:------:|:-------:|:-------:|
| Precision | 100%   |  8.88%  | 14.71%  |
| Recall    | 14.59% | 14.69%  | 24.35%  |
|    F1     | 25.46% | 11.07%  | 18.34%  |

_Tiempos_

|   \    | Tiempo(seg) |
|:------:|:-----------:|
|  Flat  |     6.85    |
|Rbranch |     7.35    |
|Lbranch |     7.59    |