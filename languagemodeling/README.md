PLN 2017: Práctico 1.
=====================

Ejercicio1
----------
El corpus que elegí fue los libros de games of thrones. Los primeros 3 libros lo usaré para el entrenamiento de mi modelo, el cuarto uso 3/4 de libro para usarlo como corpus de test.
Para cargar el corpus utilize el "corpus reader" de NTLK (PlaintextCorpusReader) donde tiene un método sents que permite iterar sobre las oraciones tokenizadas del corpus.
Se debe crear el directorio "corpus" dentro de /languagemodeling/scripts para colocar los corpus de entrenamiento y de test. En la interfaz de train.py agregue un argumento opcional para ingresar el nombre del archivo de nuestro corpus, en caso contrario buscará el archivo 'corpus_GOT123_train.txt'.

Ejercicio2
----------

Aquí se nos pidió implementar la clase NGram, donde es un modelo de n-gramas. A continuación explicaremos un poco sobre los métodos modificados.
- \__init__: este método ya estaba implementado. Lo que modifique fue que en cada frase agregue n-1 <s> y 1 </s> para evitar el underflow. Además guardo la cantidad de wordtypes y una lista de tokens para usar en un futuro (consideré que sería mejor hacer más costosa la inicializacización de NGram para luego simplificar implementaciones y reducir costos de otros ejercicios).
- count: cuenta la cantidad de tokens, para ello utilizamos la estructura counts inicializada en init.
- cond_prob: método ya implementado. La única modificación es que si los tokens previos no están en el corpus la probabilidad es 0.
- sent_prob: es la productoria de las probabilidades de las frases. Debido a la multiplicación se puede producir un underflow, por ello sería mejor utilizar sent_log_probability.
- sent_log_probability: calcula el logaritmo de la probabilidad de una frase utilizando una propiedad mátematica .i.e.
    log (s1 * s2 * ... * sn) = log(s1) + log(s2) + ... + log(s3)

Ejercicio3
----------
Este ejercicio consistía en implementar un generador de frases. Para ello tuvimos que crear dos nuevas estructuras "probs" y "sorted_probes". La segunda ordenadara las probabilidades de la primera. "probs" es un diccionario donde cada clave es una n-tupla y sus elementos son probabilidades i.e. si tenemos
probs = {('como','hola'): {'andas': 0.5, 'estas': 0.5}} quiere decir P(andas | hola como) = 0.5.

Algunas alcaraciones sobre los métodos:
- El método generate_token utiliza "sorted_probs" (que será ordenada de menor a mayor) para facilitar la implementación el método de la transformada inversa. En este método generamos un token de acuerdo a los anteriores.

- generate_sent utiliza el método generate_token para generar cada palabra y así generamos la frase entera hasta que encontremos un final de frase (</s>).

Creamos el script generate.py donde carga un modelo y genera las oraciones. Debido a que quería hacer las cuatro frases automaticamente hice opcional el argumento de language model file. Si no  ingresamos un archivo del modelo, el script generará las oraciones de los n-gramas (n = {1, 2, 3, 4}) automaticamente. Para poder generarlas necesitamos modelos ya entrenados guardados en /languagemodeling/script (el train.py guarda los modelos en esta carpeta por defecto). El output será en un archivo output.txt generado en un directorio output dentro de script.

#### Oraciones generadas:
##### UNIGRAMA
>pesaba en delante de botas Los dolor fervorosamente le lado se otro junto ribera pero más . alzó los quesido muy cincopoco sureño lo una existido , el tenía una justo suelo incluso había instante ellos en la retomaron crepúsculo quétenía , sucio Cuando demasiado huevos en huellas monstruo el, dilo el ; ; detenerme ocupar a como solo honor vez Qué

##### BIGRAMA
> En el encargado de todo lo habían desaparecido , Arya cabalgó sobre Invernalia y se acercaban tenía la vez que estar durmiendo sobre el veneno , le ocultaba la puerta .Hace catorce años ?No sintió que el Forca Verde , el camino .La próxima batalla .Cuando salió de Robett Glover tenía una carcajada amarga .

#### TRIGRAMA
> No soy una persona se atrevería a recomendaros a la Guardia Real como escolta , por qué no ?Sí , debéis retroceder , el prisionero de Robb dijo Catelyn .Pues resulta que no ha leído tu cartita .Qué agradable sería despertar cada mañana sin saber bien por su hermano también es una cosa , ser lo corrigió Dany .El hijo de su pueblo .

#### CUATRIGRAMA
> Más que Joffrey .Oh , no me deja en paz ?Ríete lo que quieras a que te salgan alas .Viserys estaba furioso .Cuando Harwin le quitó la corona y doblar la rodilla .

Ejercicio 4
-----------
Está clase hereda NGram, ya que es prácticamente lo mismo salvo un métodos modificado y otro nuevo. Gracias a que en la clase NGram guardamos la cantidad de word_types la implementación de la clase es simple.

- cond_probability: esté es un método sobre-escrito de NGram. La únca diferencia con cond_probability de NGram es que sumamos uno al numerador y V (cantidad de word types) al denominador.

Además se agrego al script train.py la posibilidad de entrenar un modelo addOneGram para luego estudiar su comportamiento

Ejercicio 5
-----------
Como había mencionado al principio, utilice los libros primero, segundo y tercero de games of thrones para entrenar el modelo y una parte del cuarto libro para testearlo. Luego se programó el script eval.py para evaluar un modelo sobre el conjunto de test, calculando la perplejidad. En este script agregue un argumento opcional para ingresar el corpus de test.
Aquí esta el resultado:

| n | n-gram |   Addone   |
|:-:|:------:|:----------:|
| 1 | inf    | 1270.49    |
| 2 | inf    | 2846.63    |
| 3 | inf    | 20012.06   |
| 4 | inf    | 35313.75   |

Claramente no era lo que esperaba. Son valores muy altos y addone ayudo bastante contra eso. Pero podemos ver que en n's más grandes la perplejidad baja... no es un comportamiento deseable.
