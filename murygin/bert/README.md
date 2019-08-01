* Change serving path in function "export\_saved\_model"
* Comment or not functions "create\_training\_graph" and "create\_eval\_graph"
* If "create\_training\_graph" isn't commented than you can set second argument "quant\_delay"
* Run "run\_bert\_nlu.py" as it described in comments in the beginning of file
* Run "tflite\_convert --output\_file=./bert.tflite --saved\_model\_dir=PATH\_TO\_SAVEDMODEL"
