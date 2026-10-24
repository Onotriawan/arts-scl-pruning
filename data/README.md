# ArTS dataset (not included)

   The Arabic Traffic Signs (ArTS) dataset was developed at Prince Mohammad Bin Fahd University and is available on Mendeley Data: https://data.mendeley.com/datasets/4tznkn45mx/1 (see also reference [40] in the paper). Please download it from this source and place it here:

```
data/ArabicTrafficSigns/
  train/                                           2,200 training images
  test/                                            518 test images
  Labels_Train_Arabic_Traffic_Signs_24_2200.csv    image name, class label (no header)
  Labels_Test_Arabic_Traffic_Signs_24_518.csv      image name, class label (no header)
```

A different location can be used with `--data-root <folder>`.

Class indices: 0 Front or Left, 1 Front or Right, 2 Hump, 3 Left Turn, 4 Narrows from Right, 5 Narrows from Left, 6 No Horn, 7 No Parking, 8 No U Turn, 9 Overtaking Forbidden, 10 Parking, 11 Pedestrian Crossing, 12 Right or Left, 13 Right Turn, 14 Rotor, 15 Slow, 16 Speed 100, 17 Speed 30, 18 Speed 40, 19 Speed 50, 20 Speed 60, 21 Speed 80, 22 Stop, 23 U Turn.
