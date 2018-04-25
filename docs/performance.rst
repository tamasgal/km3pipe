Performance Analysis
====================


.. contents:: :local:


Introduction
------------
Coming soon.


Time / Memory / CPU Usage 
--------------------------

Pass the `Pipeline(timeit=True)` argument to show detailed time and CPU
 usage for every module in the pipeline, and overall memory::

  Pipeline and module initialisation took 0.015s (CPU 0.013s).
  --------------------------[ Blob     100 ]---------------------------
  --------------------------[ Blob     200 ]---------------------------
  --------------------------[ Blob     300 ]---------------------------
  --------------------------[ Blob     400 ]---------------------------
  --------------------------[ Blob     500 ]---------------------------
  ================================[ . ]================================
  ============================================================
  500 cycles drained in 6.592123s (CPU 6.513101s). Memory peak: 312.91 MB
    wall  mean: 0.013066s  medi: 0.012130s  min: 0.004782s  max: 0.114435s  std: 0.007302s
    CPU   mean: 0.012913s  medi: 0.012014s  min: 0.004772s  max: 0.110865s  std: 0.007137s
  HDF5Pump - process: 2.510s (CPU 2.456s) - finish: 0.003s (CPU 0.003s)
    wall  mean: 0.005021s  medi: 0.004509s  min: 0.003390s  max: 0.038976s  std: 0.002127s
    CPU   mean: 0.004912s  medi: 0.004434s  min: 0.003382s  max: 0.035576s  std: 0.001951s
  StatusBar - process: 0.001s (CPU 0.001s) - finish: 0.000s (CPU 0.000s)
    wall  mean: 0.000205s  medi: 0.000223s  min: 0.000128s  max: 0.000280s  std: 0.000065s
    CPU   mean: 0.000206s  medi: 0.000223s  min: 0.000129s  max: 0.000282s  std: 0.000066s
  filter_muons - process: 0.059s (CPU 0.059s) - finish: 0.000s (CPU 0.000s)
    wall  mean: 0.000119s  medi: 0.000104s  min: 0.000091s  max: 0.000326s  std: 0.000038s
    CPU   mean: 0.000119s  medi: 0.000104s  min: 0.000091s  max: 0.000327s  std: 0.000037s
  DOMHits - process: 3.839s (CPU 3.821s) - finish: 0.038s (CPU 0.038s)
    wall  mean: 0.007678s  medi: 0.007034s  min: 0.000944s  max: 0.075129s  std: 0.005830s
    CPU   mean: 0.007642s  medi: 0.007021s  min: 0.000945s  max: 0.074965s  std: 0.005801s
