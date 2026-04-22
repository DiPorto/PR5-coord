# Обробка координатних даних: Придушення шумів у потоці (Real-time)
**Автор:** Почка Дмитро ІПЗ - 4.02

## Мета роботи
Метою роботи є реалізація архітектури для потокової обробки координатних даних у режимі real-time, імплементація трьох цифрових фільтрів (`SMA`, `EMA`, `Median Filter`) та дослідження компромісу між ступенем згладжування і динамічними спотвореннями сигналу. Окрема увага приділяється спектральному аналізу помилки фільтрації. :contentReference[oaicite:1]{index=1}

## Використані інструменти
- Python
- Google Colab
- NumPy
- Matplotlib
- SciPy :contentReference[oaicite:2]{index=2}

---

## Реалізація

У роботі реалізовано три типи фільтрів для послідовної обробки координат у потоці даних:

- **SMA (Simple Moving Average)** — просте ковзне середнє
- **EMA (Exponential Moving Average)** — експоненційне ковзне середнє
- **Median Filter** — медіанний фільтр :contentReference[oaicite:3]{index=3}

### Реалізовані методи `update()`

```python
class SMAFilter:
    def __init__(self, w):
        self.w = w
        self.q = deque(maxlen=w)
        self.sum = 0.0

    def update(self, x):
        if len(self.q) == self.w:
            self.sum -= self.q[0]

        self.q.append(x)
        self.sum += x

        return self.sum / len(self.q)


class EMAFilter:
    def __init__(self, alpha):
        self.a = alpha
        self.last = None

    def update(self, x):
        if self.last is None:
            self.last = x
        else:
            self.last = self.a * x + (1 - self.a) * self.last

        return self.last


class MedianFilter:
    def __init__(self, w):
        if w % 2 == 0:
            w += 1
        self.w = w
        self.q = deque(maxlen=w)

    def update(self, x):
        self.q.append(x)
        return np.median(self.q)

```
# Графіки
## Експеримент 1. Базовий режим
Вхідні параметри
- W_SMA = 20
- A_EMA = 0.1
- W_MED = 21
- FS = 50 Гц
- DURATION = 20 с
- NOISE_STD = 0.8
- OUTLIER_PROB = 0.02
- OUTLIER_SCALE = 10.0

![Базовий експеримент](img/Figure_1.png)

## Експеримент 2. Екстремальне згладжування
Вхідні параметри
- W_SMA = 100
- A_EMA = 0.02
- W_MED = 21

![Екстремальний експеримент](img/Figure_2.png)

## Експеримент 3. Медіанний фільтр з малим вікном
Вхідні параметри
- W_SMA = 20
- A_EMA = 0.1
- W_MED = 5

![Екстремальний експеримент](img/Figure_3.png)
